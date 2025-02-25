

#include "Compack3D_tri.h"
#include <stdexcept>
#include <mpi.h>
#include <chrono>
#include <cmath>

#define COEF_ALP ( 1.0 / 3.0      )
#define COEF_A   (14.0 / 9.0 / 2.0)
#define COEF_B   ( 1.0 / 9.0 / 4.0)
#define DOM_LEN   6.283185307179586
#define K_WAVE    2.0

using namespace cmpk;
using namespace tri;

typedef double Real;
typedef MemSpace::Host mem_space;

template<typename RealType> void setRHS(RealType*, const int, const int, const int);
template<typename RealType> RealType checkSoln(RealType*, const int, const int, const int, MPI_Comm);
template<typename MemSpaceType, typename RealType, typename FuncType>
void defFactPartitionedTri(RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType&, RealType&, const int, const int, const int, const int, MPI_Comm, FuncType);

int main(int argc, char* argv[]) {
    (void) argc; (void) argv;

    MPI_Init(NULL, NULL);

    // Basic configuration
    constexpr int N_sub = 64;
    constexpr int max_local_fact_size = 128;
    int Np, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &Np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int start_idx = rank * N_sub;
    const int N_tot     = Np   * N_sub;


    //  Set system
    Real* fact_local_prev = nullptr;
    Real* fact_local_curr = nullptr;
    Real* fact_local_next = nullptr;
    Real* fact_dist_prev  = nullptr;
    Real* fact_dist_curr  = nullptr;
    Real* fact_dist_next  = nullptr;
    Real* Si              = nullptr;
    Real* Ri              = nullptr;
    Real  Li_tilde_tail   = 0.00000;
    Real  Ui_tilde_head   = 0.00000;

    allocFactBuffers<mem_space, Real>(
            &fact_local_prev, &fact_local_curr, &fact_local_next,
            &fact_dist_prev, &fact_dist_curr, &fact_dist_next, &Si, &Ri,
            N_sub, Np);

    auto time_start = std::chrono::high_resolution_clock::now();
    defFactPartitionedTri<mem_space, Real>(
            fact_local_prev, fact_local_curr, fact_local_next,
            fact_dist_prev, fact_dist_curr, fact_dist_next,
            Si, Ri, Li_tilde_tail, Ui_tilde_head,
            start_idx, N_sub, Np, max_local_fact_size, MPI_COMM_WORLD, [=](const int row, const int offset)->Real {
                (void) row;
                if (offset == 0)
                    return 1.00;
                else if (offset == 1 || offset == -1)
                    return COEF_ALP;
                return 0.0;
            });
    auto time_end = std::chrono::high_resolution_clock::now();

    if (rank == 0) printf("Time used for factorization %.6f msec.\n", std::chrono::duration<double,  std::milli>(time_end - time_start).count());

    Real* x = new Real [N_tot];
    setRHS<Real>(x, start_idx, N_sub, N_tot);
    ////////////////////////////////////
    //---- START SOLUTION PROCESS ----//
    const int N_local = N_sub - 1;
    Real* yi = &x[1];
    vanillaLocalSolTriPCR(yi, fact_local_prev, fact_local_curr, fact_local_next, N_local, max_local_fact_size);
    Real y_prev;
    Real y_bot = yi[N_local - 1];
    MPI_Request mpi_reqs [2];
    MPI_Status  mpi_stats[2];
    MPI_Irecv(&y_prev, 1, MPIDataType<Real>::value, (rank - 1 + Np) % Np, 1, MPI_COMM_WORLD, &mpi_reqs[0]);
    MPI_Isend(&y_bot , 1, MPIDataType<Real>::value, (rank + 1     ) % Np, 1, MPI_COMM_WORLD, &mpi_reqs[1]);
    MPI_Waitall(2, mpi_reqs, mpi_stats);

    // Assemble b_hat
    x[0] -= Li_tilde_tail * y_prev + Ui_tilde_head * yi[0];


    // Solve reduced system
    Real x_prev_buf = 0.0;
    Real x_curr_buf = 0.0;
    Real x_next_buf = 0.0;
    vanillaDistSolve<Real, Real> (x[0], x_prev_buf, x_curr_buf, x_next_buf, fact_dist_prev, fact_dist_curr, fact_dist_next, Np, MPI_COMM_WORLD);

    Real x_tilde_next;
    MPI_Irecv(&x_tilde_next, 1, MPIDataType<Real>::value, (rank + 1     ) % Np, 1, MPI_COMM_WORLD, &mpi_reqs[0]);
    MPI_Isend( x           , 1, MPIDataType<Real>::value, (rank - 1 + Np) % Np, 1, MPI_COMM_WORLD, &mpi_reqs[1]);
    MPI_Waitall(2, mpi_reqs, mpi_stats);

    for (int i = 0; i < N_local; i++) {
        yi[i] -= Si[i] * x[0] + Ri[i] * x_tilde_next;
    }
    //---- END OF SOLUTION PROCESS ----//
    /////////////////////////////////////

    Real err = checkSoln(x, start_idx, N_sub, N_tot, MPI_COMM_WORLD);
    if (rank == 0) printf("Err = %12.5e\n", err);
    delete [] x;

    freeFactBuffers<mem_space>(
            fact_local_prev, fact_local_curr, fact_local_next,
            fact_dist_prev, fact_dist_curr, fact_dist_next, Si, Ri);

    MPI_Finalize();
    return 0;
}



template<typename RealType>
void setRHS(RealType* x, const int start_idx, const int N_sub, const int N_tot) {
    const RealType dx = DOM_LEN / N_tot;
    for (int i_sub = 0; i_sub < N_sub; i_sub ++) {
        const int i = i_sub + start_idx; 
        x[i_sub] = (COEF_A * (sin(K_WAVE * (i + 1) * dx) - sin(K_WAVE * (i - 1) * dx))
                 +  COEF_B * (sin(K_WAVE * (i + 2) * dx) - sin(K_WAVE * (i - 2) * dx))) / dx;
    }
}



template<typename RealType>
RealType checkSoln(RealType* x, const int start_idx, const int N_sub, const int N_tot, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    const RealType dx = DOM_LEN / N_tot;
    const Real K_WAVE_MOD = (2.0 * COEF_A * sin(K_WAVE * dx) + 2.0 * COEF_B * sin(2.0 * K_WAVE * dx)) / dx
                          / (1.0 + 2.0 * COEF_ALP * cos(K_WAVE * dx));
    Real err_loc = 0.0;
    //printf("%3s, %12s, %12s # k_mod = %.5e\n", "idx", "x_ref", "x_num", K_WAVE_MOD);
    for (int i_sub = 0; i_sub < N_sub; i_sub ++) {
        const int i = i_sub + start_idx; 
        const Real x_ref = K_WAVE_MOD * cos(K_WAVE * i * dx);
        Real err_loc_i = x_ref - x[i_sub];
        err_loc += err_loc_i * err_loc_i;
        if (fabs(err_loc_i) > 1e-13) printf("idx=%3d, x_ref=%12.5e, x_num=%12.5e\n", i, x_ref, x[i_sub]);
    }
    Real err;
    MPI_Allreduce(&err_loc, &err, 1, MPIDataType<Real>::value, MPI_SUM, comm);
    err = sqrt(err / N_tot);
    return err;
}



template<typename MemSpaceType, typename RealType, typename FuncType>
void defFactPartitionedTri(
        RealType* fact_local_prev, RealType* fact_local_curr, RealType* fact_local_next,
        RealType* fact_dist_prev, RealType* fact_dist_curr, RealType* fact_dist_next, RealType* Si, RealType* Ri, RealType& Li_tilde_tail, RealType& Ui_tilde_head,
        const int start_idx, const int N_sub, const int Np, const int max_local_fact_size, MPI_Comm comm_sub, FuncType sys_def
) {
    static_assert(std::is_same<MemSpaceType, MemSpace::Host>::value || std::is_same<MemSpaceType, MemSpace::Device>::value, "Valid MemSpace typenames are MemSpace::Host and MemSpace::Device.");
    RealType* fact_local_prev_host = nullptr;
    RealType* fact_local_curr_host = nullptr;
    RealType* fact_local_next_host = nullptr;
    RealType* fact_dist_prev_host  = nullptr;
    RealType* fact_dist_curr_host  = nullptr;
    RealType* fact_dist_next_host  = nullptr;
    RealType* Si_host              = nullptr;
    RealType* Ri_host              = nullptr;

    if constexpr (std::is_same<MemSpaceType, MemSpace::Device>::value) {
        const int local_fact_size = (N_sub - 1) * log2Ceil(N_sub - 1);
        fact_local_prev_host = new RealType [local_fact_size];
        fact_local_curr_host = new RealType [local_fact_size];
        fact_local_next_host = new RealType [local_fact_size];
        const int dist_fact_size = numDistSolSteps<int>(Np);
        fact_dist_prev_host = new RealType[dist_fact_size];
        fact_dist_curr_host = new RealType[dist_fact_size];
        fact_dist_next_host = new RealType[dist_fact_size];
        const int local_soln_size = N_sub - 1;
        Si_host = new RealType[local_soln_size];
        Ri_host = new RealType[local_soln_size];
    } else {
        fact_local_prev_host = fact_local_prev;
        fact_local_curr_host = fact_local_curr;
        fact_local_next_host = fact_local_next;
        fact_dist_prev_host  = fact_dist_prev;
        fact_dist_curr_host  = fact_dist_curr;
        fact_dist_next_host  = fact_dist_next;
        Si_host              = Si;                     
        Ri_host              = Ri;                     
    }

    RealType* part_L = new RealType [N_sub];
    RealType* part_D = new RealType [N_sub];
    RealType* part_U = new RealType [N_sub];

    for (int j = 0; j < N_sub; j++) {
        const int idx = j + start_idx;
        part_L[j] = sys_def(idx, -1);
        part_D[j] = sys_def(idx,  0);
        part_U[j] = sys_def(idx,  1);
    }

    factPartitionedTriHost(
        fact_local_prev_host, fact_local_curr_host, fact_local_next_host,
        fact_dist_prev_host, fact_dist_curr_host, fact_dist_next_host, Si_host, Ri_host,
        part_L, part_D, part_U,
        N_sub, Np, max_local_fact_size, comm_sub
    );

    delete [] part_L;
    delete [] part_D;
    delete [] part_U;

    Li_tilde_tail = sys_def(start_idx,-1);
    Ui_tilde_head = sys_def(start_idx, 1);


    if constexpr (std::is_same<MemSpaceType, MemSpace::Device>::value) {
       #ifdef COMPACK3D_DEVICE_ENABLED_CUDA
        const size_t local_fact_size = (N_sub - 1) * log2Ceil(N_sub - 1) * sizeof(RealType);
        cudaMemcpy((void*)fact_local_prev, (void*)fact_local_prev_host, local_fact_size, cudaMemcpyHostToDevice);
        cudaMemcpy((void*)fact_local_curr, (void*)fact_local_curr_host, local_fact_size, cudaMemcpyHostToDevice);
        cudaMemcpy((void*)fact_local_next, (void*)fact_local_next_host, local_fact_size, cudaMemcpyHostToDevice);
        const size_t dist_fact_size = numDistSolSteps<size_t>(Np) * sizeof(RealType);
        cudaMemcpy((void*)fact_dist_prev, (void*)fact_dist_prev_host, dist_fact_size, cudaMemcpyHostToDevice);
        cudaMemcpy((void*)fact_dist_curr, (void*)fact_dist_curr_host, dist_fact_size, cudaMemcpyHostToDevice);
        cudaMemcpy((void*)fact_dist_next, (void*)fact_dist_next_host, dist_fact_size, cudaMemcpyHostToDevice);
        const size_t local_soln_size = (N_sub - 1) * sizeof(RealType);
        cudaMemcpy((void*)Si, (void*)Si_host, local_soln_size, cudaMemcpyHostToDevice);
        cudaMemcpy((void*)Ri, (void*)Ri_host, local_soln_size, cudaMemcpyHostToDevice);
       #endif
        delete [] fact_local_prev_host;
        delete [] fact_local_curr_host;
        delete [] fact_local_next_host;
        delete [] fact_dist_prev_host;
        delete [] fact_dist_curr_host;
        delete [] fact_dist_next_host;
        delete [] Si_host;
        delete [] Ri_host;
        fact_local_prev_host = nullptr;
        fact_local_curr_host = nullptr;
        fact_local_next_host = nullptr;
        fact_dist_prev       = nullptr;
        fact_dist_curr       = nullptr;
        fact_dist_next       = nullptr;
        Si_host              = nullptr;
        Ri_host              = nullptr;
    } 
}
