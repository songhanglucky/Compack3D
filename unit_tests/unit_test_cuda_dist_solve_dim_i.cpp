#include "Compack3D_penta.h"
#include <stdexcept>
#include <mpi.h>
#include <chrono>
#include <cmath>

#define COEF_ALP  0.50
#define COEF_BET  0.05
#define COEF_A   ( 17.0 /  12.0 / 2.0)
#define COEF_B   (101.0 / 150.0 / 4.0)
#define COEF_C   (  1.0 / 100.0 / 6.0)
#define DOM_LEN   6.283185307179586
#define K_WAVE    2.0

#ifdef COMPACK3D_DEVICE_ENABLED_CUDA
#define cudaCheckError() {                                                                 \
    cudaDeviceSynchronize();                                                               \
    cudaError_t e=cudaPeekAtLastError();                                                   \
    if(e!=cudaSuccess) {                                                                   \
        printf("Cuda failure [%u] %s:%d: \"%s\"\n", e, __FILE__, __LINE__,cudaGetErrorString(e));  \
        exit(0);                                                                           \
    }                                                                                      \
}
#else
#define cudaCheckError() {}
#endif

using namespace cmpk;
using namespace penta;

typedef double Real;
typedef  float RealComm;
typedef MemSpace::Device mem_space;

template<typename MemSpaceType, typename RealType, typename FuncType>
void defFactPartitionedPenta(RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, const int, const int, const int, const int, MPI_Comm, FuncType);

template<typename RealType>
void setRHS(RealType*, const unsigned int, const unsigned int, const unsigned int, MPI_Comm);

template<typename RealType>
RealType checkSoln(RealType*, const unsigned int, const unsigned int, const unsigned int, MPI_Comm);


int main(int argc, char* argv[]) {
    (void) argc; (void) argv;

    MPI_Init(NULL, NULL);

    int num_devices;
    cudaGetDeviceCount(&num_devices);
#ifdef OMPI_MPI_H                  // OpenMPI
    const int device_id  = atoi(getenv("OMPI_COMM_WORLD_LOCAL_RANK"));
#elif defined MVAPICH2_VERSION     // MVAPICH2
    const int device_id  = atoi(getenv("MV2_COMM_WORLD_LOCAL_RANK"));
#else                              // No MPI or unrecognized MPI.
#error "Unrecognizable MPI library. Supported libraries are OpenMPI and MVAPICH2."
#endif
    if (device_id >= num_devices) {
        throw std::runtime_error("Number of MPI processes requested exceeds the maximum number of available devices.");
    }
    cudaSetDevice(device_id % num_devices);
    cudaCheckError();


    // Basic configuration
    constexpr unsigned int Ni = 64;
    constexpr unsigned int Nj = 64;
    constexpr unsigned int Nk = 64;
    constexpr unsigned int N_sub = Ni;
    constexpr unsigned int max_local_fact_size = 1024;
    int Np, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &Np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const unsigned int start_idx = rank * N_sub;

    // Set system
    Real* fact_local_prev_2 = nullptr;
    Real* fact_local_prev_1 = nullptr;
    Real* fact_local_curr   = nullptr;
    Real* fact_local_next_1 = nullptr;
    Real* fact_local_next_2 = nullptr;
    Real* fact_dist_prev    = nullptr;
    Real* fact_dist_curr    = nullptr;
    Real* fact_dist_next    = nullptr;
    Real* Si                = nullptr;
    Real* Ri                = nullptr;
    Real* Li_tilde_tail     = nullptr;
    Real* Ui_tilde_head     = nullptr;

    allocFactBuffers<mem_space, Real>(
            &fact_local_prev_2, &fact_local_prev_1, &fact_local_curr, &fact_local_next_1, &fact_local_next_2,
            &fact_dist_prev, &fact_dist_curr, &fact_dist_next, &Si, &Ri, &Li_tilde_tail, &Ui_tilde_head,
            N_sub, Np);

    defFactPartitionedPenta<mem_space, Real>(
            fact_local_prev_2, fact_local_prev_1, fact_local_curr, fact_local_next_1, fact_local_next_2,
            fact_dist_prev, fact_dist_curr, fact_dist_next,
            Si, Ri, Li_tilde_tail, Ui_tilde_head,
            start_idx, N_sub, Np, max_local_fact_size, MPI_COMM_WORLD, [=](const int row, const int offset)->Real {
                (void) row;
                if (offset == 0)
                    return 1.00;
                else if (offset == 1 || offset == -1)
                    return COEF_ALP;
                else if (offset == 2 || offset == -2)
                    return COEF_BET;
                return 0.0;
            });

    MPI_Request mpi_reqs [2];
    MPI_Status  mpi_stats[2];
    constexpr unsigned int N_local = N_sub - 2;
    constexpr unsigned int arr_stride_i = Nj * Nk;
    constexpr unsigned int arr_stride_j = Nk;
    Real* x           = nullptr;
    Real* x_host      = nullptr;
    Real* x_tilde_cur = nullptr;
    Real* x_tilde_nbr = nullptr;
    Real* x_tilde_buf = nullptr;
    memAllocArray<MemSpace::Device, Real>(&x          , Ni * Nj * Nk);
    memAllocArray<MemSpace::Device, Real>(&x_tilde_cur,  2 * Nj * Nk);
    memAllocArray<MemSpace::Device, Real>(&x_tilde_nbr,  2 * Nj * Nk);
    memAllocArray<MemSpace::Device, Real>(&x_tilde_buf,  2 * Nj * Nk);
    memAllocArray<MemSpace::Host  , Real>(&x_host     , Ni * Nj * Nk);
    RealComm* x_comm_prev = nullptr;
    RealComm* x_comm_curr = nullptr;
    RealComm* x_comm_next = nullptr;
    memAllocArray<MemSpace::Device, RealComm>(&x_comm_prev,  2 * Nj * Nk);
    memAllocArray<MemSpace::Device, RealComm>(&x_comm_curr,  2 * Nj * Nk);
    memAllocArray<MemSpace::Device, RealComm>(&x_comm_next,  2 * Nj * Nk);
    cudaCheckError();

    setRHS<Real>(x_host, Ni, Nj, Nk, MPI_COMM_WORLD);
    deepCopy<MemSpace::Device, MemSpace::Host>(x, x_host, Ni * Nj * Nk);

    Real* x_tilde_host = nullptr; // DEBUG ONLY
    memAllocArray<MemSpace::Host, Real>(&x_tilde_host, 2 * Nj * Nk); // DEBUG ONLY

    cudaCheckError();
    ////////////////////////////////////
    //---- START SOLUTION PROCESS ----//

    Real* x_loc = &x[2 * arr_stride_i];
    localSolPentaPCRDimI<Real>(x_loc, fact_local_prev_2, fact_local_prev_1, fact_local_curr, fact_local_next_1, fact_local_next_2, max_local_fact_size, N_local, Nj, Nk, arr_stride_i, arr_stride_j);
    copySlicesFrom3DArrayDimI<Real>(x_tilde_buf, &x_loc[(N_local-2) * arr_stride_i], 2, Nj, Nk, arr_stride_i, arr_stride_j); // get y_prev_bot
    deviceFence();
    MPI_Irecv(x_tilde_nbr, 2*Nj*Nk, MPIDataType<Real>::value, (rank - 1 + Np) % Np, 1, MPI_COMM_WORLD, &mpi_reqs[0]);
    MPI_Isend(x_tilde_buf, 2*Nj*Nk, MPIDataType<Real>::value, (rank + 1     ) % Np, 1, MPI_COMM_WORLD, &mpi_reqs[1]);
    copySlicesFrom3DArrayDimI<Real>(x_tilde_cur, x, 2, Nj, Nk, arr_stride_i, arr_stride_j);
    MPI_Waitall(2, mpi_reqs, mpi_stats);
    copySlicesFrom3DArrayDimI<Real>(x_tilde_buf, x_loc, 2, Nj, Nk, arr_stride_i, arr_stride_j); // get y_curr_head
    calcReducedSystemRHSLocal(x_tilde_cur, x_tilde_nbr, x_tilde_buf, Li_tilde_tail, Ui_tilde_head, Nj*Nk);
    distSolve<Real, RealComm, mem_space>(x_tilde_cur, x_comm_prev, x_comm_curr, x_comm_next, fact_dist_prev, fact_dist_curr, fact_dist_next, Nj*Nk, Np, MPI_COMM_WORLD, nullptr, nullptr, nullptr);
    deviceFence();
    MPI_Irecv(x_tilde_nbr, 2*Nj*Nk, MPIDataType<Real>::value, (rank + 1     ) % Np, 1, MPI_COMM_WORLD, &mpi_reqs[0]);
    MPI_Isend(x_tilde_cur, 2*Nj*Nk, MPIDataType<Real>::value, (rank - 1 + Np) % Np, 1, MPI_COMM_WORLD, &mpi_reqs[1]);
    copySlicesTo3DArrayDimI<Real>(x, x_tilde_cur, 2, Nj, Nk, arr_stride_i, arr_stride_j);
    MPI_Waitall(2, mpi_reqs, mpi_stats);
    updateLocalSolDimI<Real>(x_loc, x_tilde_cur, x_tilde_nbr, Si, Ri, Ni-2, Nj, Nk, Nj*Nk, Nk);
    deviceFence();
    cudaCheckError();
    //---- END OF SOLUTION PROCESS ----//
    /////////////////////////////////////

    deepCopy<MemSpace::Host, MemSpace::Device>(x_host, x, Ni * Nj * Nk);
    const Real err = checkSoln(x_host, Ni, Nj, Nk, MPI_COMM_WORLD);
    printf("Err = %.5e\n", err);
    
    cudaCheckError();
    memFreeArray<MemSpace::Device>(x);
    memFreeArray<MemSpace::Device>(x_tilde_cur);
    memFreeArray<MemSpace::Device>(x_tilde_nbr);
    memFreeArray<MemSpace::Device>(x_tilde_buf);
    memFreeArray<MemSpace::Device>(x_comm_prev);
    memFreeArray<MemSpace::Device>(x_comm_curr);
    memFreeArray<MemSpace::Device>(x_comm_next);
    memFreeArray<MemSpace::Host  >(x_tilde_host);

    cudaCheckError();
    MPI_Finalize();
    return 0;
}



template<typename RealType>
void setRHS(RealType* x, const unsigned int Ni_sub, const unsigned int Nj, const unsigned int Nk, MPI_Comm comm) {
    int rank, Np;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &Np);
    const unsigned int start_idx = rank * Ni_sub;
    const unsigned int N_tot     = Np   * Ni_sub;
    const RealType dx = DOM_LEN / N_tot;
    for (unsigned int i_sub = 0; i_sub < Ni_sub; i_sub++) {
        const unsigned int i = i_sub + start_idx; 
        for (unsigned int j = 0; j < Nj; j++) {
            for (unsigned int k = 0; k < Nk; k++) {
                x[i_sub * Nj * Nk + j * Nk + k]
                    = (COEF_A * (sin(K_WAVE * int(i + 1) * dx) - sin(K_WAVE * int(i - 1) * dx))
                    +  COEF_B * (sin(K_WAVE * int(i + 2) * dx) - sin(K_WAVE * int(i - 2) * dx))
                    +  COEF_C * (sin(K_WAVE * int(i + 3) * dx) - sin(K_WAVE * int(i - 3) * dx))) / dx;
            } // for k
        } // for j
    } // for i_sub
}


template<typename RealType>
RealType checkSoln(RealType* x, const unsigned int Ni_sub, const unsigned int Nj, const unsigned int Nk, MPI_Comm comm) {
    int rank, Np;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &Np);
    const unsigned int start_idx = rank * Ni_sub;
    const unsigned int N_tot     = Np   * Ni_sub;

    const RealType dx = DOM_LEN / N_tot;
    const Real K_WAVE_MOD = (2.0 * COEF_A * sin(K_WAVE * dx) + 2.0 * COEF_B * sin(2.0 * K_WAVE * dx) + 2.0 * COEF_C * sin(3.0 * K_WAVE * dx)) / dx
                          / (1.0 + 2.0 * COEF_ALP * cos(K_WAVE * dx) + 2.0 * COEF_BET * cos(2.0 * K_WAVE * dx));
    Real err_loc = 0.0;
    for (unsigned int i_sub = 0; i_sub < Ni_sub; i_sub ++) {
        const unsigned int i = i_sub + start_idx; 
        const Real x_ref = K_WAVE_MOD * cos(K_WAVE * int(i) * dx);
        for (unsigned int j = 0; j < Nj; j++) {
            for (unsigned int k = 0; k < Nk; k++) {
                Real err_loc_i = x_ref - x[i_sub * Nj * Nk + j * Nk + k];
                err_loc += err_loc_i * err_loc_i;
            } // k
        } // j
    } // i
    Real err;
    MPI_Allreduce(&err_loc, &err, 1, MPIDataType<Real>::value, MPI_SUM, comm);
    err = sqrt((err / N_tot) / (Nj * Nk));

    if (err > 1e-13) {
        for (int r = 0; r < Np; r++) {
            if (r == rank) {
                unsigned int j = 0;
                unsigned int k = 0;
                printf("%3s, %12s, %12s, %12s # k_mod = %.5e\n", "idx", "x_ref", "x_num", "dif", K_WAVE_MOD);
                for (unsigned int i_sub = 0; i_sub < Ni_sub; i_sub ++) {
                    const Real x_ref = K_WAVE_MOD * cos(K_WAVE * (i_sub + start_idx) * dx);
                    const Real x_num = x[i_sub * Nj * Nk + j * Nk + k];
                    printf("%3d, %12.5e, %12.5e, %12.5e\n", i_sub + start_idx, x_ref, x_num, x_ref - x_num);
                }
            }
            MPI_Barrier(comm);
        } // for r
    }

    return err;
}



template<typename MemSpaceType, typename RealType, typename FuncType>
void defFactPartitionedPenta(
        RealType* fact_local_prev_2, RealType* fact_local_prev_1, RealType* fact_local_curr, RealType* fact_local_next_1, RealType* fact_local_next_2,
        RealType* fact_dist_prev, RealType* fact_dist_curr, RealType* fact_dist_next, RealType* Si, RealType* Ri, RealType* Li_tilde_tail, RealType* Ui_tilde_head,
        const int start_idx, const int N_sub, const int Np, const int max_local_fact_size, MPI_Comm comm_sub, FuncType sys_def
) {
    static_assert(std::is_same<MemSpaceType, MemSpace::Host>::value || std::is_same<MemSpaceType, MemSpace::Device>::value, "Valid MemSpace typenames are MemSpace::Host and MemSpace::Device.");
    RealType* fact_local_prev_2_host = nullptr;
    RealType* fact_local_prev_1_host = nullptr;
    RealType* fact_local_curr_host   = nullptr;
    RealType* fact_local_next_1_host = nullptr;
    RealType* fact_local_next_2_host = nullptr;
    RealType* fact_dist_prev_host    = nullptr;
    RealType* fact_dist_curr_host    = nullptr;
    RealType* fact_dist_next_host    = nullptr;
    RealType* Si_host                = nullptr;
    RealType* Ri_host                = nullptr;
    RealType* Li_tilde_tail_host     = nullptr;
    RealType* Ui_tilde_head_host     = nullptr;

    if constexpr (std::is_same<MemSpaceType, MemSpace::Device>::value) {
        const int local_fact_size = (N_sub - 2) * log2Ceil(N_sub - 2);
        fact_local_prev_2_host = new RealType [local_fact_size];
        fact_local_prev_1_host = new RealType [local_fact_size];
        fact_local_curr_host   = new RealType [local_fact_size];
        fact_local_next_1_host = new RealType [local_fact_size];
        fact_local_next_2_host = new RealType [local_fact_size];
        const int dist_fact_size = 4 * numDistSolSteps<int>(Np);
        fact_dist_prev_host = new RealType[dist_fact_size];
        fact_dist_curr_host = new RealType[dist_fact_size];
        fact_dist_next_host = new RealType[dist_fact_size];
        const int local_soln_size = 2 * (N_sub - 2);
        Si_host = new RealType[local_soln_size];
        Ri_host = new RealType[local_soln_size];
        Li_tilde_tail_host = new RealType[3];
        Ui_tilde_head_host = new RealType[3];
    } else {
        fact_local_prev_2_host = fact_local_prev_2;
        fact_local_prev_1_host = fact_local_prev_1;
        fact_local_curr_host   = fact_local_curr;
        fact_local_next_1_host = fact_local_next_1;
        fact_local_next_2_host = fact_local_next_2;
        fact_dist_prev_host    = fact_dist_prev;
        fact_dist_curr_host    = fact_dist_curr;
        fact_dist_next_host    = fact_dist_next;
        Si_host                = Si;                     
        Ri_host                = Ri;                     
        Li_tilde_tail_host = Li_tilde_tail;
        Ui_tilde_head_host = Ui_tilde_head;
    }

    RealType* part_L2 = new RealType [N_sub];
    RealType* part_L1 = new RealType [N_sub];
    RealType* part_D  = new RealType [N_sub];
    RealType* part_U1 = new RealType [N_sub];
    RealType* part_U2 = new RealType [N_sub];

    for (int j = 0; j < N_sub; j++) {
        const int idx = j + start_idx;
        part_L2[j] = sys_def(idx, -2);
        part_L1[j] = sys_def(idx, -1);
        part_D [j] = sys_def(idx,  0);
        part_U1[j] = sys_def(idx,  1);
        part_U2[j] = sys_def(idx,  2);
    }

    factPartitionedPentaHost(
        fact_local_prev_2_host, fact_local_prev_1_host, fact_local_curr_host, fact_local_next_1_host, fact_local_next_2_host,
        fact_dist_prev_host, fact_dist_curr_host, fact_dist_next_host, Si_host, Ri_host,
        part_L2, part_L1, part_D, part_U1, part_U2,
        N_sub, Np, max_local_fact_size, comm_sub
    );

    delete [] part_L2;
    delete [] part_L1;
    delete [] part_D;
    delete [] part_U1;
    delete [] part_U2;

    Li_tilde_tail_host[0] = sys_def(start_idx    , -2);
    Li_tilde_tail_host[1] = sys_def(start_idx    , -1);
    Li_tilde_tail_host[2] = sys_def(start_idx + 1, -2);
    Ui_tilde_head_host[0] = sys_def(start_idx    ,  2);
    Ui_tilde_head_host[1] = sys_def(start_idx + 1,  1);
    Ui_tilde_head_host[2] = sys_def(start_idx + 1,  2);

    if constexpr (std::is_same<MemSpaceType, MemSpace::Device>::value) {
       #ifdef COMPACK3D_DEVICE_ENABLED_CUDA
        const size_t local_fact_size = (N_sub - 2) * log2Ceil(N_sub - 2) * sizeof(RealType);
        cudaMemcpy((void*)fact_local_prev_2, (void*)fact_local_prev_2_host, local_fact_size, cudaMemcpyHostToDevice);
        cudaMemcpy((void*)fact_local_prev_1, (void*)fact_local_prev_1_host, local_fact_size, cudaMemcpyHostToDevice);
        cudaMemcpy((void*)fact_local_curr  , (void*)fact_local_curr_host  , local_fact_size, cudaMemcpyHostToDevice);
        cudaMemcpy((void*)fact_local_next_1, (void*)fact_local_next_1_host, local_fact_size, cudaMemcpyHostToDevice);
        cudaMemcpy((void*)fact_local_next_2, (void*)fact_local_next_2_host, local_fact_size, cudaMemcpyHostToDevice);
        const size_t dist_fact_size = 4 * numDistSolSteps<size_t>(Np) * sizeof(RealType);
        cudaMemcpy((void*)fact_dist_prev, (void*)fact_dist_prev_host, dist_fact_size, cudaMemcpyHostToDevice);
        cudaMemcpy((void*)fact_dist_curr, (void*)fact_dist_curr_host, dist_fact_size, cudaMemcpyHostToDevice);
        cudaMemcpy((void*)fact_dist_next, (void*)fact_dist_next_host, dist_fact_size, cudaMemcpyHostToDevice);
        const size_t local_soln_size = 2 * (N_sub - 2) * sizeof(RealType);
        cudaMemcpy((void*)Si, (void*)Si_host, local_soln_size, cudaMemcpyHostToDevice);
        cudaMemcpy((void*)Ri, (void*)Ri_host, local_soln_size, cudaMemcpyHostToDevice);
        cudaMemcpy((void*)Li_tilde_tail, (void*)Li_tilde_tail_host, 3 * sizeof(RealType), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)Ui_tilde_head, (void*)Ui_tilde_head_host, 3 * sizeof(RealType), cudaMemcpyHostToDevice);
       #endif
        delete [] fact_local_prev_2_host;
        delete [] fact_local_prev_1_host;
        delete [] fact_local_curr_host;
        delete [] fact_local_next_1_host;
        delete [] fact_local_next_2_host;
        delete [] fact_dist_prev_host;
        delete [] fact_dist_curr_host;
        delete [] fact_dist_next_host;
        delete [] Si_host;
        delete [] Ri_host;
        delete [] Li_tilde_tail_host;
        delete [] Ui_tilde_head_host;
        fact_local_prev_2_host = nullptr;
        fact_local_prev_1_host = nullptr;
        fact_local_curr_host   = nullptr;
        fact_local_next_1_host = nullptr;
        fact_local_next_2_host = nullptr;
        fact_dist_prev         = nullptr;
        fact_dist_curr         = nullptr;
        fact_dist_next         = nullptr;
        Si_host                = nullptr;
        Ri_host                = nullptr;
        Li_tilde_tail_host     = nullptr;
        Ui_tilde_head_host     = nullptr;
    } 
}
