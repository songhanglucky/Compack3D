#include "Compack3D_api.h"
#include <stdexcept>
#include <mpi.h>
#include <chrono>
#include <cmath>

#define COEF_ALP ( 1.0 / 3.0      )
#define COEF_A   (14.0 / 9.0 / 2.0)
#define COEF_B   ( 1.0 / 9.0 / 4.0)
#define DOM_LEN   6.283185307179586
#define K_WAVE    2.0

#define cudaCheckError() {                                                                 \
    cudaDeviceSynchronize();                                                               \
    cudaError_t e=cudaPeekAtLastError();                                                   \
    if(e!=cudaSuccess) {                                                                   \
        printf("Cuda failure [%u] %s:%d: \"%s\"\n", e, __FILE__, __LINE__,cudaGetErrorString(e));  \
        exit(0);                                                                           \
    }                                                                                      \
}


typedef double Real;
typedef  float RealComm;
typedef cmpk::DistTriSolDimI<Real, RealComm, cmpk::MemSpace::Device> TriSol;


template<typename RealType>
void setRHS(RealType*, const unsigned int, const unsigned int, const unsigned int, MPI_Comm);
template<typename RealType>
RealType checkSoln(RealType*, const unsigned int, const unsigned int, const unsigned int, MPI_Comm);


int main(int argc, char* argv[]) {
    (void) argc; (void) argv;

    //////////////////////////
    // INPUT CONFIGURATIONS //
    //////////////////////////
    constexpr unsigned int NUM_GRIDS_I = 254;
    constexpr unsigned int NUM_GRIDS_J = 255;
    constexpr unsigned int NUM_GRIDS_K = 256;


    //////////////////////////////////
    // INITIALIZATIONS OF UTILITIES //
    //////////////////////////////////
    MPI_Init(NULL, NULL);
    MPI_Comm comm_sub = MPI_COMM_WORLD;
    int rank, Np;
    MPI_Comm_rank(comm_sub, &rank);
    MPI_Comm_size(comm_sub,   &Np);

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


    ///////////////////////
    // SET LINEAR SYSTEM //
    ///////////////////////
    Real* l = nullptr; cmpk::memAllocArray<cmpk::MemSpace::Host, Real>(&l, NUM_GRIDS_I);
    Real* d = nullptr; cmpk::memAllocArray<cmpk::MemSpace::Host, Real>(&d, NUM_GRIDS_I);
    Real* u = nullptr; cmpk::memAllocArray<cmpk::MemSpace::Host, Real>(&u, NUM_GRIDS_I);
    for (unsigned int i = 0; i < NUM_GRIDS_I; i++) {
        l[i] = COEF_ALP;
        d[i] =      1.0;
        u[i] = COEF_ALP;
    }


    //////////////////////////////////////////////////
    // CONFIGURE THE PARALLEL PENTA-DIAGONAL SOLVER //
    //////////////////////////////////////////////////
    Real*     shared_x_tilde_cur = nullptr; // scratch buffer on device memory
    Real*     shared_x_tilde_nbr = nullptr; // scratch buffer on device memory
    Real*     shared_x_tilde_buf = nullptr; // scratch buffer on device memory
    RealComm* shared_x_comm_prev = nullptr; // scratch buffer on device memory
    RealComm* shared_x_comm_curr = nullptr; // scratch buffer on device memory
    RealComm* shared_x_comm_next = nullptr; // scratch buffer on device memory
    cmpk::memAllocArray<cmpk::MemSpace::Device, Real    >(&shared_x_tilde_cur, NUM_GRIDS_J * NUM_GRIDS_K);
    cmpk::memAllocArray<cmpk::MemSpace::Device, Real    >(&shared_x_tilde_nbr, NUM_GRIDS_J * NUM_GRIDS_K);
    cmpk::memAllocArray<cmpk::MemSpace::Device, Real    >(&shared_x_tilde_buf, NUM_GRIDS_J * NUM_GRIDS_K);
    cmpk::memAllocArray<cmpk::MemSpace::Device, RealComm>(&shared_x_comm_prev, NUM_GRIDS_J * NUM_GRIDS_K);
    cmpk::memAllocArray<cmpk::MemSpace::Device, RealComm>(&shared_x_comm_curr, NUM_GRIDS_J * NUM_GRIDS_K);
    cmpk::memAllocArray<cmpk::MemSpace::Device, RealComm>(&shared_x_comm_next, NUM_GRIDS_J * NUM_GRIDS_K);
    Real*     shared_x_tilde_send_host = nullptr; // host scratch buffer for non-device-aware communication
    Real*     shared_x_tilde_recv_host = nullptr; // host scratch buffer for non-device-aware communication
    RealComm* shared_x_comm_prev_host  = nullptr; // host scratch buffer for non-device-aware communication
    RealComm* shared_x_comm_curr_host  = nullptr; // host scratch buffer for non-device-aware communication
    RealComm* shared_x_comm_next_host  = nullptr; // host scratch buffer for non-device-aware communication
    #ifndef COMPACK3D_DEVICE_COMM_ENABLED
    cmpk::memAllocArray<cmpk::MemSpace::Host, Real    >(&shared_x_tilde_send_host, NUM_GRIDS_J * NUM_GRIDS_K);
    cmpk::memAllocArray<cmpk::MemSpace::Host, Real    >(&shared_x_tilde_recv_host, NUM_GRIDS_J * NUM_GRIDS_K);
    cmpk::memAllocArray<cmpk::MemSpace::Host, RealComm>(&shared_x_comm_prev_host , NUM_GRIDS_J * NUM_GRIDS_K);
    cmpk::memAllocArray<cmpk::MemSpace::Host, RealComm>(&shared_x_comm_curr_host , NUM_GRIDS_J * NUM_GRIDS_K);
    cmpk::memAllocArray<cmpk::MemSpace::Host, RealComm>(&shared_x_comm_next_host , NUM_GRIDS_J * NUM_GRIDS_K);
    #endif
    TriSol tri_sol(NUM_GRIDS_I, NUM_GRIDS_J, NUM_GRIDS_K, NUM_GRIDS_J * NUM_GRIDS_K, NUM_GRIDS_K, comm_sub);
    tri_sol.resetSharedBuffers(
            shared_x_tilde_cur,
            shared_x_tilde_nbr,
            shared_x_tilde_buf,
            shared_x_comm_prev,
            shared_x_comm_curr,
            shared_x_comm_next,
            shared_x_comm_prev_host,
            shared_x_comm_curr_host,
            shared_x_comm_next_host,
            shared_x_tilde_send_host,
            shared_x_tilde_recv_host
    );
    tri_sol.resetSystem(l, d, u);
    cmpk::memFreeArray<cmpk::MemSpace::Host>(l); l = nullptr;
    cmpk::memFreeArray<cmpk::MemSpace::Host>(d); d = nullptr;
    cmpk::memFreeArray<cmpk::MemSpace::Host>(u); u = nullptr;

    /////////////////////////
    // SET RIGHT-HAND SIDE //
    /////////////////////////
    Real* x      = nullptr;
    Real* x_host = nullptr;
    cmpk::memAllocArray<cmpk::MemSpace::Device, Real>(&x     , NUM_GRIDS_I * NUM_GRIDS_J * NUM_GRIDS_K);
    cmpk::memAllocArray<cmpk::MemSpace::Host  , Real>(&x_host, NUM_GRIDS_I * NUM_GRIDS_J * NUM_GRIDS_K);
    setRHS<Real>(x_host, NUM_GRIDS_I, NUM_GRIDS_J, NUM_GRIDS_K, comm_sub);
    cmpk::deepCopy<cmpk::MemSpace::Device, cmpk::MemSpace::Host>(x, x_host, NUM_GRIDS_I * NUM_GRIDS_J * NUM_GRIDS_K);


    /////////////////////////
    // SOLVE LINEAR SYSTEM //
    /////////////////////////
    for (int i = 0; i < 10; i++) MPI_Barrier(comm_sub);
    cmpk::deviceFence();
    auto time_start = std::chrono::high_resolution_clock::now();
    tri_sol.solve(x);

    cmpk::deviceFence();
    auto time_stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(time_stop - time_start);


    //////////////////////
    // CHECK THE RESULT //
    //////////////////////
    cudaCheckError();
    cmpk::deepCopy<cmpk::MemSpace::Host, cmpk::MemSpace::Device>(x_host, x, NUM_GRIDS_I * NUM_GRIDS_J * NUM_GRIDS_K);
    const Real err_rms = checkSoln<Real>(x_host, NUM_GRIDS_I, NUM_GRIDS_J, NUM_GRIDS_K, comm_sub);
    if (rank == 0)
        printf("Test using %d partitions with partition grid size %d x %d x %d.\nThe error is %12.5e\nSolution process took %.3f msec.\n",
            Np, NUM_GRIDS_I, NUM_GRIDS_J, NUM_GRIDS_K, err_rms, 0.001*duration.count());


    ////////////////
    // FINALIZING //
    ////////////////
    cmpk::memFreeArray<cmpk::MemSpace::Device, Real    >(shared_x_tilde_cur);  shared_x_tilde_cur = nullptr;
    cmpk::memFreeArray<cmpk::MemSpace::Device, Real    >(shared_x_tilde_nbr);  shared_x_tilde_nbr = nullptr;
    cmpk::memFreeArray<cmpk::MemSpace::Device, Real    >(shared_x_tilde_buf);  shared_x_tilde_buf = nullptr;
    cmpk::memFreeArray<cmpk::MemSpace::Device, RealComm>(shared_x_comm_prev);  shared_x_comm_prev = nullptr;
    cmpk::memFreeArray<cmpk::MemSpace::Device, RealComm>(shared_x_comm_curr);  shared_x_comm_curr = nullptr;
    cmpk::memFreeArray<cmpk::MemSpace::Device, RealComm>(shared_x_comm_next);  shared_x_comm_next = nullptr;
    if (shared_x_tilde_send_host) { cmpk::memFreeArray<cmpk::MemSpace::Host, Real    >(shared_x_tilde_send_host); shared_x_tilde_send_host = nullptr; }
    if (shared_x_tilde_recv_host) { cmpk::memFreeArray<cmpk::MemSpace::Host, Real    >(shared_x_tilde_recv_host); shared_x_tilde_recv_host = nullptr; }
    if (shared_x_comm_prev_host ) { cmpk::memFreeArray<cmpk::MemSpace::Host, RealComm>(shared_x_comm_prev_host ); shared_x_comm_prev_host  = nullptr; }
    if (shared_x_comm_curr_host ) { cmpk::memFreeArray<cmpk::MemSpace::Host, RealComm>(shared_x_comm_curr_host ); shared_x_comm_curr_host  = nullptr; }
    if (shared_x_comm_next_host ) { cmpk::memFreeArray<cmpk::MemSpace::Host, RealComm>(shared_x_comm_next_host ); shared_x_comm_next_host  = nullptr; }

    return MPI_Finalize();
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
                    +  COEF_B * (sin(K_WAVE * int(i + 2) * dx) - sin(K_WAVE * int(i - 2) * dx))) / dx;
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
    const Real K_WAVE_MOD = (2.0 * COEF_A * sin(K_WAVE * dx) + 2.0 * COEF_B * sin(2.0 * K_WAVE * dx)) / dx
                          / (1.0 + 2.0 * COEF_ALP * cos(K_WAVE * dx));
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
    MPI_Allreduce(&err_loc, &err, 1, cmpk::MPIDataType<Real>::value, MPI_SUM, comm);
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


