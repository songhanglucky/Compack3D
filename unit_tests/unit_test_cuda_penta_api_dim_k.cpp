#include "Compack3D_api.h"
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
typedef cmpk::DistPentaSolDimK<Real, RealComm, cmpk::MemSpace::Device> PentaSol;


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
    Real* l2 = nullptr; cmpk::memAllocArray<cmpk::MemSpace::Host, Real>(&l2, NUM_GRIDS_K);
    Real* l1 = nullptr; cmpk::memAllocArray<cmpk::MemSpace::Host, Real>(&l1, NUM_GRIDS_K);
    Real* d  = nullptr; cmpk::memAllocArray<cmpk::MemSpace::Host, Real>(&d , NUM_GRIDS_K);
    Real* u1 = nullptr; cmpk::memAllocArray<cmpk::MemSpace::Host, Real>(&u1, NUM_GRIDS_K);
    Real* u2 = nullptr; cmpk::memAllocArray<cmpk::MemSpace::Host, Real>(&u2, NUM_GRIDS_K);
    for (unsigned int k = 0; k < NUM_GRIDS_K; k++) {
        l2[k] = COEF_BET;
        l1[k] = COEF_ALP;
        d [k] =      1.0;
        u1[k] = COEF_ALP;
        u2[k] = COEF_BET;
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
    cmpk::memAllocArray<cmpk::MemSpace::Device, Real    >(&shared_x_tilde_cur, 2 * NUM_GRIDS_I * NUM_GRIDS_J);
    cmpk::memAllocArray<cmpk::MemSpace::Device, Real    >(&shared_x_tilde_nbr, 2 * NUM_GRIDS_I * NUM_GRIDS_J);
    cmpk::memAllocArray<cmpk::MemSpace::Device, Real    >(&shared_x_tilde_buf, 2 * NUM_GRIDS_I * NUM_GRIDS_J);
    cmpk::memAllocArray<cmpk::MemSpace::Device, RealComm>(&shared_x_comm_prev, 2 * NUM_GRIDS_I * NUM_GRIDS_J);
    cmpk::memAllocArray<cmpk::MemSpace::Device, RealComm>(&shared_x_comm_curr, 2 * NUM_GRIDS_I * NUM_GRIDS_J);
    cmpk::memAllocArray<cmpk::MemSpace::Device, RealComm>(&shared_x_comm_next, 2 * NUM_GRIDS_I * NUM_GRIDS_J);
    Real*     shared_x_tilde_send_host = nullptr; // host scratch buffer for non-device-aware communication
    Real*     shared_x_tilde_recv_host = nullptr; // host scratch buffer for non-device-aware communication
    RealComm* shared_x_comm_prev_host  = nullptr; // host scratch buffer for non-device-aware communication
    RealComm* shared_x_comm_curr_host  = nullptr; // host scratch buffer for non-device-aware communication
    RealComm* shared_x_comm_next_host  = nullptr; // host scratch buffer for non-device-aware communication
    #ifndef COMPACK3D_DEVICE_COMM_ENABLED
    cmpk::memAllocArray<cmpk::MemSpace::Host, Real    >(&shared_x_tilde_send_host, 2 * NUM_GRIDS_I * NUM_GRIDS_J);
    cmpk::memAllocArray<cmpk::MemSpace::Host, Real    >(&shared_x_tilde_recv_host, 2 * NUM_GRIDS_I * NUM_GRIDS_J);
    cmpk::memAllocArray<cmpk::MemSpace::Host, RealComm>(&shared_x_comm_prev_host , 2 * NUM_GRIDS_I * NUM_GRIDS_J);
    cmpk::memAllocArray<cmpk::MemSpace::Host, RealComm>(&shared_x_comm_curr_host , 2 * NUM_GRIDS_I * NUM_GRIDS_J);
    cmpk::memAllocArray<cmpk::MemSpace::Host, RealComm>(&shared_x_comm_next_host , 2 * NUM_GRIDS_I * NUM_GRIDS_J);
    #endif
    PentaSol penta_sol(NUM_GRIDS_I, NUM_GRIDS_J, NUM_GRIDS_K, NUM_GRIDS_J * NUM_GRIDS_K, NUM_GRIDS_K, comm_sub);
    penta_sol.resetSharedBuffers(
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
    penta_sol.resetSystem(l2, l1, d, u1, u2);
    cmpk::memFreeArray<cmpk::MemSpace::Host>(l2); l2 = nullptr;
    cmpk::memFreeArray<cmpk::MemSpace::Host>(l1); l1 = nullptr;
    cmpk::memFreeArray<cmpk::MemSpace::Host>(d ); d  = nullptr;
    cmpk::memFreeArray<cmpk::MemSpace::Host>(u1); u1 = nullptr;
    cmpk::memFreeArray<cmpk::MemSpace::Host>(u2); u2 = nullptr;


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
    penta_sol.solve(x);
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
void setRHS(RealType* x, const unsigned int Ni, const unsigned int Nj, const unsigned int Nk_sub, MPI_Comm comm) {
    int rank, Np;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &Np);
    const unsigned int start_idx = rank * Nk_sub;
    const unsigned int N_tot     = Np   * Nk_sub;
    const RealType dz = DOM_LEN / N_tot;
    for (unsigned int i = 0; i < Ni; i++) {
        for (unsigned int j = 0; j < Nj; j++) {
            for (unsigned int k_sub = 0; k_sub < Nk_sub; k_sub++) {
                const unsigned int k = k_sub + start_idx; 
                x[i * Nj * Nk_sub + j * Nk_sub + k_sub]
                    = (COEF_A * (sin(K_WAVE * int(k + 1) * dz) - sin(K_WAVE * int(k - 1) * dz))
                    +  COEF_B * (sin(K_WAVE * int(k + 2) * dz) - sin(K_WAVE * int(k - 2) * dz))
                    +  COEF_C * (sin(K_WAVE * int(k + 3) * dz) - sin(K_WAVE * int(k - 3) * dz))) / dz;
            } // for k
        } // for j
    } // for i_sub
}


template<typename RealType>
RealType checkSoln(RealType* x, const unsigned int Ni, const unsigned int Nj, const unsigned int Nk_sub, MPI_Comm comm) {
    int rank, Np;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &Np);
    const unsigned int start_idx = rank * Nk_sub;
    const unsigned int N_tot     = Np   * Nk_sub;

    const RealType dz = DOM_LEN / N_tot;
    const Real K_WAVE_MOD = (2.0 * COEF_A * sin(K_WAVE * dz) + 2.0 * COEF_B * sin(2.0 * K_WAVE * dz) + 2.0 * COEF_C * sin(3.0 * K_WAVE * dz)) / dz
                          / (1.0 + 2.0 * COEF_ALP * cos(K_WAVE * dz) + 2.0 * COEF_BET * cos(2.0 * K_WAVE * dz));
    Real err_loc = 0.0;
    for (unsigned int i = 0; i < Ni; i++) {
        for (unsigned int j = 0; j < Nj; j++) {
            for (unsigned int k_sub = 0; k_sub < Nk_sub; k_sub++) {
                const unsigned int k = k_sub + start_idx; 
                const Real x_ref = K_WAVE_MOD * cos(K_WAVE * int(k) * dz);
                Real err_loc_k = x_ref - x[i * Nj * Nk_sub + j * Nk_sub + k_sub];
                err_loc += err_loc_k * err_loc_k;
            } // k
        } // j
    } // i
    Real err;
    MPI_Allreduce(&err_loc, &err, 1, cmpk::MPIDataType<Real>::value, MPI_SUM, comm);
    err = sqrt((err / N_tot) / (Ni * Nj));

    if (err > 1e-13) {
        for (int r = 0; r < Np; r++) {
            if (r == rank) {
                unsigned int i = 0;
                unsigned int j = 0;
                printf("%3s, %12s, %12s, %12s # k_mod = %.5e\n", "idx", "x_ref", "x_num", "dif", K_WAVE_MOD);
                for (unsigned int k_sub = 0; k_sub < Nk_sub; k_sub ++) {
                    const Real x_ref = K_WAVE_MOD * cos(K_WAVE * (k_sub + start_idx) * dz);
                    const Real x_num = x[i * Nj * Nk_sub + j * Nk_sub + k_sub];
                    printf("%3d, %12.5e, %12.5e, %12.5e\n", k_sub + start_idx, x_ref, x_num, x_ref - x_num);
                }
            }
            MPI_Barrier(comm);
        } // for r
    }

    return err;
}

