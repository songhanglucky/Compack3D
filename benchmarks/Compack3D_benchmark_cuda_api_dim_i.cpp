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
typedef cmpk::DistPentaSolDimI<Real, RealComm, cmpk::MemSpace::Device> PentaSol;


template<typename RealType>
void setRHS(RealType*, const unsigned int, const unsigned int, const unsigned int, MPI_Comm);
template<typename RealType>
RealType checkSoln(RealType*, const unsigned int, const unsigned int, const unsigned int, MPI_Comm);


int main(int argc, char* argv[]) {
    (void) argc; (void) argv;

    //////////////////////////
    // INPUT CONFIGURATIONS //
    //////////////////////////
    constexpr unsigned int NUM_GRIDS_I = 256;
    constexpr unsigned int NUM_GRIDS_J = 256;
    constexpr unsigned int NUM_GRIDS_K = 256;
    constexpr unsigned int NUM_TESTS   = 100;
    constexpr unsigned int NUM_WARMUPS = 100;


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



    ///////////////////////////
    // ALLOCATE TEST BUFFERS //
    ///////////////////////////
    Real* x      = nullptr;
    Real* x_host = nullptr;
    cmpk::memAllocArray<cmpk::MemSpace::Device, Real>(&x     , NUM_GRIDS_I * NUM_GRIDS_J * NUM_GRIDS_K);
    cmpk::memAllocArray<cmpk::MemSpace::Host  , Real>(&x_host, NUM_GRIDS_I * NUM_GRIDS_J * NUM_GRIDS_K);

    ///////////////////////
    // SET LINEAR SYSTEM //
    ///////////////////////
    Real* l2 = nullptr; cmpk::memAllocArray<cmpk::MemSpace::Host, Real>(&l2, NUM_GRIDS_I);
    Real* l1 = nullptr; cmpk::memAllocArray<cmpk::MemSpace::Host, Real>(&l1, NUM_GRIDS_I);
    Real* d  = nullptr; cmpk::memAllocArray<cmpk::MemSpace::Host, Real>(&d , NUM_GRIDS_I);
    Real* u1 = nullptr; cmpk::memAllocArray<cmpk::MemSpace::Host, Real>(&u1, NUM_GRIDS_I);
    Real* u2 = nullptr; cmpk::memAllocArray<cmpk::MemSpace::Host, Real>(&u2, NUM_GRIDS_I);
    for (unsigned int i = 0; i < NUM_GRIDS_I; i++) {
        l2[i] = COEF_BET;
        l1[i] = COEF_ALP;
        d [i] =      1.0;
        u1[i] = COEF_ALP;
        u2[i] = COEF_BET;
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
    cmpk::memAllocArray<cmpk::MemSpace::Device, Real    >(&shared_x_tilde_cur, 2 * NUM_GRIDS_J * NUM_GRIDS_K);
    cmpk::memAllocArray<cmpk::MemSpace::Device, Real    >(&shared_x_tilde_nbr, 2 * NUM_GRIDS_J * NUM_GRIDS_K);
    cmpk::memAllocArray<cmpk::MemSpace::Device, Real    >(&shared_x_tilde_buf, 2 * NUM_GRIDS_J * NUM_GRIDS_K);
    cmpk::memAllocArray<cmpk::MemSpace::Device, RealComm>(&shared_x_comm_prev, 2 * NUM_GRIDS_J * NUM_GRIDS_K);
    cmpk::memAllocArray<cmpk::MemSpace::Device, RealComm>(&shared_x_comm_curr, 2 * NUM_GRIDS_J * NUM_GRIDS_K);
    cmpk::memAllocArray<cmpk::MemSpace::Device, RealComm>(&shared_x_comm_next, 2 * NUM_GRIDS_J * NUM_GRIDS_K);
    Real*     shared_x_tilde_send_host = nullptr; // host scratch buffer for non-device-aware communication
    Real*     shared_x_tilde_recv_host = nullptr; // host scratch buffer for non-device-aware communication
    RealComm* shared_x_comm_prev_host  = nullptr; // host scratch buffer for non-device-aware communication
    RealComm* shared_x_comm_curr_host  = nullptr; // host scratch buffer for non-device-aware communication
    RealComm* shared_x_comm_next_host  = nullptr; // host scratch buffer for non-device-aware communication
    #ifndef COMPACK3D_DEVICE_COMM_ENABLED
    cmpk::memAllocArray<cmpk::MemSpace::Host, Real    >(&shared_x_tilde_send_host, 2 * NUM_GRIDS_J * NUM_GRIDS_K);
    cmpk::memAllocArray<cmpk::MemSpace::Host, Real    >(&shared_x_tilde_recv_host, 2 * NUM_GRIDS_J * NUM_GRIDS_K);
    cmpk::memAllocArray<cmpk::MemSpace::Host, RealComm>(&shared_x_comm_prev_host , 2 * NUM_GRIDS_J * NUM_GRIDS_K);
    cmpk::memAllocArray<cmpk::MemSpace::Host, RealComm>(&shared_x_comm_curr_host , 2 * NUM_GRIDS_J * NUM_GRIDS_K);
    cmpk::memAllocArray<cmpk::MemSpace::Host, RealComm>(&shared_x_comm_next_host , 2 * NUM_GRIDS_J * NUM_GRIDS_K);
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


    //////////////////////
    // PERFORMANCE TEST //
    //////////////////////
    double time_acc = 0.0;
    for (int i = -static_cast<int>(NUM_WARMUPS); i < static_cast<int>(NUM_TESTS); i++) {
        const double tic = MPI_Wtime();
        penta_sol.solve(x);
        const double time_elapse = 1e3 * (MPI_Wtime() - tic);
        if (rank == 0) printf("Test [%3d] %12.5e msec.\n", i, time_elapse);
        if (i > -1) {
            time_acc += time_elapse;
        }
    }
    if (rank == 0) printf("Average time for %d runs: %12.5e msec\n", NUM_TESTS, time_acc / NUM_TESTS);

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


