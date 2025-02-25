
#include "Compack3D_api.h"
#include "Compack3D_tri.h"
#include "Compack3D_penta.h"
#include <stdexcept>
#include <chrono>
#define MAX_LOCAL_FACT_SIZE 1024


namespace cmpk {

/*!
 * Constructor of distributed solver of penta-diagonal system
 * \tparam RealType        real-value type for primitive data type
 * \tparam RealTypeComm    real-value type for communication on distributed memory used in solving the reduced system
 * \tparam MemSpaceType    memory space, valid values are cmpk::MemSpace::Host and cmpk::MemSpace::Device
 * \param [in]   num_grids_i    number of grid points in i-direction in the current partition
 * \param [in]   num_grids_j    number of grid points in j-direction in the current partition
 * \param [in]   num_grids_k    number of grid points in k-direction in the current partition
 * \param [in]  arr_stride_i    number of entries strided in i-direction in the current partition, i.e., number of entries from i to i+1
 * \param [in]  arr_stride_j    number of entries strided in j-direction in the current partition, i.e., number of entries from j to j+1
 * \param [in]      mpi_comm    a MPI-sub-communicator that only contains the partitions involved in the system
 * \note k-direction is mapped to a contiguous memory layout
 * \note The description of the data size and layout are based on the current distributed partition
 */
template<typename RealType, typename RealTypeComm, typename MemSpaceType>
DistPentaSolDimI<RealType, RealTypeComm, MemSpaceType>::DistPentaSolDimI(
        const unsigned int num_grids_i,
        const unsigned int num_grids_j,
        const unsigned int num_grids_k,
        const unsigned int arr_stride_i,
        const unsigned int arr_stride_j,
        MPI_Comm mpi_comm) :
    NI           ( num_grids_i),
    NJ           ( num_grids_j),
    NK           ( num_grids_k),
    ARR_STRIDE_I (arr_stride_i),
    ARR_STRIDE_J (arr_stride_j)
{
    if ((this->NI < 4) || (this->NJ < 1) || (this->NK < 1)) throw std::invalid_argument("[Compack3D] Partitioned data size is invalid.");
    this->comm = mpi_comm;
    int Np;
    MPI_Comm_size(mpi_comm, &Np);
    this->fact_local_prev_2 = nullptr;
    this->fact_local_prev_1 = nullptr;
    this->fact_local_curr   = nullptr;
    this->fact_local_next_1 = nullptr;
    this->fact_local_next_2 = nullptr;
    this->fact_dist_prev    = nullptr;
    this->fact_dist_curr    = nullptr;
    this->fact_dist_next    = nullptr;
    this->Si                = nullptr;
    this->Ri                = nullptr;
    this->Li_tilde_tail     = nullptr;
    this->Ui_tilde_head     = nullptr;
    penta::allocFactBuffers<MemSpaceType, RealType>(
            &this->fact_local_prev_2,
            &this->fact_local_prev_1,
            &this->fact_local_curr,
            &this->fact_local_next_1,
            &this->fact_local_next_2,
            &this->fact_dist_prev,
            &this->fact_dist_curr,
            &this->fact_dist_next,
            &this->Si,
            &this->Ri,
            &this->Li_tilde_tail,
            &this->Ui_tilde_head,
            num_grids_i, Np);
}



/*!
 * Destructor
 */
template<typename RealType, typename RealTypeComm, typename MemSpaceType>
DistPentaSolDimI<RealType, RealTypeComm, MemSpaceType>::~DistPentaSolDimI() {
    memFreeArray<MemSpaceType, RealType>(this->fact_local_prev_2);
    memFreeArray<MemSpaceType, RealType>(this->fact_local_prev_1);
    memFreeArray<MemSpaceType, RealType>(this->fact_local_curr);
    memFreeArray<MemSpaceType, RealType>(this->fact_local_next_1);
    memFreeArray<MemSpaceType, RealType>(this->fact_local_next_2);
    memFreeArray<MemSpaceType, RealType>(this->fact_dist_prev);
    memFreeArray<MemSpaceType, RealType>(this->fact_dist_curr);
    memFreeArray<MemSpaceType, RealType>(this->fact_dist_next);
    memFreeArray<MemSpaceType, RealType>(this->Si);
    memFreeArray<MemSpaceType, RealType>(this->Ri);
    memFreeArray<MemSpaceType, RealType>(this->Li_tilde_tail);
    memFreeArray<MemSpaceType, RealType>(this->Ui_tilde_head);
}


/*!
 * Detach all shared buffers
 * \note This function call does not free allocated memory and may cause memory leak
 */
template<typename RealType, typename RealTypeComm, typename MemSpaceType>
void DistPentaSolDimI<RealType, RealTypeComm, MemSpaceType>::detachSharedBuffers() {
    this->x_tilde_cur       = nullptr;
    this->x_tilde_nbr       = nullptr;
    this->x_tilde_buf       = nullptr;
    this->x_comm_prev       = nullptr;
    this->x_comm_curr       = nullptr;
    this->x_comm_next       = nullptr;
    this->x_comm_prev_host  = nullptr;
    this->x_comm_curr_host  = nullptr;
    this->x_comm_next_host  = nullptr;
    this->x_tilde_send_host = nullptr;
    this->x_tilde_recv_host = nullptr;
}



/*!
 * Reset shared buffers
 * \param [in]   shared_x_tilde_cur         shared buffer allocated on device
 * \param [in]   shared_x_tilde_nbr         shared buffer allocated on device
 * \param [in]   shared_x_tilde_buf         shared buffer allocated on device
 * \param [in]   shared_x_comm_prev         shared buffer allocated on device
 * \param [in]   shared_x_comm_curr         shared buffer allocated on device
 * \param [in]   shared_x_comm_next         shared buffer allocated on device
 * \param [in]   shared_x_comm_prev_host    shared buffer for x_tilde allocated on host for non-device-aware communication
 * \param [in]   shared_x_comm_curr_host    shared buffer for x_tilde allocated on host for non-device-aware communication
 * \param [in]   shared_x_comm_next_host    shared buffer for x_tilde allocated on host for non-device-aware communication
 * \param [in]   shared_x_tilde_send_host   shared buffer for x_tilde allocated on host for non-device-aware communication
 * \param [in]   shared_x_tilde_recv_host   shared buffer for x_tilde allocated on host for non-device-aware communication
 * \note This function call does not free existing memory and may cause memory leak
 * \note All buffers must be disjoint of size 2 * Nj * Nk
 * \note For device-aware communication shared_x_comm_prev_host, shared_x_comm_curr_host, and shared_x_comm_curr_host must be nullptr
 */
template<typename RealType, typename RealTypeComm, typename MemSpaceType>
void DistPentaSolDimI<RealType, RealTypeComm, MemSpaceType>::resetSharedBuffers(
        RealType*     shared_x_tilde_cur,
        RealType*     shared_x_tilde_nbr,
        RealType*     shared_x_tilde_buf,
        RealTypeComm* shared_x_comm_prev,
        RealTypeComm* shared_x_comm_curr,
        RealTypeComm* shared_x_comm_next,
        RealTypeComm* shared_x_comm_prev_host,
        RealTypeComm* shared_x_comm_curr_host,
        RealTypeComm* shared_x_comm_next_host,
        RealType*     shared_x_tilde_send_host,
        RealType*     shared_x_tilde_recv_host
    )
{
    this->x_tilde_cur      = shared_x_tilde_cur;
    this->x_tilde_nbr      = shared_x_tilde_nbr;
    this->x_tilde_buf      = shared_x_tilde_buf;
    this->x_comm_prev      = shared_x_comm_prev;
    this->x_comm_curr      = shared_x_comm_curr;
    this->x_comm_next      = shared_x_comm_next;
    this->x_comm_prev_host = shared_x_comm_prev_host;
    this->x_comm_curr_host = shared_x_comm_curr_host;
    this->x_comm_next_host = shared_x_comm_next_host;
    this->x_tilde_send_host = shared_x_tilde_send_host;
    this->x_tilde_recv_host = shared_x_tilde_recv_host;
}



/*!
 * Reset penta-diagonal system
 * \param [in]    l2    2nd lower-diagonal entries
 * \param [in]    l1    1st lower-diagonal entries
 * \param [in]    d     diagonal entries
 * \param [in]    u1    1st upper-diagonal entries
 * \param [in]    u2    2nd upper-diagonal entries
 * \note All input buffers are allocated on the host memory
 * \note All data in the input buffers will be destroyed
 * \note All data in the input buffers are for the partitioned system in the current rank
 */
template<typename RealType, typename RealTypeComm, typename MemSpaceType>
void DistPentaSolDimI<RealType, RealTypeComm, MemSpaceType>::resetSystem(
        RealType* l2, RealType* l1, RealType* d, RealType* u1, RealType* u2
    )
{
    int Np;
    MPI_Comm_size(this->comm, &Np);
    penta::factPartitionedPenta<MemSpaceType, RealType>(
        this->fact_local_prev_2,
        this->fact_local_prev_1,
        this->fact_local_curr,
        this->fact_local_next_1,
        this->fact_local_next_2,
        this->fact_dist_prev,
        this->fact_dist_curr,
        this->fact_dist_next,
        this->Si,
        this->Ri,
        this->Li_tilde_tail,
        this->Ui_tilde_head,
        l2, l1, d, u1, u2,
        this->NI,
        static_cast<unsigned int>(Np),
        MAX_LOCAL_FACT_SIZE,
        this->comm 
    );
}



/*!
 * Solve the system
 * \param [in, out]    x    right-hand side as input and solution as output
 */
template<typename RealType, typename RealTypeComm, typename MemSpaceType>
void DistPentaSolDimI<RealType, RealTypeComm, MemSpaceType>::solve(RealType* x) {
    #ifdef COMPACK3D_DEVICE_COMM_ENABLED
    constexpr bool USE_DEVICE_AWARE_COMM = true;
    #else
    constexpr bool USE_DEVICE_AWARE_COMM = std::is_same<MemSpaceType, MemSpace::Host>::value;
    #endif

    int rank, Np;
    MPI_Comm_rank(this->comm, &rank);
    MPI_Comm_size(this->comm, &Np);
    const int rank_prev = (rank + Np - 1) % Np;
    const int rank_next = (rank      + 1) % Np;

    RealType* x_loc = &x[2 * this->ARR_STRIDE_I];
    const unsigned int N_local = this->NI - 2;
    penta::localSolPentaPCRDimI<RealType>(
            x_loc,
            this->fact_local_prev_2,
            this->fact_local_prev_1,
            this->fact_local_curr,
            this->fact_local_next_1,
            this->fact_local_next_2,
            MAX_LOCAL_FACT_SIZE,
            N_local, this->NJ, this->NK,
            this->ARR_STRIDE_I, this->ARR_STRIDE_J
    );
    copySlicesFrom3DArrayDimI<RealType>(
            this->x_tilde_buf, &x_loc[(N_local-2) * this->ARR_STRIDE_I],
            2, this->NJ, this->NK, this->ARR_STRIDE_I, this->ARR_STRIDE_J);

    if constexpr (USE_DEVICE_AWARE_COMM) {
        deviceFence();
        MPI_Irecv(this->x_tilde_nbr, 2*this->NJ*this->NK, MPIDataType<RealType>::value, rank_prev, 1, this->comm, &this->mpi_reqs[0]);
        MPI_Isend(this->x_tilde_buf, 2*this->NJ*this->NK, MPIDataType<RealType>::value, rank_next, 1, this->comm, &this->mpi_reqs[1]);
    } else {
        deepCopy<MemSpace::Host, MemSpaceType, RealType>(this->x_tilde_send_host, this->x_tilde_buf, 2*this->NJ*this->NK);
        MPI_Irecv (this->x_tilde_recv_host, 2*this->NJ*this->NK, MPIDataType<RealType>::value, rank_prev, 1, this->comm, &this->mpi_reqs[0]);
        MPI_Issend(this->x_tilde_send_host, 2*this->NJ*this->NK, MPIDataType<RealType>::value, rank_next, 1, this->comm, &this->mpi_reqs[1]);
    }

    copySlicesFrom3DArrayDimI<RealType>(this->x_tilde_cur, x, 2, this->NJ, this->NK, this->ARR_STRIDE_I, this->ARR_STRIDE_J);

    MPI_Waitall(2, this->mpi_reqs, this->mpi_stats);
    if constexpr (!USE_DEVICE_AWARE_COMM) {
        deepCopy<MemSpaceType, MemSpace::Host, RealType>(this->x_tilde_nbr, this->x_tilde_recv_host, 2*this->NJ*this->NK);
    }

    copySlicesFrom3DArrayDimI<RealType>(this->x_tilde_buf, x_loc, 2, this->NJ, this->NK, this->ARR_STRIDE_I, this->ARR_STRIDE_J);
    penta::calcReducedSystemRHSLocal(this->x_tilde_cur, this->x_tilde_nbr, this->x_tilde_buf, this->Li_tilde_tail, this->Ui_tilde_head, this->NJ*this->NK);
    penta::distSolve<RealType, RealTypeComm, MemSpaceType>(
            this->x_tilde_cur,
            this->x_comm_prev,
            this->x_comm_curr,
            this->x_comm_next,
            this->fact_dist_prev,
            this->fact_dist_curr,
            this->fact_dist_next,
            this->NJ*this->NK, Np,
            this->comm, this->mpi_reqs, this->mpi_stats,
            this->x_comm_prev_host,
            this->x_comm_curr_host,
            this->x_comm_next_host
    );

    if constexpr (USE_DEVICE_AWARE_COMM) {
        deviceFence();
        MPI_Irecv(this->x_tilde_nbr, 2*this->NJ*this->NK, MPIDataType<RealType>::value, rank_next, 1, this->comm, &this->mpi_reqs[0]);
        MPI_Isend(this->x_tilde_cur, 2*this->NJ*this->NK, MPIDataType<RealType>::value, rank_prev, 1, this->comm, &this->mpi_reqs[1]);
    } else {
        deepCopy<MemSpace::Host, MemSpaceType, RealType>(this->x_tilde_send_host, this->x_tilde_cur, 2*this->NJ*this->NK);
        MPI_Irecv (this->x_tilde_recv_host, 2*this->NJ*this->NK, MPIDataType<RealType>::value, rank_next, 1, this->comm, &this->mpi_reqs[0]);
        MPI_Issend(this->x_tilde_send_host, 2*this->NJ*this->NK, MPIDataType<RealType>::value, rank_prev, 1, this->comm, &this->mpi_reqs[1]);
    }

    copySlicesTo3DArrayDimI<RealType>(x, this->x_tilde_cur, 2, this->NJ, this->NK, this->ARR_STRIDE_I, this->ARR_STRIDE_J);

    MPI_Waitall(2, this->mpi_reqs, this->mpi_stats);
    if constexpr (!USE_DEVICE_AWARE_COMM) {
        deepCopy<MemSpaceType, MemSpace::Host, RealType>(this->x_tilde_nbr, this->x_tilde_recv_host, 2*this->NJ*this->NK);
    }

    penta::updateLocalSolDimI<RealType>(x_loc, this->x_tilde_cur, this->x_tilde_nbr, this->Si, this->Ri, this->NI-2, this->NJ, this->NK, this->ARR_STRIDE_I, this->ARR_STRIDE_J);
}



/*!
 * Constructor of distributed solver of penta-diagonal system
 * \tparam RealType        real-value type for primitive data type
 * \tparam RealTypeComm    real-value type for communication on distributed memory used in solving the reduced system
 * \tparam MemSpaceType    memory space, valid values are cmpk::MemSpace::Host and cmpk::MemSpace::Device
 * \param [in]   num_grids_i    number of grid points in i-direction in the current partition
 * \param [in]   num_grids_j    number of grid points in j-direction in the current partition
 * \param [in]   num_grids_k    number of grid points in k-direction in the current partition
 * \param [in]  arr_stride_i    number of entries strided in i-direction in the current partition, i.e., number of entries from i to i+1
 * \param [in]  arr_stride_j    number of entries strided in j-direction in the current partition, i.e., number of entries from j to j+1
 * \param [in]      mpi_comm    a MPI-sub-communicator that only contains the partitions involved in the system
 * \note k-direction is mapped to a contiguous memory layout
 * \note The description of the data size and layout are based on the current distributed partition
 */
template<typename RealType, typename RealTypeComm, typename MemSpaceType>
DistPentaSolDimJ<RealType, RealTypeComm, MemSpaceType>::DistPentaSolDimJ(
        const unsigned int num_grids_i,
        const unsigned int num_grids_j,
        const unsigned int num_grids_k,
        const unsigned int arr_stride_i,
        const unsigned int arr_stride_j,
        MPI_Comm mpi_comm) :
    NI           ( num_grids_i),
    NJ           ( num_grids_j),
    NK           ( num_grids_k),
    ARR_STRIDE_I (arr_stride_i),
    ARR_STRIDE_J (arr_stride_j)
{
    if ((this->NI < 1) || (this->NJ < 4) || (this->NK < 1)) throw std::invalid_argument("[Compack3D] Partitioned data size is invalid.");
    this->comm = mpi_comm;
    int Np;
    MPI_Comm_size(mpi_comm, &Np);
    this->fact_local_prev_2 = nullptr;
    this->fact_local_prev_1 = nullptr;
    this->fact_local_curr   = nullptr;
    this->fact_local_next_1 = nullptr;
    this->fact_local_next_2 = nullptr;
    this->fact_dist_prev    = nullptr;
    this->fact_dist_curr    = nullptr;
    this->fact_dist_next    = nullptr;
    this->Si                = nullptr;
    this->Ri                = nullptr;
    this->Li_tilde_tail     = nullptr;
    this->Ui_tilde_head     = nullptr;
    penta::allocFactBuffers<MemSpaceType, RealType>(
            &this->fact_local_prev_2,
            &this->fact_local_prev_1,
            &this->fact_local_curr,
            &this->fact_local_next_1,
            &this->fact_local_next_2,
            &this->fact_dist_prev,
            &this->fact_dist_curr,
            &this->fact_dist_next,
            &this->Si,
            &this->Ri,
            &this->Li_tilde_tail,
            &this->Ui_tilde_head,
            num_grids_j, Np);
}



/*!
 * Destructor
 */
template<typename RealType, typename RealTypeComm, typename MemSpaceType>
DistPentaSolDimJ<RealType, RealTypeComm, MemSpaceType>::~DistPentaSolDimJ() {
    memFreeArray<MemSpaceType, RealType>(this->fact_local_prev_2);
    memFreeArray<MemSpaceType, RealType>(this->fact_local_prev_1);
    memFreeArray<MemSpaceType, RealType>(this->fact_local_curr);
    memFreeArray<MemSpaceType, RealType>(this->fact_local_next_1);
    memFreeArray<MemSpaceType, RealType>(this->fact_local_next_2);
    memFreeArray<MemSpaceType, RealType>(this->fact_dist_prev);
    memFreeArray<MemSpaceType, RealType>(this->fact_dist_curr);
    memFreeArray<MemSpaceType, RealType>(this->fact_dist_next);
    memFreeArray<MemSpaceType, RealType>(this->Si);
    memFreeArray<MemSpaceType, RealType>(this->Ri);
    memFreeArray<MemSpaceType, RealType>(this->Li_tilde_tail);
    memFreeArray<MemSpaceType, RealType>(this->Ui_tilde_head);
}


/*!
 * Detach all shared buffers
 * \note This function call does not free allocated memory and may cause memory leak
 */
template<typename RealType, typename RealTypeComm, typename MemSpaceType>
void DistPentaSolDimJ<RealType, RealTypeComm, MemSpaceType>::detachSharedBuffers() {
    this->x_tilde_cur       = nullptr;
    this->x_tilde_nbr       = nullptr;
    this->x_tilde_buf       = nullptr;
    this->x_comm_prev       = nullptr;
    this->x_comm_curr       = nullptr;
    this->x_comm_next       = nullptr;
    this->x_comm_prev_host  = nullptr;
    this->x_comm_curr_host  = nullptr;
    this->x_comm_next_host  = nullptr;
    this->x_tilde_send_host = nullptr;
    this->x_tilde_recv_host = nullptr;
}



/*!
 * Reset shared buffers
 * \param [in]   shared_x_tilde_cur         shared buffer allocated on device
 * \param [in]   shared_x_tilde_nbr         shared buffer allocated on device
 * \param [in]   shared_x_tilde_buf         shared buffer allocated on device
 * \param [in]   shared_x_comm_prev         shared buffer allocated on device
 * \param [in]   shared_x_comm_curr         shared buffer allocated on device
 * \param [in]   shared_x_comm_next         shared buffer allocated on device
 * \param [in]   shared_x_comm_prev_host    shared buffer for x_tilde allocated on host for non-device-aware communication
 * \param [in]   shared_x_comm_curr_host    shared buffer for x_tilde allocated on host for non-device-aware communication
 * \param [in]   shared_x_comm_next_host    shared buffer for x_tilde allocated on host for non-device-aware communication
 * \param [in]   shared_x_tilde_send_host   shared buffer for x_tilde allocated on host for non-device-aware communication
 * \param [in]   shared_x_tilde_recv_host   shared buffer for x_tilde allocated on host for non-device-aware communication
 * \note This function call does not free existing memory and may cause memory leak
 * \note All buffers must be disjoint of size 2 * Nj * Nk
 * \note For device-aware communication shared_x_comm_prev_host, shared_x_comm_curr_host, and shared_x_comm_curr_host must be nullptr
 */
template<typename RealType, typename RealTypeComm, typename MemSpaceType>
void DistPentaSolDimJ<RealType, RealTypeComm, MemSpaceType>::resetSharedBuffers(
        RealType*     shared_x_tilde_cur,
        RealType*     shared_x_tilde_nbr,
        RealType*     shared_x_tilde_buf,
        RealTypeComm* shared_x_comm_prev,
        RealTypeComm* shared_x_comm_curr,
        RealTypeComm* shared_x_comm_next,
        RealTypeComm* shared_x_comm_prev_host,
        RealTypeComm* shared_x_comm_curr_host,
        RealTypeComm* shared_x_comm_next_host,
        RealType*     shared_x_tilde_send_host,
        RealType*     shared_x_tilde_recv_host
    )
{
    this->x_tilde_cur      = shared_x_tilde_cur;
    this->x_tilde_nbr      = shared_x_tilde_nbr;
    this->x_tilde_buf      = shared_x_tilde_buf;
    this->x_comm_prev      = shared_x_comm_prev;
    this->x_comm_curr      = shared_x_comm_curr;
    this->x_comm_next      = shared_x_comm_next;
    this->x_comm_prev_host = shared_x_comm_prev_host;
    this->x_comm_curr_host = shared_x_comm_curr_host;
    this->x_comm_next_host = shared_x_comm_next_host;
    this->x_tilde_send_host = shared_x_tilde_send_host;
    this->x_tilde_recv_host = shared_x_tilde_recv_host;
}



/*!
 * Reset penta-diagonal system
 * \param [in]    l2    2nd lower-diagonal entries
 * \param [in]    l1    1st lower-diagonal entries
 * \param [in]    d     diagonal entries
 * \param [in]    u1    1st upper-diagonal entries
 * \param [in]    u2    2nd upper-diagonal entries
 * \note All input buffers are allocated on the host memory
 * \note All data in the input buffers will be destroyed
 * \note All data in the input buffers are for the partitioned system in the current rank
 */
template<typename RealType, typename RealTypeComm, typename MemSpaceType>
void DistPentaSolDimJ<RealType, RealTypeComm, MemSpaceType>::resetSystem(
        RealType* l2, RealType* l1, RealType* d, RealType* u1, RealType* u2
    )
{
    int Np;
    MPI_Comm_size(this->comm, &Np);
    penta::factPartitionedPenta<MemSpaceType, RealType>(
        this->fact_local_prev_2,
        this->fact_local_prev_1,
        this->fact_local_curr,
        this->fact_local_next_1,
        this->fact_local_next_2,
        this->fact_dist_prev,
        this->fact_dist_curr,
        this->fact_dist_next,
        this->Si,
        this->Ri,
        this->Li_tilde_tail,
        this->Ui_tilde_head,
        l2, l1, d, u1, u2,
        this->NJ,
        static_cast<unsigned int>(Np),
        MAX_LOCAL_FACT_SIZE,
        this->comm 
    );
}



/*!
 * Solve the system
 * \param [in, out]    x    right-hand side as input and solution as output
 */
template<typename RealType, typename RealTypeComm, typename MemSpaceType>
void DistPentaSolDimJ<RealType, RealTypeComm, MemSpaceType>::solve(RealType* x) {
    #ifdef COMPACK3D_DEVICE_COMM_ENABLED
    constexpr bool USE_DEVICE_AWARE_COMM = true;
    #else
    constexpr bool USE_DEVICE_AWARE_COMM = std::is_same<MemSpaceType, MemSpace::Host>::value;
    #endif

    int rank, Np;
    MPI_Comm_rank(this->comm, &rank);
    MPI_Comm_size(this->comm, &Np);
    const int rank_prev = (rank + Np - 1) % Np;
    const int rank_next = (rank      + 1) % Np;

    RealType* x_loc = &x[2 * this->ARR_STRIDE_J];
    const unsigned int N_local = this->NJ - 2;
    penta::localSolPentaPCRDimJ<RealType>(
            x_loc,
            this->fact_local_prev_2,
            this->fact_local_prev_1,
            this->fact_local_curr,
            this->fact_local_next_1,
            this->fact_local_next_2,
            MAX_LOCAL_FACT_SIZE,
            this->NI, N_local, this->NK,
            this->ARR_STRIDE_I, this->ARR_STRIDE_J
    );
    copySlicesFrom3DArrayDimJ<RealType>(
            this->x_tilde_buf, &x_loc[(N_local-2) * this->ARR_STRIDE_J],
            this->NI, 2, this->NK, this->ARR_STRIDE_I, this->ARR_STRIDE_J);

    if constexpr (USE_DEVICE_AWARE_COMM) {
        deviceFence();
        MPI_Irecv(this->x_tilde_nbr, 2*this->NI*this->NK, MPIDataType<RealType>::value, rank_prev, 204, this->comm, &this->mpi_reqs[0]);
        MPI_Isend(this->x_tilde_buf, 2*this->NI*this->NK, MPIDataType<RealType>::value, rank_next, 204, this->comm, &this->mpi_reqs[1]);
    } else {
        deepCopy<MemSpace::Host, MemSpaceType, RealType>(this->x_tilde_send_host, this->x_tilde_buf, 2*this->NI*this->NK);
        MPI_Irecv (this->x_tilde_recv_host, 2*this->NI*this->NK, MPIDataType<RealType>::value, rank_prev, 204, this->comm, &this->mpi_reqs[0]);
        MPI_Issend(this->x_tilde_send_host, 2*this->NI*this->NK, MPIDataType<RealType>::value, rank_next, 204, this->comm, &this->mpi_reqs[1]);
    }

    copySlicesFrom3DArrayDimJ<RealType>(this->x_tilde_cur, x, this->NI, 2, this->NK, this->ARR_STRIDE_I, this->ARR_STRIDE_J);

    MPI_Waitall(2, this->mpi_reqs, this->mpi_stats);
    if constexpr (!USE_DEVICE_AWARE_COMM) {
        deepCopy<MemSpaceType, MemSpace::Host, RealType>(this->x_tilde_nbr, this->x_tilde_recv_host, 2*this->NI*this->NK);
    }

    copySlicesFrom3DArrayDimJ<RealType>(this->x_tilde_buf, x_loc, this->NI, 2, this->NK, this->ARR_STRIDE_I, this->ARR_STRIDE_J);
    penta::calcReducedSystemRHSLocal(this->x_tilde_cur, this->x_tilde_nbr, this->x_tilde_buf, this->Li_tilde_tail, this->Ui_tilde_head, this->NI*this->NK);
    penta::distSolve<RealType, RealTypeComm, MemSpaceType>(
            this->x_tilde_cur,
            this->x_comm_prev,
            this->x_comm_curr,
            this->x_comm_next,
            this->fact_dist_prev,
            this->fact_dist_curr,
            this->fact_dist_next,
            this->NI*this->NK, Np,
            this->comm, this->mpi_reqs, this->mpi_stats,
            this->x_comm_prev_host,
            this->x_comm_curr_host,
            this->x_comm_next_host
    );

    if constexpr (USE_DEVICE_AWARE_COMM) {
        deviceFence();
        MPI_Irecv(this->x_tilde_nbr, 2*this->NI*this->NK, MPIDataType<RealType>::value, rank_next, 204, this->comm, &this->mpi_reqs[0]);
        MPI_Isend(this->x_tilde_cur, 2*this->NI*this->NK, MPIDataType<RealType>::value, rank_prev, 204, this->comm, &this->mpi_reqs[1]);
    } else  {
        deepCopy<MemSpace::Host, MemSpaceType, RealType>(this->x_tilde_send_host, this->x_tilde_cur, 2*this->NI*this->NK);
        MPI_Irecv (this->x_tilde_recv_host, 2*this->NI*this->NK, MPIDataType<RealType>::value, rank_next, 204, this->comm, &this->mpi_reqs[0]);
        MPI_Issend(this->x_tilde_send_host, 2*this->NI*this->NK, MPIDataType<RealType>::value, rank_prev, 204, this->comm, &this->mpi_reqs[1]);
    }

    copySlicesTo3DArrayDimJ<RealType>(x, this->x_tilde_cur, this->NI, 2, this->NK, this->ARR_STRIDE_I, this->ARR_STRIDE_J);

    MPI_Waitall(2, this->mpi_reqs, this->mpi_stats);
    if constexpr (!USE_DEVICE_AWARE_COMM) {
        deepCopy<MemSpaceType, MemSpace::Host, RealType>(this->x_tilde_nbr, this->x_tilde_recv_host, 2*this->NI*this->NK);
    }


    penta::updateLocalSolDimJ<RealType>(x_loc, this->x_tilde_cur, this->x_tilde_nbr, this->Si, this->Ri, this->NI, this->NJ-2, this->NK, this->ARR_STRIDE_I, this->ARR_STRIDE_J);
}



/*!
 * Constructor of distributed solver of penta-diagonal system
 * \tparam RealType        real-value type for primitive data type
 * \tparam RealTypeComm    real-value type for communication on distributed memory used in solving the reduced system
 * \tparam MemSpaceType    memory space, valid values are cmpk::MemSpace::Host and cmpk::MemSpace::Device
 * \param [in]   num_grids_i    number of grid points in i-direction in the current partition
 * \param [in]   num_grids_j    number of grid points in j-direction in the current partition
 * \param [in]   num_grids_k    number of grid points in k-direction in the current partition
 * \param [in]  arr_stride_i    number of entries strided in i-direction in the current partition, i.e., number of entries from i to i+1
 * \param [in]  arr_stride_j    number of entries strided in j-direction in the current partition, i.e., number of entries from j to j+1
 * \param [in]      mpi_comm    a MPI-sub-communicator that only contains the partitions involved in the system
 * \note k-direction is mapped to a contiguous memory layout
 * \note The description of the data size and layout are based on the current distributed partition
 */
template<typename RealType, typename RealTypeComm, typename MemSpaceType>
DistPentaSolDimK<RealType, RealTypeComm, MemSpaceType>::DistPentaSolDimK(
        const unsigned int num_grids_i,
        const unsigned int num_grids_j,
        const unsigned int num_grids_k,
        const unsigned int arr_stride_i,
        const unsigned int arr_stride_j,
        MPI_Comm mpi_comm) :
    NI           ( num_grids_i),
    NJ           ( num_grids_j),
    NK           ( num_grids_k),
    ARR_STRIDE_I (arr_stride_i),
    ARR_STRIDE_J (arr_stride_j)
{
    if ((this->NI < 1) || (this->NJ < 4) || (this->NK < 1)) throw std::invalid_argument("[Compack3D] Partitioned data size is invalid.");
    this->comm = mpi_comm;
    int Np;
    MPI_Comm_size(mpi_comm, &Np);
    this->fact_local_prev_2 = nullptr;
    this->fact_local_prev_1 = nullptr;
    this->fact_local_curr   = nullptr;
    this->fact_local_next_1 = nullptr;
    this->fact_local_next_2 = nullptr;
    this->fact_dist_prev    = nullptr;
    this->fact_dist_curr    = nullptr;
    this->fact_dist_next    = nullptr;
    this->Si                = nullptr;
    this->Ri                = nullptr;
    this->Li_tilde_tail     = nullptr;
    this->Ui_tilde_head     = nullptr;
    penta::allocFactBuffers<MemSpaceType, RealType>(
            &this->fact_local_prev_2,
            &this->fact_local_prev_1,
            &this->fact_local_curr,
            &this->fact_local_next_1,
            &this->fact_local_next_2,
            &this->fact_dist_prev,
            &this->fact_dist_curr,
            &this->fact_dist_next,
            &this->Si,
            &this->Ri,
            &this->Li_tilde_tail,
            &this->Ui_tilde_head,
            num_grids_k, Np);
}



/*!
 * Destructor
 */
template<typename RealType, typename RealTypeComm, typename MemSpaceType>
DistPentaSolDimK<RealType, RealTypeComm, MemSpaceType>::~DistPentaSolDimK() {
    memFreeArray<MemSpaceType, RealType>(this->fact_local_prev_2);
    memFreeArray<MemSpaceType, RealType>(this->fact_local_prev_1);
    memFreeArray<MemSpaceType, RealType>(this->fact_local_curr);
    memFreeArray<MemSpaceType, RealType>(this->fact_local_next_1);
    memFreeArray<MemSpaceType, RealType>(this->fact_local_next_2);
    memFreeArray<MemSpaceType, RealType>(this->fact_dist_prev);
    memFreeArray<MemSpaceType, RealType>(this->fact_dist_curr);
    memFreeArray<MemSpaceType, RealType>(this->fact_dist_next);
    memFreeArray<MemSpaceType, RealType>(this->Si);
    memFreeArray<MemSpaceType, RealType>(this->Ri);
    memFreeArray<MemSpaceType, RealType>(this->Li_tilde_tail);
    memFreeArray<MemSpaceType, RealType>(this->Ui_tilde_head);
}


/*!
 * Detach all shared buffers
 * \note This function call does not free allocated memory and may cause memory leak
 */
template<typename RealType, typename RealTypeComm, typename MemSpaceType>
void DistPentaSolDimK<RealType, RealTypeComm, MemSpaceType>::detachSharedBuffers() {
    this->x_tilde_cur       = nullptr;
    this->x_tilde_nbr       = nullptr;
    this->x_tilde_buf       = nullptr;
    this->x_comm_prev       = nullptr;
    this->x_comm_curr       = nullptr;
    this->x_comm_next       = nullptr;
    this->x_comm_prev_host  = nullptr;
    this->x_comm_curr_host  = nullptr;
    this->x_comm_next_host  = nullptr;
    this->x_tilde_send_host = nullptr;
    this->x_tilde_recv_host = nullptr;
}



/*!
 * Reset shared buffers
 * \param [in]   shared_x_tilde_cur         shared buffer allocated on device
 * \param [in]   shared_x_tilde_nbr         shared buffer allocated on device
 * \param [in]   shared_x_tilde_buf         shared buffer allocated on device
 * \param [in]   shared_x_comm_prev         shared buffer allocated on device
 * \param [in]   shared_x_comm_curr         shared buffer allocated on device
 * \param [in]   shared_x_comm_next         shared buffer allocated on device
 * \param [in]   shared_x_comm_prev_host    shared buffer for x_tilde allocated on host for non-device-aware communication
 * \param [in]   shared_x_comm_curr_host    shared buffer for x_tilde allocated on host for non-device-aware communication
 * \param [in]   shared_x_comm_next_host    shared buffer for x_tilde allocated on host for non-device-aware communication
 * \param [in]   shared_x_tilde_send_host   shared buffer for x_tilde allocated on host for non-device-aware communication
 * \param [in]   shared_x_tilde_recv_host   shared buffer for x_tilde allocated on host for non-device-aware communication
 * \note This function call does not free existing memory and may cause memory leak
 * \note All buffers must be disjoint of size 2 * Nj * Nk
 * \note For device-aware communication shared_x_comm_prev_host, shared_x_comm_curr_host, and shared_x_comm_curr_host must be nullptr
 */
template<typename RealType, typename RealTypeComm, typename MemSpaceType>
void DistPentaSolDimK<RealType, RealTypeComm, MemSpaceType>::resetSharedBuffers(
        RealType*     shared_x_tilde_cur,
        RealType*     shared_x_tilde_nbr,
        RealType*     shared_x_tilde_buf,
        RealTypeComm* shared_x_comm_prev,
        RealTypeComm* shared_x_comm_curr,
        RealTypeComm* shared_x_comm_next,
        RealTypeComm* shared_x_comm_prev_host,
        RealTypeComm* shared_x_comm_curr_host,
        RealTypeComm* shared_x_comm_next_host,
        RealType*     shared_x_tilde_send_host,
        RealType*     shared_x_tilde_recv_host
    )
{
    this->x_tilde_cur       = shared_x_tilde_cur;
    this->x_tilde_nbr       = shared_x_tilde_nbr;
    this->x_tilde_buf       = shared_x_tilde_buf;
    this->x_comm_prev       = shared_x_comm_prev;
    this->x_comm_curr       = shared_x_comm_curr;
    this->x_comm_next       = shared_x_comm_next;
    this->x_comm_prev_host  = shared_x_comm_prev_host;
    this->x_comm_curr_host  = shared_x_comm_curr_host;
    this->x_comm_next_host  = shared_x_comm_next_host;
    this->x_tilde_send_host = shared_x_tilde_send_host;
    this->x_tilde_recv_host = shared_x_tilde_recv_host;
}



/*!
 * Reset penta-diagonal system
 * \param [in]    l2    2nd lower-diagonal entries
 * \param [in]    l1    1st lower-diagonal entries
 * \param [in]    d     diagonal entries
 * \param [in]    u1    1st upper-diagonal entries
 * \param [in]    u2    2nd upper-diagonal entries
 * \note All input buffers are allocated on the host memory
 * \note All data in the input buffers will be destroyed
 * \note All data in the input buffers are for the partitioned system in the current rank
 */
template<typename RealType, typename RealTypeComm, typename MemSpaceType>
void DistPentaSolDimK<RealType, RealTypeComm, MemSpaceType>::resetSystem(
        RealType* l2, RealType* l1, RealType* d, RealType* u1, RealType* u2
    )
{
    int Np;
    MPI_Comm_size(this->comm, &Np);
    penta::factPartitionedPenta<MemSpaceType, RealType>(
        this->fact_local_prev_2,
        this->fact_local_prev_1,
        this->fact_local_curr,
        this->fact_local_next_1,
        this->fact_local_next_2,
        this->fact_dist_prev,
        this->fact_dist_curr,
        this->fact_dist_next,
        this->Si,
        this->Ri,
        this->Li_tilde_tail,
        this->Ui_tilde_head,
        l2, l1, d, u1, u2,
        this->NK,
        static_cast<unsigned int>(Np),
        MAX_LOCAL_FACT_SIZE,
        this->comm 
    );
}



/*!
 * Solve the system
 * \param [in, out]    x    right-hand side as input and solution as output
 */
template<typename RealType, typename RealTypeComm, typename MemSpaceType>
void DistPentaSolDimK<RealType, RealTypeComm, MemSpaceType>::solve(RealType* x) {
    #ifdef COMPACK3D_DEVICE_COMM_ENABLED
    constexpr bool USE_DEVICE_AWARE_COMM = true;
    #else
    constexpr bool USE_DEVICE_AWARE_COMM = std::is_same<MemSpaceType, MemSpace::Host>::value;
    #endif

    int rank, Np;
    MPI_Comm_rank(this->comm, &rank);
    MPI_Comm_size(this->comm, &Np);
    const int rank_prev = (rank + Np - 1) % Np;
    const int rank_next = (rank      + 1) % Np;

    RealType* x_loc = &x[2];
    const unsigned int N_local = this->NK - 2;
    penta::localSolPentaPCRDimK<RealType>(
            x_loc,
            this->fact_local_prev_2,
            this->fact_local_prev_1,
            this->fact_local_curr,
            this->fact_local_next_1,
            this->fact_local_next_2,
            MAX_LOCAL_FACT_SIZE,
            this->NI, this->NJ, N_local,
            this->ARR_STRIDE_I, this->ARR_STRIDE_J
    );
    copySlicesFrom3DArrayDimK<RealType>(
            this->x_tilde_buf, &x_loc[N_local-2],
            this->NI, this->NJ, 2, this->ARR_STRIDE_I, this->ARR_STRIDE_J);

    if constexpr (USE_DEVICE_AWARE_COMM) {
        deviceFence();
        MPI_Irecv(this->x_tilde_nbr, 2*this->NI*this->NJ, MPIDataType<RealType>::value, rank_prev, 204, this->comm, &this->mpi_reqs[0]);
        MPI_Isend(this->x_tilde_buf, 2*this->NI*this->NJ, MPIDataType<RealType>::value, rank_next, 204, this->comm, &this->mpi_reqs[1]);
    } else {
        deepCopy<MemSpace::Host, MemSpaceType, RealType>(this->x_tilde_send_host, this->x_tilde_buf, 2*this->NI*this->NJ);
        MPI_Irecv (this->x_tilde_recv_host, 2*this->NI*this->NJ, MPIDataType<RealType>::value, rank_prev, 204, this->comm, &this->mpi_reqs[0]);
        MPI_Issend(this->x_tilde_send_host, 2*this->NI*this->NJ, MPIDataType<RealType>::value, rank_next, 204, this->comm, &this->mpi_reqs[1]);
    }

    copySlicesFrom3DArrayDimK<RealType>(this->x_tilde_cur, x, this->NI, this->NJ, 2, this->ARR_STRIDE_I, this->ARR_STRIDE_J);

    MPI_Waitall(2, this->mpi_reqs, this->mpi_stats);
    if constexpr (!USE_DEVICE_AWARE_COMM) {
        deepCopy<MemSpaceType, MemSpace::Host, RealType>(this->x_tilde_nbr, this->x_tilde_recv_host, 2*this->NI*this->NJ);
    }

    copySlicesFrom3DArrayDimK<RealType>(this->x_tilde_buf, x_loc, this->NI, this->NJ, 2, this->ARR_STRIDE_I, this->ARR_STRIDE_J);
    penta::calcReducedSystemRHSLocal(this->x_tilde_cur, this->x_tilde_nbr, this->x_tilde_buf, this->Li_tilde_tail, this->Ui_tilde_head, this->NI*this->NJ);
    penta::distSolve<RealType, RealTypeComm, MemSpaceType>(
            this->x_tilde_cur,
            this->x_comm_prev,
            this->x_comm_curr,
            this->x_comm_next,
            this->fact_dist_prev,
            this->fact_dist_curr,
            this->fact_dist_next,
            this->NI*this->NJ, Np,
            this->comm, this->mpi_reqs, this->mpi_stats,
            this->x_comm_prev_host,
            this->x_comm_curr_host,
            this->x_comm_next_host
    );

    if constexpr (USE_DEVICE_AWARE_COMM) {
        deviceFence();
        MPI_Irecv(this->x_tilde_nbr, 2*this->NI*this->NJ, MPIDataType<RealType>::value, rank_next, 204, this->comm, &this->mpi_reqs[0]);
        MPI_Isend(this->x_tilde_cur, 2*this->NI*this->NJ, MPIDataType<RealType>::value, rank_prev, 204, this->comm, &this->mpi_reqs[1]);
    } else {
        deepCopy<MemSpace::Host, MemSpaceType, RealType>(this->x_tilde_send_host, this->x_tilde_cur, 2*this->NI*this->NJ);
        MPI_Irecv (this->x_tilde_recv_host, 2*this->NI*this->NJ, MPIDataType<RealType>::value, rank_next, 204, this->comm, &this->mpi_reqs[0]);
        MPI_Issend(this->x_tilde_send_host, 2*this->NI*this->NJ, MPIDataType<RealType>::value, rank_prev, 204, this->comm, &this->mpi_reqs[1]);
    }

    copySlicesTo3DArrayDimK<RealType>(x, this->x_tilde_cur, this->NI, this->NJ, 2, this->ARR_STRIDE_I, this->ARR_STRIDE_J);

    MPI_Waitall(2, this->mpi_reqs, this->mpi_stats);
    if constexpr (!USE_DEVICE_AWARE_COMM) {
        deepCopy<MemSpaceType, MemSpace::Host, RealType>(this->x_tilde_nbr, this->x_tilde_recv_host, 2*this->NI*this->NJ);
    }

    penta::updateLocalSolDimK<RealType>(x_loc, this->x_tilde_cur, this->x_tilde_nbr, this->Si, this->Ri, this->NI, this->NJ, this->NK-2, this->ARR_STRIDE_I, this->ARR_STRIDE_J);
}



/*!
 * Constructor of distributed solver of penta-diagonal system
 * \tparam RealType        real-value type for primitive data type
 * \tparam RealTypeComm    real-value type for communication on distributed memory used in solving the reduced system
 * \tparam MemSpaceType    memory space, valid values are cmpk::MemSpace::Host and cmpk::MemSpace::Device
 * \param [in]   num_grids_i    number of grid points in i-direction in the current partition
 * \param [in]   num_grids_j    number of grid points in j-direction in the current partition
 * \param [in]   num_grids_k    number of grid points in k-direction in the current partition
 * \param [in]  arr_stride_i    number of entries strided in i-direction in the current partition, i.e., number of entries from i to i+1
 * \param [in]  arr_stride_j    number of entries strided in j-direction in the current partition, i.e., number of entries from j to j+1
 * \param [in]      mpi_comm    a MPI-sub-communicator that only contains the partitions involved in the system
 * \note k-direction is mapped to a contiguous memory layout
 * \note The description of the data size and layout are based on the current distributed partition
 */
template<typename RealType, typename RealTypeComm, typename MemSpaceType>
DistTriSolDimI<RealType, RealTypeComm, MemSpaceType>::DistTriSolDimI(
        const unsigned int num_grids_i,
        const unsigned int num_grids_j,
        const unsigned int num_grids_k,
        const unsigned int arr_stride_i,
        const unsigned int arr_stride_j,
        MPI_Comm mpi_comm) :
    NI           ( num_grids_i),
    NJ           ( num_grids_j),
    NK           ( num_grids_k),
    ARR_STRIDE_I (arr_stride_i),
    ARR_STRIDE_J (arr_stride_j)
{
    if ((this->NI < 4) || (this->NJ < 1) || (this->NK < 1)) throw std::invalid_argument("[Compack3D] Partitioned data size is invalid.");
    this->comm = mpi_comm;
    int Np;
    MPI_Comm_size(mpi_comm, &Np);
    this->fact_local_prev = nullptr;
    this->fact_local_curr = nullptr;
    this->fact_local_next = nullptr;
    this->fact_dist_prev  = nullptr;
    this->fact_dist_curr  = nullptr;
    this->fact_dist_next  = nullptr;
    this->Si              = nullptr;
    this->Ri              = nullptr;
    tri::allocFactBuffers<MemSpaceType, RealType>(
            &this->fact_local_prev,
            &this->fact_local_curr,
            &this->fact_local_next,
            &this->fact_dist_prev,
            &this->fact_dist_curr,
            &this->fact_dist_next,
            &this->Si,
            &this->Ri,
            num_grids_i, Np);
}



/*!
 * Destructor
 */
template<typename RealType, typename RealTypeComm, typename MemSpaceType>
DistTriSolDimI<RealType, RealTypeComm, MemSpaceType>::~DistTriSolDimI() {
    memFreeArray<MemSpaceType  , RealType>(this->fact_local_prev);
    memFreeArray<MemSpaceType  , RealType>(this->fact_local_curr);
    memFreeArray<MemSpaceType  , RealType>(this->fact_local_next);
    memFreeArray<MemSpace::Host, RealType>(this->fact_dist_prev);
    memFreeArray<MemSpace::Host, RealType>(this->fact_dist_curr);
    memFreeArray<MemSpace::Host, RealType>(this->fact_dist_next);
    memFreeArray<MemSpaceType  , RealType>(this->Si);
    memFreeArray<MemSpaceType  , RealType>(this->Ri);
}


/*!
 * Detach all shared buffers
 * \note This function call does not free allocated memory and may cause memory leak
 */
template<typename RealType, typename RealTypeComm, typename MemSpaceType>
void DistTriSolDimI<RealType, RealTypeComm, MemSpaceType>::detachSharedBuffers() {
    this->x_tilde_cur       = nullptr;
    this->x_tilde_nbr       = nullptr;
    this->x_tilde_buf       = nullptr;
    this->x_comm_prev       = nullptr;
    this->x_comm_curr       = nullptr;
    this->x_comm_next       = nullptr;
    this->x_comm_prev_host  = nullptr;
    this->x_comm_curr_host  = nullptr;
    this->x_comm_next_host  = nullptr;
    this->x_tilde_send_host = nullptr;
    this->x_tilde_recv_host = nullptr;
}



/*!
 * Reset shared buffers
 * \param [in]   shared_x_tilde_cur         shared buffer allocated on device
 * \param [in]   shared_x_tilde_nbr         shared buffer allocated on device
 * \param [in]   shared_x_tilde_buf         shared buffer allocated on device
 * \param [in]   shared_x_comm_prev         shared buffer allocated on device
 * \param [in]   shared_x_comm_curr         shared buffer allocated on device
 * \param [in]   shared_x_comm_next         shared buffer allocated on device
 * \param [in]   shared_x_comm_prev_host    shared buffer for x_tilde allocated on host for non-device-aware communication
 * \param [in]   shared_x_comm_curr_host    shared buffer for x_tilde allocated on host for non-device-aware communication
 * \param [in]   shared_x_comm_next_host    shared buffer for x_tilde allocated on host for non-device-aware communication
 * \param [in]   shared_x_tilde_send_host   shared buffer for x_tilde allocated on host for non-device-aware communication
 * \param [in]   shared_x_tilde_recv_host   shared buffer for x_tilde allocated on host for non-device-aware communication
 * \note This function call does not free existing memory and may cause memory leak
 * \note All buffers must be disjoint of size 2 * Nj * Nk
 * \note For device-aware communication shared_x_comm_prev_host, shared_x_comm_curr_host, and shared_x_comm_curr_host must be nullptr
 */
template<typename RealType, typename RealTypeComm, typename MemSpaceType>
void DistTriSolDimI<RealType, RealTypeComm, MemSpaceType>::resetSharedBuffers(
        RealType*     shared_x_tilde_cur,
        RealType*     shared_x_tilde_nbr,
        RealType*     shared_x_tilde_buf,
        RealTypeComm* shared_x_comm_prev,
        RealTypeComm* shared_x_comm_curr,
        RealTypeComm* shared_x_comm_next,
        RealTypeComm* shared_x_comm_prev_host,
        RealTypeComm* shared_x_comm_curr_host,
        RealTypeComm* shared_x_comm_next_host,
        RealType*     shared_x_tilde_send_host,
        RealType*     shared_x_tilde_recv_host
    )
{
    this->x_tilde_cur       = shared_x_tilde_cur;
    this->x_tilde_nbr       = shared_x_tilde_nbr;
    this->x_tilde_buf       = shared_x_tilde_buf;
    this->x_comm_prev       = shared_x_comm_prev;
    this->x_comm_curr       = shared_x_comm_curr;
    this->x_comm_next       = shared_x_comm_next;
    this->x_comm_prev_host  = shared_x_comm_prev_host;
    this->x_comm_curr_host  = shared_x_comm_curr_host;
    this->x_comm_next_host  = shared_x_comm_next_host;
    this->x_tilde_send_host = shared_x_tilde_send_host;
    this->x_tilde_recv_host = shared_x_tilde_recv_host;
}



/*!
 * Reset penta-diagonal system
 * \param [in]    l     lower-diagonal entries
 * \param [in]    d     diagonal entries
 * \param [in]    u     upper-diagonal entries
 * \note All input buffers are allocated on the host memory
 * \note All data in the input buffers will be destroyed
 * \note All data in the input buffers are for the partitioned system in the current rank
 */
template<typename RealType, typename RealTypeComm, typename MemSpaceType>
void DistTriSolDimI<RealType, RealTypeComm, MemSpaceType>::resetSystem(
        RealType* l, RealType* d, RealType* u
    )
{
    int Np;
    MPI_Comm_size(this->comm, &Np);
    tri::factPartitionedTri<MemSpaceType, RealType>(
        this->fact_local_prev,
        this->fact_local_curr,
        this->fact_local_next,
        this->fact_dist_prev,
        this->fact_dist_curr,
        this->fact_dist_next,
        this->Si,
        this->Ri,
        this->Li_tilde_tail,
        this->Ui_tilde_head,
        l, d, u,
        this->NI,
        static_cast<unsigned int>(Np),
        MAX_LOCAL_FACT_SIZE,
        this->comm 
    );
}



/*!
 * Solve the system
 * \param [in, out]    x    right-hand side as input and solution as output
 */
template<typename RealType, typename RealTypeComm, typename MemSpaceType>
void DistTriSolDimI<RealType, RealTypeComm, MemSpaceType>::solve(RealType* x) {
    #ifdef COMPACK3D_DEVICE_COMM_ENABLED
    constexpr bool USE_DEVICE_AWARE_COMM = true;
    #else
    constexpr bool USE_DEVICE_AWARE_COMM = std::is_same<MemSpaceType, MemSpace::Host>::value;
    #endif

    int rank, Np;
    MPI_Comm_rank(this->comm, &rank);
    MPI_Comm_size(this->comm, &Np);
    const int rank_prev = (rank + Np - 1) % Np;
    const int rank_next = (rank      + 1) % Np;

    RealType* x_loc = &x[this->ARR_STRIDE_I];
    const unsigned int N_local = this->NI - 1;
    tri::localSolTriPCRDimI<RealType>(
            x_loc,
            this->fact_local_prev,
            this->fact_local_curr,
            this->fact_local_next,
            MAX_LOCAL_FACT_SIZE,
            N_local, this->NJ, this->NK,
            this->ARR_STRIDE_I, this->ARR_STRIDE_J
    );
    copySlicesFrom3DArrayDimI<RealType>(
            this->x_tilde_buf, &x_loc[(N_local-1) * this->ARR_STRIDE_I],
            1, this->NJ, this->NK, this->ARR_STRIDE_I, this->ARR_STRIDE_J);

    if constexpr (USE_DEVICE_AWARE_COMM) {
        deviceFence();
        MPI_Irecv(this->x_tilde_nbr, this->NJ*this->NK, MPIDataType<RealType>::value, rank_prev, 1, this->comm, &this->mpi_reqs[0]);
        MPI_Isend(this->x_tilde_buf, this->NJ*this->NK, MPIDataType<RealType>::value, rank_next, 1, this->comm, &this->mpi_reqs[1]);
    } else {
        deepCopy<MemSpace::Host, MemSpaceType, RealType>(this->x_tilde_send_host, this->x_tilde_buf, this->NJ*this->NK);
        MPI_Irecv (this->x_tilde_recv_host, this->NJ*this->NK, MPIDataType<RealType>::value, rank_prev, 1, this->comm, &this->mpi_reqs[0]);
        MPI_Issend(this->x_tilde_send_host, this->NJ*this->NK, MPIDataType<RealType>::value, rank_next, 1, this->comm, &this->mpi_reqs[1]);
    }

    copySlicesFrom3DArrayDimI<RealType>(this->x_tilde_cur, x, 1, this->NJ, this->NK, this->ARR_STRIDE_I, this->ARR_STRIDE_J);

    MPI_Waitall(2, this->mpi_reqs, this->mpi_stats);
    if constexpr (!USE_DEVICE_AWARE_COMM) {
        deepCopy<MemSpaceType, MemSpace::Host, RealType>(this->x_tilde_nbr, this->x_tilde_recv_host, this->NJ*this->NK);
    }

    copySlicesFrom3DArrayDimI<RealType>(this->x_tilde_buf, x_loc, 1, this->NJ, this->NK, this->ARR_STRIDE_I, this->ARR_STRIDE_J);
    tri::calcReducedSystemRHSLocal(this->x_tilde_cur, this->x_tilde_nbr, this->x_tilde_buf, this->Li_tilde_tail, this->Ui_tilde_head, this->NJ*this->NK);
    tri::distSolve<RealType, RealTypeComm, MemSpaceType>(
            this->x_tilde_cur,
            this->x_comm_prev,
            this->x_comm_curr,
            this->x_comm_next,
            this->fact_dist_prev,
            this->fact_dist_curr,
            this->fact_dist_next,
            this->NJ*this->NK, Np,
            this->comm, this->mpi_reqs, this->mpi_stats,
            this->x_comm_prev_host,
            this->x_comm_curr_host,
            this->x_comm_next_host
    );

    if constexpr (USE_DEVICE_AWARE_COMM) {
        deviceFence();
        MPI_Irecv(this->x_tilde_nbr, this->NJ*this->NK, MPIDataType<RealType>::value, rank_next, 1, this->comm, &this->mpi_reqs[0]);
        MPI_Isend(this->x_tilde_cur, this->NJ*this->NK, MPIDataType<RealType>::value, rank_prev, 1, this->comm, &this->mpi_reqs[1]);
    } else {
        deepCopy<MemSpace::Host, MemSpaceType, RealType>(this->x_tilde_send_host, this->x_tilde_cur, this->NJ*this->NK);
        MPI_Irecv (this->x_tilde_recv_host, this->NJ*this->NK, MPIDataType<RealType>::value, rank_next, 1, this->comm, &this->mpi_reqs[0]);
        MPI_Issend(this->x_tilde_send_host, this->NJ*this->NK, MPIDataType<RealType>::value, rank_prev, 1, this->comm, &this->mpi_reqs[1]);
    }

    copySlicesTo3DArrayDimI<RealType>(x, this->x_tilde_cur, 1, this->NJ, this->NK, this->ARR_STRIDE_I, this->ARR_STRIDE_J);

    MPI_Waitall(2, this->mpi_reqs, this->mpi_stats);
    if constexpr (!USE_DEVICE_AWARE_COMM) {
        deepCopy<MemSpaceType, MemSpace::Host, RealType>(this->x_tilde_nbr, this->x_tilde_recv_host, this->NJ*this->NK);
    }

    tri::updateLocalSolDimI<RealType>(x_loc, this->x_tilde_cur, this->x_tilde_nbr, this->Si, this->Ri, this->NI-1, this->NJ, this->NK, this->ARR_STRIDE_I, this->ARR_STRIDE_J);
}



/*!
 * Constructor of distributed solver of tri-diagonal system
 * \tparam RealType        real-value type for primitive data type
 * \tparam RealTypeComm    real-value type for communication on distributed memory used in solving the reduced system
 * \tparam MemSpaceType    memory space, valid values are cmpk::MemSpace::Host and cmpk::MemSpace::Device
 * \param [in]   num_grids_i    number of grid points in i-direction in the current partition
 * \param [in]   num_grids_j    number of grid points in j-direction in the current partition
 * \param [in]   num_grids_k    number of grid points in k-direction in the current partition
 * \param [in]  arr_stride_i    number of entries strided in i-direction in the current partition, i.e., number of entries from i to i+1
 * \param [in]  arr_stride_j    number of entries strided in j-direction in the current partition, i.e., number of entries from j to j+1
 * \param [in]      mpi_comm    a MPI-sub-communicator that only contains the partitions involved in the system
 * \note k-direction is mapped to a contiguous memory layout
 * \note The description of the data size and layout are based on the current distributed partition
 */
template<typename RealType, typename RealTypeComm, typename MemSpaceType>
DistTriSolDimJ<RealType, RealTypeComm, MemSpaceType>::DistTriSolDimJ(
        const unsigned int num_grids_i,
        const unsigned int num_grids_j,
        const unsigned int num_grids_k,
        const unsigned int arr_stride_i,
        const unsigned int arr_stride_j,
        MPI_Comm mpi_comm) :
    NI           ( num_grids_i),
    NJ           ( num_grids_j),
    NK           ( num_grids_k),
    ARR_STRIDE_I (arr_stride_i),
    ARR_STRIDE_J (arr_stride_j)
{
    if ((this->NJ < 4) || (this->NK < 1) || (this->NI < 1)) throw std::invalid_argument("[Compack3D] Partitioned data size is invalid.");
    this->comm = mpi_comm;
    int Np;
    MPI_Comm_size(mpi_comm, &Np);
    this->fact_local_prev = nullptr;
    this->fact_local_curr = nullptr;
    this->fact_local_next = nullptr;
    this->fact_dist_prev  = nullptr;
    this->fact_dist_curr  = nullptr;
    this->fact_dist_next  = nullptr;
    this->Si              = nullptr;
    this->Ri              = nullptr;
    tri::allocFactBuffers<MemSpaceType, RealType>(
            &this->fact_local_prev,
            &this->fact_local_curr,
            &this->fact_local_next,
            &this->fact_dist_prev,
            &this->fact_dist_curr,
            &this->fact_dist_next,
            &this->Si,
            &this->Ri,
            num_grids_j, Np);
}



/*!
 * Destructor
 */
template<typename RealType, typename RealTypeComm, typename MemSpaceType>
DistTriSolDimJ<RealType, RealTypeComm, MemSpaceType>::~DistTriSolDimJ() {
    memFreeArray<MemSpaceType  , RealType>(this->fact_local_prev);
    memFreeArray<MemSpaceType  , RealType>(this->fact_local_curr);
    memFreeArray<MemSpaceType  , RealType>(this->fact_local_next);
    memFreeArray<MemSpace::Host, RealType>(this->fact_dist_prev);
    memFreeArray<MemSpace::Host, RealType>(this->fact_dist_curr);
    memFreeArray<MemSpace::Host, RealType>(this->fact_dist_next);
    memFreeArray<MemSpaceType  , RealType>(this->Si);
    memFreeArray<MemSpaceType  , RealType>(this->Ri);
}


/*!
 * Detach all shared buffers
 * \note This function call does not free allocated memory and may cause memory leak
 */
template<typename RealType, typename RealTypeComm, typename MemSpaceType>
void DistTriSolDimJ<RealType, RealTypeComm, MemSpaceType>::detachSharedBuffers() {
    this->x_tilde_cur       = nullptr;
    this->x_tilde_nbr       = nullptr;
    this->x_tilde_buf       = nullptr;
    this->x_comm_prev       = nullptr;
    this->x_comm_curr       = nullptr;
    this->x_comm_next       = nullptr;
    this->x_comm_prev_host  = nullptr;
    this->x_comm_curr_host  = nullptr;
    this->x_comm_next_host  = nullptr;
    this->x_tilde_send_host = nullptr;
    this->x_tilde_recv_host = nullptr;
}



/*!
 * Reset shared buffers
 * \param [in]   shared_x_tilde_cur         shared buffer allocated on device
 * \param [in]   shared_x_tilde_nbr         shared buffer allocated on device
 * \param [in]   shared_x_tilde_buf         shared buffer allocated on device
 * \param [in]   shared_x_comm_prev         shared buffer allocated on device
 * \param [in]   shared_x_comm_curr         shared buffer allocated on device
 * \param [in]   shared_x_comm_next         shared buffer allocated on device
 * \param [in]   shared_x_comm_prev_host    shared buffer for x_tilde allocated on host for non-device-aware communication
 * \param [in]   shared_x_comm_curr_host    shared buffer for x_tilde allocated on host for non-device-aware communication
 * \param [in]   shared_x_comm_next_host    shared buffer for x_tilde allocated on host for non-device-aware communication
 * \param [in]   shared_x_tilde_send_host   shared buffer for x_tilde allocated on host for non-device-aware communication
 * \param [in]   shared_x_tilde_recv_host   shared buffer for x_tilde allocated on host for non-device-aware communication
 * \note This function call does not free existing memory and may cause memory leak
 * \note All buffers must be disjoint of size 2 * Nj * Nk
 * \note For device-aware communication shared_x_comm_prev_host, shared_x_comm_curr_host, and shared_x_comm_curr_host must be nullptr
 */
template<typename RealType, typename RealTypeComm, typename MemSpaceType>
void DistTriSolDimJ<RealType, RealTypeComm, MemSpaceType>::resetSharedBuffers(
        RealType*     shared_x_tilde_cur,
        RealType*     shared_x_tilde_nbr,
        RealType*     shared_x_tilde_buf,
        RealTypeComm* shared_x_comm_prev,
        RealTypeComm* shared_x_comm_curr,
        RealTypeComm* shared_x_comm_next,
        RealTypeComm* shared_x_comm_prev_host,
        RealTypeComm* shared_x_comm_curr_host,
        RealTypeComm* shared_x_comm_next_host,
        RealType*     shared_x_tilde_send_host,
        RealType*     shared_x_tilde_recv_host
    )
{
    this->x_tilde_cur       = shared_x_tilde_cur;
    this->x_tilde_nbr       = shared_x_tilde_nbr;
    this->x_tilde_buf       = shared_x_tilde_buf;
    this->x_comm_prev       = shared_x_comm_prev;
    this->x_comm_curr       = shared_x_comm_curr;
    this->x_comm_next       = shared_x_comm_next;
    this->x_comm_prev_host  = shared_x_comm_prev_host;
    this->x_comm_curr_host  = shared_x_comm_curr_host;
    this->x_comm_next_host  = shared_x_comm_next_host;
    this->x_tilde_send_host = shared_x_tilde_send_host;
    this->x_tilde_recv_host = shared_x_tilde_recv_host;
}



/*!
 * Reset penta-diagonal system
 * \param [in]    l     lower-diagonal entries
 * \param [in]    d     diagonal entries
 * \param [in]    u     upper-diagonal entries
 * \note All input buffers are allocated on the host memory
 * \note All data in the input buffers will be destroyed
 * \note All data in the input buffers are for the partitioned system in the current rank
 */
template<typename RealType, typename RealTypeComm, typename MemSpaceType>
void DistTriSolDimJ<RealType, RealTypeComm, MemSpaceType>::resetSystem(
        RealType* l, RealType* d, RealType* u
    )
{
    int Np;
    MPI_Comm_size(this->comm, &Np);
    tri::factPartitionedTri<MemSpaceType, RealType>(
        this->fact_local_prev,
        this->fact_local_curr,
        this->fact_local_next,
        this->fact_dist_prev,
        this->fact_dist_curr,
        this->fact_dist_next,
        this->Si,
        this->Ri,
        this->Li_tilde_tail,
        this->Ui_tilde_head,
        l, d, u,
        this->NJ,
        static_cast<unsigned int>(Np),
        MAX_LOCAL_FACT_SIZE,
        this->comm 
    );
}



/*!
 * Solve the system
 * \param [in, out]    x    right-hand side as input and solution as output
 */
template<typename RealType, typename RealTypeComm, typename MemSpaceType>
void DistTriSolDimJ<RealType, RealTypeComm, MemSpaceType>::solve(RealType* x) {
    #ifdef COMPACK3D_DEVICE_COMM_ENABLED
    constexpr bool USE_DEVICE_AWARE_COMM = true;
    #else
    constexpr bool USE_DEVICE_AWARE_COMM = std::is_same<MemSpaceType, MemSpace::Host>::value;
    #endif

    int rank, Np;
    MPI_Comm_rank(this->comm, &rank);
    MPI_Comm_size(this->comm, &Np);
    const int rank_prev = (rank + Np - 1) % Np;
    const int rank_next = (rank      + 1) % Np;

    RealType* x_loc = &x[this->ARR_STRIDE_J];
    const unsigned int N_local = this->NJ - 1;
    tri::localSolTriPCRDimJ<RealType>(
            x_loc,
            this->fact_local_prev,
            this->fact_local_curr,
            this->fact_local_next,
            MAX_LOCAL_FACT_SIZE,
            this->NI, N_local, this->NK,
            this->ARR_STRIDE_I, this->ARR_STRIDE_J
    );
    copySlicesFrom3DArrayDimJ<RealType>(
            this->x_tilde_buf, &x_loc[(N_local-1) * this->ARR_STRIDE_J],
            this->NI, 1, this->NK, this->ARR_STRIDE_I, this->ARR_STRIDE_J);

    if constexpr (USE_DEVICE_AWARE_COMM) {
        deviceFence();
        MPI_Irecv(this->x_tilde_nbr, this->NK*this->NI, MPIDataType<RealType>::value, rank_prev, 1, this->comm, &this->mpi_reqs[0]);
        MPI_Isend(this->x_tilde_buf, this->NK*this->NI, MPIDataType<RealType>::value, rank_next, 1, this->comm, &this->mpi_reqs[1]);
    } else {
        deepCopy<MemSpace::Host, MemSpaceType, RealType>(this->x_tilde_send_host, this->x_tilde_buf, this->NK*this->NI);
        MPI_Irecv (this->x_tilde_recv_host, this->NK*this->NI, MPIDataType<RealType>::value, rank_prev, 1, this->comm, &this->mpi_reqs[0]);
        MPI_Issend(this->x_tilde_send_host, this->NK*this->NI, MPIDataType<RealType>::value, rank_next, 1, this->comm, &this->mpi_reqs[1]);
    }

    copySlicesFrom3DArrayDimJ<RealType>(this->x_tilde_cur, x, this->NI, 1, this->NK, this->ARR_STRIDE_I, this->ARR_STRIDE_J);

    MPI_Waitall(2, this->mpi_reqs, this->mpi_stats);
    if constexpr (!USE_DEVICE_AWARE_COMM) {
        deepCopy<MemSpaceType, MemSpace::Host, RealType>(this->x_tilde_nbr, this->x_tilde_recv_host, this->NK*this->NI);
    }

    copySlicesFrom3DArrayDimJ<RealType>(this->x_tilde_buf, x_loc, this->NI, 1, this->NK, this->ARR_STRIDE_I, this->ARR_STRIDE_J);
    tri::calcReducedSystemRHSLocal(this->x_tilde_cur, this->x_tilde_nbr, this->x_tilde_buf, this->Li_tilde_tail, this->Ui_tilde_head, this->NK*this->NI);
    tri::distSolve<RealType, RealTypeComm, MemSpaceType>(
            this->x_tilde_cur,
            this->x_comm_prev,
            this->x_comm_curr,
            this->x_comm_next,
            this->fact_dist_prev,
            this->fact_dist_curr,
            this->fact_dist_next,
            this->NK*this->NI, Np,
            this->comm, this->mpi_reqs, this->mpi_stats,
            this->x_comm_prev_host,
            this->x_comm_curr_host,
            this->x_comm_next_host
    );

    if constexpr (USE_DEVICE_AWARE_COMM) {
        deviceFence();
        MPI_Irecv(this->x_tilde_nbr, this->NK*this->NI, MPIDataType<RealType>::value, rank_next, 1, this->comm, &this->mpi_reqs[0]);
        MPI_Isend(this->x_tilde_cur, this->NK*this->NI, MPIDataType<RealType>::value, rank_prev, 1, this->comm, &this->mpi_reqs[1]);
    } else {
        deepCopy<MemSpace::Host, MemSpaceType, RealType>(this->x_tilde_send_host, this->x_tilde_cur, this->NK*this->NI);
        MPI_Irecv (this->x_tilde_recv_host, this->NK*this->NI, MPIDataType<RealType>::value, rank_next, 1, this->comm, &this->mpi_reqs[0]);
        MPI_Issend(this->x_tilde_send_host, this->NK*this->NI, MPIDataType<RealType>::value, rank_prev, 1, this->comm, &this->mpi_reqs[1]);
    }

    copySlicesTo3DArrayDimJ<RealType>(x, this->x_tilde_cur, this->NI, 1, this->NK, this->ARR_STRIDE_I, this->ARR_STRIDE_J);

    MPI_Waitall(2, this->mpi_reqs, this->mpi_stats);
    if constexpr (!USE_DEVICE_AWARE_COMM) {
        deepCopy<MemSpaceType, MemSpace::Host, RealType>(this->x_tilde_nbr, this->x_tilde_recv_host, this->NK*this->NI);
    }

    tri::updateLocalSolDimJ<RealType>(x_loc, this->x_tilde_cur, this->x_tilde_nbr, this->Si, this->Ri, this->NI, this->NJ-1, this->NK, this->ARR_STRIDE_I, this->ARR_STRIDE_J);
}



/*!
 * Constructor of distributed solver of tri-diagonal system
 * \tparam RealType        real-value type for primitive data type
 * \tparam RealTypeComm    real-value type for communication on distributed memory used in solving the reduced system
 * \tparam MemSpaceType    memory space, valid values are cmpk::MemSpace::Host and cmpk::MemSpace::Device
 * \param [in]   num_grids_i    number of grid points in i-direction in the current partition
 * \param [in]   num_grids_j    number of grid points in j-direction in the current partition
 * \param [in]   num_grids_k    number of grid points in k-direction in the current partition
 * \param [in]  arr_stride_i    number of entries strided in i-direction in the current partition, i.e., number of entries from i to i+1
 * \param [in]  arr_stride_j    number of entries strided in j-direction in the current partition, i.e., number of entries from j to j+1
 * \param [in]      mpi_comm    a MPI-sub-communicator that only contains the partitions involved in the system
 * \note k-direction is mapped to a contiguous memory layout
 * \note The description of the data size and layout are based on the current distributed partition
 */
template<typename RealType, typename RealTypeComm, typename MemSpaceType>
DistTriSolDimK<RealType, RealTypeComm, MemSpaceType>::DistTriSolDimK(
        const unsigned int num_grids_i,
        const unsigned int num_grids_j,
        const unsigned int num_grids_k,
        const unsigned int arr_stride_i,
        const unsigned int arr_stride_j,
        MPI_Comm mpi_comm) :
    NI           ( num_grids_i),
    NJ           ( num_grids_j),
    NK           ( num_grids_k),
    ARR_STRIDE_I (arr_stride_i),
    ARR_STRIDE_J (arr_stride_j)
{
    if ((this->NK < 4) || (this->NI < 1) || (this->NJ < 1)) throw std::invalid_argument("[Compack3D] Partitioned data size is invalid.");
    this->comm = mpi_comm;
    int Np;
    MPI_Comm_size(mpi_comm, &Np);
    this->fact_local_prev = nullptr;
    this->fact_local_curr = nullptr;
    this->fact_local_next = nullptr;
    this->fact_dist_prev  = nullptr;
    this->fact_dist_curr  = nullptr;
    this->fact_dist_next  = nullptr;
    this->Si              = nullptr;
    this->Ri              = nullptr;
    tri::allocFactBuffers<MemSpaceType, RealType>(
            &this->fact_local_prev,
            &this->fact_local_curr,
            &this->fact_local_next,
            &this->fact_dist_prev,
            &this->fact_dist_curr,
            &this->fact_dist_next,
            &this->Si,
            &this->Ri,
            num_grids_k, Np);
}



/*!
 * Destructor
 */
template<typename RealType, typename RealTypeComm, typename MemSpaceType>
DistTriSolDimK<RealType, RealTypeComm, MemSpaceType>::~DistTriSolDimK() {
    memFreeArray<MemSpaceType  , RealType>(this->fact_local_prev);
    memFreeArray<MemSpaceType  , RealType>(this->fact_local_curr);
    memFreeArray<MemSpaceType  , RealType>(this->fact_local_next);
    memFreeArray<MemSpace::Host, RealType>(this->fact_dist_prev);
    memFreeArray<MemSpace::Host, RealType>(this->fact_dist_curr);
    memFreeArray<MemSpace::Host, RealType>(this->fact_dist_next);
    memFreeArray<MemSpaceType  , RealType>(this->Si);
    memFreeArray<MemSpaceType  , RealType>(this->Ri);
}


/*!
 * Detach all shared buffers
 * \note This function call does not free allocated memory and may cause memory leak
 */
template<typename RealType, typename RealTypeComm, typename MemSpaceType>
void DistTriSolDimK<RealType, RealTypeComm, MemSpaceType>::detachSharedBuffers() {
    this->x_tilde_cur       = nullptr;
    this->x_tilde_nbr       = nullptr;
    this->x_tilde_buf       = nullptr;
    this->x_comm_prev       = nullptr;
    this->x_comm_curr       = nullptr;
    this->x_comm_next       = nullptr;
    this->x_comm_prev_host  = nullptr;
    this->x_comm_curr_host  = nullptr;
    this->x_comm_next_host  = nullptr;
    this->x_tilde_send_host = nullptr;
    this->x_tilde_recv_host = nullptr;
}



/*!
 * Reset shared buffers
 * \param [in]   shared_x_tilde_cur         shared buffer allocated on device
 * \param [in]   shared_x_tilde_nbr         shared buffer allocated on device
 * \param [in]   shared_x_tilde_buf         shared buffer allocated on device
 * \param [in]   shared_x_comm_prev         shared buffer allocated on device
 * \param [in]   shared_x_comm_curr         shared buffer allocated on device
 * \param [in]   shared_x_comm_next         shared buffer allocated on device
 * \param [in]   shared_x_comm_prev_host    shared buffer for x_tilde allocated on host for non-device-aware communication
 * \param [in]   shared_x_comm_curr_host    shared buffer for x_tilde allocated on host for non-device-aware communication
 * \param [in]   shared_x_comm_next_host    shared buffer for x_tilde allocated on host for non-device-aware communication
 * \param [in]   shared_x_tilde_send_host   shared buffer for x_tilde allocated on host for non-device-aware communication
 * \param [in]   shared_x_tilde_recv_host   shared buffer for x_tilde allocated on host for non-device-aware communication
 * \note This function call does not free existing memory and may cause memory leak
 * \note All buffers must be disjoint of size 2 * Nj * Nk
 * \note For device-aware communication shared_x_comm_prev_host, shared_x_comm_curr_host, and shared_x_comm_curr_host must be nullptr
 */
template<typename RealType, typename RealTypeComm, typename MemSpaceType>
void DistTriSolDimK<RealType, RealTypeComm, MemSpaceType>::resetSharedBuffers(
        RealType*     shared_x_tilde_cur,
        RealType*     shared_x_tilde_nbr,
        RealType*     shared_x_tilde_buf,
        RealTypeComm* shared_x_comm_prev,
        RealTypeComm* shared_x_comm_curr,
        RealTypeComm* shared_x_comm_next,
        RealTypeComm* shared_x_comm_prev_host,
        RealTypeComm* shared_x_comm_curr_host,
        RealTypeComm* shared_x_comm_next_host,
        RealType*     shared_x_tilde_send_host,
        RealType*     shared_x_tilde_recv_host
    )
{
    this->x_tilde_cur       = shared_x_tilde_cur;
    this->x_tilde_nbr       = shared_x_tilde_nbr;
    this->x_tilde_buf       = shared_x_tilde_buf;
    this->x_comm_prev       = shared_x_comm_prev;
    this->x_comm_curr       = shared_x_comm_curr;
    this->x_comm_next       = shared_x_comm_next;
    this->x_comm_prev_host  = shared_x_comm_prev_host;
    this->x_comm_curr_host  = shared_x_comm_curr_host;
    this->x_comm_next_host  = shared_x_comm_next_host;
    this->x_tilde_send_host = shared_x_tilde_send_host;
    this->x_tilde_recv_host = shared_x_tilde_recv_host;
}



/*!
 * Reset penta-diagonal system
 * \param [in]    l     lower-diagonal entries
 * \param [in]    d     diagonal entries
 * \param [in]    u     upper-diagonal entries
 * \note All input buffers are allocated on the host memory
 * \note All data in the input buffers will be destroyed
 * \note All data in the input buffers are for the partitioned system in the current rank
 */
template<typename RealType, typename RealTypeComm, typename MemSpaceType>
void DistTriSolDimK<RealType, RealTypeComm, MemSpaceType>::resetSystem(
        RealType* l, RealType* d, RealType* u
    )
{
    int Np;
    MPI_Comm_size(this->comm, &Np);
    tri::factPartitionedTri<MemSpaceType, RealType>(
        this->fact_local_prev,
        this->fact_local_curr,
        this->fact_local_next,
        this->fact_dist_prev,
        this->fact_dist_curr,
        this->fact_dist_next,
        this->Si,
        this->Ri,
        this->Li_tilde_tail,
        this->Ui_tilde_head,
        l, d, u,
        this->NK,
        static_cast<unsigned int>(Np),
        MAX_LOCAL_FACT_SIZE,
        this->comm 
    );
}



/*!
 * Solve the system
 * \param [in, out]    x    right-hand side as input and solution as output
 */
template<typename RealType, typename RealTypeComm, typename MemSpaceType>
void DistTriSolDimK<RealType, RealTypeComm, MemSpaceType>::solve(RealType* x) {
    #ifdef COMPACK3D_DEVICE_COMM_ENABLED
    constexpr bool USE_DEVICE_AWARE_COMM = true;
    #else
    constexpr bool USE_DEVICE_AWARE_COMM = std::is_same<MemSpaceType, MemSpace::Host>::value;
    #endif

    int rank, Np;
    MPI_Comm_rank(this->comm, &rank);
    MPI_Comm_size(this->comm, &Np);
    const int rank_prev = (rank + Np - 1) % Np;
    const int rank_next = (rank      + 1) % Np;

    RealType* x_loc = &x[1];
    const unsigned int N_local = this->NK - 1;
    tri::localSolTriPCRDimK<RealType>(
            x_loc,
            this->fact_local_prev,
            this->fact_local_curr,
            this->fact_local_next,
            MAX_LOCAL_FACT_SIZE,
            this->NI, this->NJ, N_local,
            this->ARR_STRIDE_I, this->ARR_STRIDE_J
    );
    copySlicesFrom3DArrayDimK<RealType>(
            this->x_tilde_buf, &x_loc[N_local-1],
            this->NI, this->NJ, 1, this->ARR_STRIDE_I, this->ARR_STRIDE_J);

    if constexpr (USE_DEVICE_AWARE_COMM) {
        deviceFence();
        MPI_Irecv(this->x_tilde_nbr, this->NI*this->NJ, MPIDataType<RealType>::value, rank_prev, 1, this->comm, &this->mpi_reqs[0]);
        MPI_Isend(this->x_tilde_buf, this->NI*this->NJ, MPIDataType<RealType>::value, rank_next, 1, this->comm, &this->mpi_reqs[1]);
    } else {
        deepCopy<MemSpace::Host, MemSpaceType, RealType>(this->x_tilde_send_host, this->x_tilde_buf, this->NI*this->NJ);
        MPI_Irecv (this->x_tilde_recv_host, this->NI*this->NJ, MPIDataType<RealType>::value, rank_prev, 1, this->comm, &this->mpi_reqs[0]);
        MPI_Issend(this->x_tilde_send_host, this->NI*this->NJ, MPIDataType<RealType>::value, rank_next, 1, this->comm, &this->mpi_reqs[1]);
    }

    copySlicesFrom3DArrayDimK<RealType>(this->x_tilde_cur, x, this->NI, this->NJ, 1, this->ARR_STRIDE_I, this->ARR_STRIDE_J);

    MPI_Waitall(2, this->mpi_reqs, this->mpi_stats);
    if constexpr (!USE_DEVICE_AWARE_COMM) {
        deepCopy<MemSpaceType, MemSpace::Host, RealType>(this->x_tilde_nbr, this->x_tilde_recv_host, this->NI*this->NJ);
    }

    copySlicesFrom3DArrayDimK<RealType>(this->x_tilde_buf, x_loc, this->NI, this->NJ, 1, this->ARR_STRIDE_I, this->ARR_STRIDE_J);
    tri::calcReducedSystemRHSLocal(this->x_tilde_cur, this->x_tilde_nbr, this->x_tilde_buf, this->Li_tilde_tail, this->Ui_tilde_head, this->NI*this->NJ);
    tri::distSolve<RealType, RealTypeComm, MemSpaceType>(
            this->x_tilde_cur,
            this->x_comm_prev,
            this->x_comm_curr,
            this->x_comm_next,
            this->fact_dist_prev,
            this->fact_dist_curr,
            this->fact_dist_next,
            this->NI*this->NJ, Np,
            this->comm, this->mpi_reqs, this->mpi_stats,
            this->x_comm_prev_host,
            this->x_comm_curr_host,
            this->x_comm_next_host
    );

    if constexpr (USE_DEVICE_AWARE_COMM) {
        deviceFence();
        MPI_Irecv(this->x_tilde_nbr, this->NI*this->NJ, MPIDataType<RealType>::value, rank_next, 1, this->comm, &this->mpi_reqs[0]);
        MPI_Isend(this->x_tilde_cur, this->NI*this->NJ, MPIDataType<RealType>::value, rank_prev, 1, this->comm, &this->mpi_reqs[1]);
    } else {
        deepCopy<MemSpace::Host, MemSpaceType, RealType>(this->x_tilde_send_host, this->x_tilde_cur, this->NI*this->NJ);
        MPI_Irecv (this->x_tilde_recv_host, this->NI*this->NJ, MPIDataType<RealType>::value, rank_next, 1, this->comm, &this->mpi_reqs[0]);
        MPI_Issend(this->x_tilde_send_host, this->NI*this->NJ, MPIDataType<RealType>::value, rank_prev, 1, this->comm, &this->mpi_reqs[1]);
    }

    copySlicesTo3DArrayDimK<RealType>(x, this->x_tilde_cur, this->NI, this->NJ, 1, this->ARR_STRIDE_I, this->ARR_STRIDE_J);

    MPI_Waitall(2, this->mpi_reqs, this->mpi_stats);
    if constexpr (!USE_DEVICE_AWARE_COMM) {
        deepCopy<MemSpaceType, MemSpace::Host, RealType>(this->x_tilde_nbr, this->x_tilde_recv_host, this->NI*this->NJ);
    }

    tri::updateLocalSolDimK<RealType>(x_loc, this->x_tilde_cur, this->x_tilde_nbr, this->Si, this->Ri, this->NI, this->NJ, this->NK-1, this->ARR_STRIDE_I, this->ARR_STRIDE_J);
}



////////////////////////////////
//// EXPLICIT INSTANTIATION ////
////////////////////////////////

template class DistPentaSolDimI<double, double, MemSpace::Device>;
template class DistPentaSolDimI<double,  float, MemSpace::Device>;

template class DistPentaSolDimJ<double, double, MemSpace::Device>;
template class DistPentaSolDimJ<double,  float, MemSpace::Device>;

template class DistPentaSolDimK<double, double, MemSpace::Device>;
template class DistPentaSolDimK<double,  float, MemSpace::Device>;

template class DistTriSolDimI<double, double, MemSpace::Device>;
template class DistTriSolDimI<double,  float, MemSpace::Device>;

template class DistTriSolDimJ<double, double, MemSpace::Device>;
template class DistTriSolDimJ<double,  float, MemSpace::Device>;

template class DistTriSolDimK<double, double, MemSpace::Device>;
template class DistTriSolDimK<double,  float, MemSpace::Device>;

} // namespace cmpk

#undef MAX_LOCAL_FACT_SIZE
