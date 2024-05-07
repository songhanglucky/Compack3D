#ifndef _COMPACK3D_API_H
#define _COMPACK3D_API_H

#include "Compack3D_utils.h"
#include "Compack3D_penta.h"

#ifdef COMPACK3D_DEVICE_ENABLED_CUDA
#include <cuda_runtime_api.h>
#include <cuda.h>
#endif

namespace cmpk {

template<typename RealType, typename RealTypeComm, typename MemSpaceType>
class DistPentaSolDimI {
  public:
    DistPentaSolDimI() = delete;
    DistPentaSolDimI(const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int, MPI_Comm);
    ~DistPentaSolDimI();
    void detachSharedBuffers();
    void resetSharedBuffers(RealType*, RealType*, RealType*, RealTypeComm*, RealTypeComm*, RealTypeComm*, RealTypeComm*, RealTypeComm*, RealTypeComm*, RealType*, RealType*);
    void resetSystem(RealType*, RealType*, RealType*, RealType*, RealType*);
    void solve(RealType*);

  protected:
    const unsigned int NI;
    const unsigned int NJ;
    const unsigned int NK;
    const unsigned int ARR_STRIDE_I; 
    const unsigned int ARR_STRIDE_J; 
    MPI_Comm comm;
    MPI_Request mpi_reqs [4];
    MPI_Status  mpi_stats[4];

    RealType* fact_local_prev_2;
    RealType* fact_local_prev_1;
    RealType* fact_local_curr;
    RealType* fact_local_next_1;
    RealType* fact_local_next_2;
    RealType* fact_dist_prev;
    RealType* fact_dist_curr;
    RealType* fact_dist_next;
    RealType* Si;
    RealType* Ri;
    RealType* Li_tilde_tail;
    RealType* Ui_tilde_head;

    RealType*     x_tilde_cur;
    RealType*     x_tilde_nbr;
    RealType*     x_tilde_buf;
    RealTypeComm* x_comm_prev;
    RealTypeComm* x_comm_curr;
    RealTypeComm* x_comm_next;
    RealTypeComm* x_comm_prev_host;
    RealTypeComm* x_comm_curr_host;
    RealTypeComm* x_comm_next_host;
    RealType*     x_tilde_send_host;
    RealType*     x_tilde_recv_host;
};



template<typename RealType, typename RealTypeComm, typename MemSpaceType>
class DistPentaSolDimJ {
  public:
    DistPentaSolDimJ() = delete;
    DistPentaSolDimJ(const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int, MPI_Comm);
    ~DistPentaSolDimJ();
    void detachSharedBuffers();
    void resetSharedBuffers(RealType*, RealType*, RealType*, RealTypeComm*, RealTypeComm*, RealTypeComm*, RealTypeComm*, RealTypeComm*, RealTypeComm*, RealType*, RealType*);
    void resetSystem(RealType*, RealType*, RealType*, RealType*, RealType*);
    void solve(RealType*);

  protected:
    const unsigned int NI;
    const unsigned int NJ;
    const unsigned int NK;
    const unsigned int ARR_STRIDE_I; 
    const unsigned int ARR_STRIDE_J; 
    MPI_Comm comm;
    MPI_Request mpi_reqs [4];
    MPI_Status  mpi_stats[4];

    RealType* fact_local_prev_2;
    RealType* fact_local_prev_1;
    RealType* fact_local_curr;
    RealType* fact_local_next_1;
    RealType* fact_local_next_2;
    RealType* fact_dist_prev;
    RealType* fact_dist_curr;
    RealType* fact_dist_next;
    RealType* Si;
    RealType* Ri;
    RealType* Li_tilde_tail;
    RealType* Ui_tilde_head;

    RealType*     x_tilde_cur;
    RealType*     x_tilde_nbr;
    RealType*     x_tilde_buf;
    RealTypeComm* x_comm_prev;
    RealTypeComm* x_comm_curr;
    RealTypeComm* x_comm_next;
    RealTypeComm* x_comm_prev_host;
    RealTypeComm* x_comm_curr_host;
    RealTypeComm* x_comm_next_host;
    RealType*     x_tilde_send_host;
    RealType*     x_tilde_recv_host;
};



template<typename RealType, typename RealTypeComm, typename MemSpaceType>
class DistPentaSolDimK {
  public:
    DistPentaSolDimK() = delete;
    DistPentaSolDimK(const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int, MPI_Comm);
    ~DistPentaSolDimK();
    void detachSharedBuffers();
    void resetSharedBuffers(RealType*, RealType*, RealType*, RealTypeComm*, RealTypeComm*, RealTypeComm*, RealTypeComm*, RealTypeComm*, RealTypeComm*, RealType*, RealType*);
    void resetSystem(RealType*, RealType*, RealType*, RealType*, RealType*);
    void solve(RealType*);

  protected:
    const unsigned int NI;
    const unsigned int NJ;
    const unsigned int NK;
    const unsigned int ARR_STRIDE_I; 
    const unsigned int ARR_STRIDE_J; 
    MPI_Comm comm;
    MPI_Request mpi_reqs [4];
    MPI_Status  mpi_stats[4];

    RealType* fact_local_prev_2;
    RealType* fact_local_prev_1;
    RealType* fact_local_curr;
    RealType* fact_local_next_1;
    RealType* fact_local_next_2;
    RealType* fact_dist_prev;
    RealType* fact_dist_curr;
    RealType* fact_dist_next;
    RealType* Si;
    RealType* Ri;
    RealType* Li_tilde_tail;
    RealType* Ui_tilde_head;

    RealType*     x_tilde_cur;
    RealType*     x_tilde_nbr;
    RealType*     x_tilde_buf;
    RealTypeComm* x_comm_prev;
    RealTypeComm* x_comm_curr;
    RealTypeComm* x_comm_next;
    RealTypeComm* x_comm_prev_host;
    RealTypeComm* x_comm_curr_host;
    RealTypeComm* x_comm_next_host;
    RealType*     x_tilde_send_host;
    RealType*     x_tilde_recv_host;
};


} // namespace cmpk

#endif // _COMPACK3D_API_H
