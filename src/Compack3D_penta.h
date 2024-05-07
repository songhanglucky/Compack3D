
#ifndef _COMPACK3D_PENTA_H
#define _COMPACK3D_PENTA_H

#include <stdexcept>
#include <type_traits>
#include <mpi.h>
#include "Compack3D_utils.h"
#ifdef COMPACK3D_DEVICE_ENABLED_CUDA
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "Compack3D_penta_kernels.cuh"
#endif

namespace cmpk {
namespace penta {


/*!
 * Data structure of a 3x3 system for factorization
 * --                         -- --      --   --   --
 * |  d_prev   l_curr        0 | | a_prev |   |  0  |
 * |  u_prev   d_curr   l_next | | a_curr | = |  1  |
 * |       0   u_curr   d_next | | a_next |   |  0  |
 * --                         -- --      --   --   --
 */
template<typename RealType>
struct FactSysTri {
    RealType prev[2]; // d, u
    RealType curr[3]; // l, d, u
    RealType next[2]; // l, d
};



/*!
 * Data structure of a 5x5 system for factorization
 * --                                                         -- --         --     --   --
 * |  l1_prev2    l2_prev1          0          0            0  | |  a_prev2  |     |  0  |
 * |  u1_prev2     d_prev1    l1_curr    l2_next1           0  | |  a_prev1  |     |  0  |
 * |  u2_prev2    u1_prev1     d_curr    l1_next1    l2_next2  | |  a_curr   |  =  |  1  |
 * |         0    u2_prev1    u1_curr     d_next1    l1_next2  | |  a_next1  |     |  0  |
 * |         0           0          0    u2_next1    u1_next2  | |  a_next2  |     |  0  |
 * --                                                         -- --         --     --   --
 */
template<typename RealType>
struct FactSysPenta {
    RealType prev2[3]; // l1, u1, u2
    RealType prev1[4]; // l2,  d, u1, u2
    RealType  curr[3]; // l1,  d, u1
    RealType next1[4]; // l2, l1,  d, u2
    RealType next2[3]; // l2, l1, u1
};

template<typename MemSpaceType, typename RealType>
void allocFactBuffers(RealType**, RealType**, RealType**, RealType**, RealType**, RealType**, RealType**, RealType**, RealType**, RealType**, RealType**, RealType**, const unsigned int, const unsigned int);

template<typename MemSpaceType, typename RealType>
void freeFactBuffers(RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*);

template<typename RealType>
void factPartitionedPentaHost(RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, const int, const int, const int, MPI_Comm);

template<typename MemSpaceType, typename RealType>
void factPartitionedPenta(RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, const unsigned int, const unsigned int, const unsigned int, MPI_Comm);

template<typename RealType>
void solFact(RealType&, RealType&, RealType&, FactSysTri<RealType>&, const int);

template<typename RealType>
void solFact(RealType&, RealType&, RealType&, RealType&, RealType&, FactSysPenta<RealType>&, const int);

template<typename RealType>
void localFactPenta(RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, const int, const int);

template<typename RealType>
void vanillaLocalSolPentaPCR(RealType*, const RealType*, const RealType*, const RealType*, const RealType*, const RealType*, const int, const int);

template<typename RealType>
void vanillaLocalSolPentaPCRBatch(RealType*, const RealType*, const RealType*, const RealType*, const RealType*, const RealType*, const int, const int, const int);

template<typename RealType>
void distFactPenta(RealType*, RealType*, RealType*, RealType*, const int, MPI_Comm);

template<typename RealType>
void mat2x2AXPY(RealType*, const RealType*, const RealType*, const RealType, const RealType);
    
template<typename RealType>
void mat2x2Inv(RealType*, const RealType*);

template<typename RealType>
void mat2x2AMultInvB(RealType*, const RealType*, const RealType*);


template<typename RealType, typename RealTypeComm>
void vanillaDistSolve (RealType*, RealTypeComm*, RealTypeComm*, RealTypeComm*, RealType*, RealType*, RealType*, const int, MPI_Comm);


template<typename RealType, typename RealTypeComm, typename MemSpaceType>
void distSolve (RealType*, RealTypeComm*, RealTypeComm*, RealTypeComm*, RealType*, RealType*, RealType*, const unsigned int, const unsigned int,  MPI_Comm, RealTypeComm*, RealTypeComm*, RealTypeComm*);


template<typename RealType, typename RealTypeComm, typename MemSpaceType>
void distSolve (RealType*, RealTypeComm*, RealTypeComm*, RealTypeComm*, RealType*, RealType*, RealType*, const unsigned int, const unsigned int,  MPI_Comm, MPI_Request*, MPI_Status*, RealTypeComm*, RealTypeComm*, RealTypeComm*);



} // namespace penta
} // namespace cmpk

#endif // _COMPACK3D_FACT_H
