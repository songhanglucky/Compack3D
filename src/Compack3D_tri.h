
#ifndef _COMPACK3D_TRI_H
#define _COMPACK3D_TRI_H

#include <stdexcept>
#include <type_traits>
#include <mpi.h>
#include "Compack3D_utils.h"
#ifdef COMPACK3D_DEVICE_ENABLED_CUDA
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "Compack3D_tri_kernels.cuh"
#include "Compack3D_penta_kernels.cuh"
#endif

namespace cmpk {
namespace tri {

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


template<typename RealType>
void solFact(RealType&, RealType&, RealType&, FactSysTri<RealType>&, const int);

template<typename MemSpaceType, typename RealType>
void allocFactBuffers(RealType**, RealType**, RealType**, RealType**, RealType**, RealType**, RealType**, RealType**, const unsigned int, const unsigned int);

template<typename MemSpaceType, typename RealType>
void freeFactBuffers(RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*);

template<typename RealType>
void factPartitionedTriHost(RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, const int, const int, const int, MPI_Comm);

template<typename MemSpaceType, typename RealType>
void factPartitionedTri(RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, RealType&, RealType&, RealType*, RealType*, RealType*, const unsigned int, const unsigned int, const unsigned int, MPI_Comm);


template<typename RealType, typename RealTypeComm, typename MemSpaceType>
void distSolve (RealType*, RealTypeComm*, RealTypeComm*, RealTypeComm*, RealType*, RealType*, RealType*, const unsigned int, const unsigned int,  MPI_Comm, RealTypeComm*, RealTypeComm*, RealTypeComm*);


template<typename RealType, typename RealTypeComm, typename MemSpaceType>
void distSolve (RealType*, RealTypeComm*, RealTypeComm*, RealTypeComm*, RealType*, RealType*, RealType*, const unsigned int, const unsigned int,  MPI_Comm, MPI_Request*, MPI_Status*, RealTypeComm*, RealTypeComm*, RealTypeComm*);


template<typename RealType>
void vanillaLocalSolTriPCR (RealType*, const RealType*, const RealType*, const RealType*, const int, const int);

template<typename RealType>
void vanillaLocalSolTriPCRBatch (RealType*, const RealType*, const RealType*, const RealType*, const int, const int, const int);


template<typename RealType>
void localFactTri (RealType*, RealType*, RealType*, RealType*, RealType*, RealType*, const int, const int);


template<typename RealType, typename RealTypeComm>
void vanillaDistSolve(RealType&, RealTypeComm&, RealTypeComm&, RealTypeComm&, RealType*, RealType*, RealType*, const int, MPI_Comm);


} // namespace tri
} // namespace cmpk


#endif
