
#ifndef _COMPACK3D_UTILS_H
#define _COMPACK3D_UTILS_H


#include <mpi.h>
#ifdef COMPACK3D_DEVICE_ENABLED_CUDA
#include "Compack3D_utils_kernels.cuh"
#endif


namespace cmpk {

template<typename IntType> 
extern IntType log2Ceil(IntType);


template<typename dtype>
struct MPIDataType {
    static const MPI_Datatype value;
};


template<typename IntType>
extern IntType locFactIdx(IntType, IntType, IntType, IntType);



struct MemSpace {
    struct Host;
    struct Device;
};



template<typename DstMemSpaceType, typename SrcMemSpaceType, typename DataType>
void deepCopy(DataType*, DataType*, const unsigned int);


template<typename MemSpaceType, typename DataType>
void memAllocArray(DataType**, const unsigned int);


template<typename MemSpaceType, typename DataType>
void memFreeArray(DataType*);


void deviceFence(void);

template<typename IntType>
IntType numDistSolSteps(IntType);

} // namespace cmpk

#endif // _COMPACK3D_UTILS_H

