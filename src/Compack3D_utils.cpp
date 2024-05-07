
#include <cstdlib>
#include <stdexcept>
#include "mpi.h"
#include "Compack3D_utils.h"
#ifdef COMPACK3D_DEVICE_ENABLED_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "Compack3D_utils_kernels.cuh"
#endif

namespace cmpk {

template<> const MPI_Datatype MPIDataType<double>::value = MPI_DOUBLE;
template<> const MPI_Datatype MPIDataType< float>::value = MPI_FLOAT;

#ifndef __CUDA_ARCH__
template<typename IntType>
IntType log2Ceil(IntType n) {
    IntType result = 0;
    bool round_up  = 0;
    while (n >> 1) {
        result ++;
        round_up = round_up || (n & 0b1);
        n >>= 1;
    }
    return result + round_up;
}
#endif


/*!
 * Index of factorization
 * \param [in]  i                    index of row
 * \param [in]  N                    size of the system
 * \param [in]  stride               current stride
 * \param [in]  max_sub_sys_size     maxium allowed sub-system size that can be solved directly
 * \returns     factorization index at the level
 */
#ifndef __CUDA_ARCH__
template<typename IntType>
IntType locFactIdx(IntType i, IntType N, IntType stride, IntType max_sub_sys_size) {
#ifdef COMPACK3D_DEBUG_MODE_ENABLED
    IntType stride_test = 1;
    if (stride < 1) throw std::invalid_argument("The input argument \"stride\" must be power of 2.");
    while (stride_test < stride) {
        stride_test <<= 1;
        if (stride_test > stride) std::invalid_argument("The input argument \"stride\" must be power of 2.");
    }
#endif
    IntType s      = 1;
    IntType offset = 0;
    while (N > max_sub_sys_size && s < stride) {
        offset += (i & 0b1) * ((N + 1) >> 1);
        N = (N >> 1) + (N & 0b1) * ((i+1) & 0b1);
        i >>= 1;
        s <<= 1;
    }
    return offset + i;
}
#endif



/*!
 * Device synchronization
 */
void deviceFence(void) {
    #ifdef COMPACK3D_DEVICE_ENABLED_CUDA
    cudaDeviceSynchronize();
    #endif
}



/*!
 * Copy data between two distinct memory buffers with aligned memory layout
 * \tparam    DstMemSpaceType    memory space for the output buffer
 * \tparam    SrcMemSpaceType    memory space for the input buffer
 * \tparm     DataType           data type
 * \param [out] dst    output buffer
 * \param [in]  src    input buffer
 * \param [in]  N      count (number of entries)
 * \note Valid memory space types are cmpk::MemSpace::Host and cmpk::MemSpace::Device
 */
template<typename DstMemSpaceType, typename SrcMemSpaceType, typename DataType>
void deepCopy(DataType* dst, DataType* src, const unsigned int N) {
    if (dst == src) return;
    if        constexpr (std::is_same<DstMemSpaceType, MemSpace::Host>::value && std::is_same<SrcMemSpaceType, MemSpace::Host>::value) {
        for (unsigned int i = 0; i < N; i++) dst[i] = src[i];
    } else if constexpr (std::is_same<DstMemSpaceType, MemSpace::Host>::value && std::is_same<SrcMemSpaceType, MemSpace::Device>::value) {
        #ifdef COMPACK3D_DEVICE_ENABLED_CUDA
        cudaMemcpy(dst, src, N * sizeof(DataType), cudaMemcpyDeviceToHost);
        #endif
    } else if constexpr (std::is_same<DstMemSpaceType, MemSpace::Device>::value && std::is_same<SrcMemSpaceType, MemSpace::Host>::value) {
        #ifdef COMPACK3D_DEVICE_ENABLED_CUDA
        cudaMemcpy(dst, src, N * sizeof(DataType), cudaMemcpyHostToDevice);
        #endif
    } else {
        #ifdef COMPACK3D_DEVICE_ENABLED_CUDA
        cudaMemcpy(dst, src, N * sizeof(DataType), cudaMemcpyDeviceToDevice);
        #endif
    }

    #ifdef COMPACK3D_DEVICE_ENABLED_CUDA
    cudaError_t e=cudaPeekAtLastError();
    if(e!=cudaSuccess) {
        printf("Deep copy failure (CUDA Error Code: %u): \"%s\"\n", e, cudaGetErrorString(e));
        exit(0);
    }
    #endif
}



/*!
 * Memory allocation on the specified memory space
 * \tparam MemSpaceType  memory space type cmpk::MemSpace::Host or cmpk::MemSpace::Device
 * \tparam DataType      data type
 * \param [in, out]  buf    pointer of a buffer (where the buffer is represented by a pointer)
 * \param [in]       N      count (number of entries)
 * \note The array buffer (*buf) must be initially a nullptr
 */
template<typename MemSpaceType, typename DataType>
void memAllocArray(DataType** buf, const unsigned int N) {
    if (*buf) throw std::invalid_argument("Compack3D does not want to take the responsibility for any potential memory leak, so we require the buffer pointer to be nullptr :)");
    if constexpr (std::is_same<MemSpaceType, MemSpace::Device>::value) {
        #ifdef COMPACK3D_DEVICE_ENABLED_CUDA
        cudaMalloc((void**) buf, N * sizeof(DataType));
        cudaError_t e=cudaPeekAtLastError();
        if(e!=cudaSuccess) {
            printf("Memory allocation failure (Error Code: %u): \"%s\"\n", e, cudaGetErrorString(e));
            exit(0);
        }
        #endif
    } else {
        (*buf) = new DataType [N];
    }
}



/*!
 * Free array memory.
 * \tparam MemSpaceType  memory space type cmpk::MemSpace::Host or cmpk::MemSpace::Device
 * \tparam DataType      data type
 * \param [in]  buf    buffer to be deallocated (where the buffer is represented by a pointer)
 */
template<typename MemSpaceType, typename DataType>
void memFreeArray(DataType* buf) {
    if constexpr (std::is_same<MemSpaceType, MemSpace::Device>::value) {
        #ifdef COMPACK3D_DEVICE_ENABLED_CUDA
        cudaFree(buf);
        cudaError_t e=cudaPeekAtLastError();
        if(e!=cudaSuccess) {
            printf("Memory deallocation failure (Error Code: %u): \"%s\"\n", e, cudaGetErrorString(e));
            exit(0);
        }
        #endif
    } else {
        delete [] buf;
    }

    buf = nullptr;
}




template struct MPIDataType<double>;
template struct MPIDataType< float>;

#ifndef __CUDA_ARCH__
template          int log2Ceil(         int);
template unsigned int log2Ceil(unsigned int);
template  std::size_t log2Ceil( std::size_t);
#endif

#ifndef __CUDA_ARCH__
template std::size_t  locFactIdx( std::size_t,  std::size_t,  std::size_t,  std::size_t);
template int          locFactIdx(         int,          int,          int,          int);
template unsigned int locFactIdx(unsigned int, unsigned int, unsigned int, unsigned int);
#endif



/*!
 * Get total number of steps for distributed PCR including detaching and reattaching steps
 * \param [in] Np    Total number of distributed memory partitions
 * \returns number of stages including PCR, detaching and reattaching steps
 */
template<typename IntType>
IntType numDistSolSteps(IntType Np) {
    IntType num_stages = 0;
    while (Np > 0) {
        num_stages ++;                           // regular PCR
        num_stages += 2 * (Np > 1) * (Np & 0b1); // detach & reattach
        Np >>= 1;
    }
    return num_stages;
}


template void deepCopy<MemSpace::Host  , MemSpace::Host  ,       double>(      double*,       double*, const unsigned int);
template void deepCopy<MemSpace::Device, MemSpace::Host  ,       double>(      double*,       double*, const unsigned int);
template void deepCopy<MemSpace::Host  , MemSpace::Device,       double>(      double*,       double*, const unsigned int);
template void deepCopy<MemSpace::Device, MemSpace::Device,       double>(      double*,       double*, const unsigned int);
template void deepCopy<MemSpace::Host  , MemSpace::Host  ,        float>(       float*,        float*, const unsigned int);
template void deepCopy<MemSpace::Device, MemSpace::Host  ,        float>(       float*,        float*, const unsigned int);
template void deepCopy<MemSpace::Host  , MemSpace::Device,        float>(       float*,        float*, const unsigned int);
template void deepCopy<MemSpace::Device, MemSpace::Device,        float>(       float*,        float*, const unsigned int);
template void deepCopy<MemSpace::Host  , MemSpace::Host  ,          int>(         int*,          int*, const unsigned int);
template void deepCopy<MemSpace::Device, MemSpace::Host  ,          int>(         int*,          int*, const unsigned int);
template void deepCopy<MemSpace::Host  , MemSpace::Device,          int>(         int*,          int*, const unsigned int);
template void deepCopy<MemSpace::Device, MemSpace::Device,          int>(         int*,          int*, const unsigned int);
template void deepCopy<MemSpace::Host  , MemSpace::Host  , unsigned int>(unsigned int*, unsigned int*, const unsigned int);
template void deepCopy<MemSpace::Device, MemSpace::Host  , unsigned int>(unsigned int*, unsigned int*, const unsigned int);
template void deepCopy<MemSpace::Host  , MemSpace::Device, unsigned int>(unsigned int*, unsigned int*, const unsigned int);
template void deepCopy<MemSpace::Device, MemSpace::Device, unsigned int>(unsigned int*, unsigned int*, const unsigned int);
template void deepCopy<MemSpace::Host  , MemSpace::Host  ,         char>(        char*,         char*, const unsigned int);
template void deepCopy<MemSpace::Device, MemSpace::Host  ,         char>(        char*,         char*, const unsigned int);
template void deepCopy<MemSpace::Host  , MemSpace::Device,         char>(        char*,         char*, const unsigned int);
template void deepCopy<MemSpace::Device, MemSpace::Device,         char>(        char*,         char*, const unsigned int);

template void memAllocArray<MemSpace::Host  ,       double>(      double**, const unsigned int);
template void memAllocArray<MemSpace::Host  ,        float>(       float**, const unsigned int);
template void memAllocArray<MemSpace::Host  ,          int>(         int**, const unsigned int);
template void memAllocArray<MemSpace::Host  , unsigned int>(unsigned int**, const unsigned int);
template void memAllocArray<MemSpace::Host  ,         char>(        char**, const unsigned int);
template void memAllocArray<MemSpace::Device,       double>(      double**, const unsigned int);
template void memAllocArray<MemSpace::Device,        float>(       float**, const unsigned int);
template void memAllocArray<MemSpace::Device,          int>(         int**, const unsigned int);
template void memAllocArray<MemSpace::Device,         char>(        char**, const unsigned int);
template void memAllocArray<MemSpace::Device, unsigned int>(unsigned int**, const unsigned int);

template void memFreeArray<MemSpace::Host  ,       double>(      double*);
template void memFreeArray<MemSpace::Host  ,        float>(       float*);
template void memFreeArray<MemSpace::Host  , unsigned int>(unsigned int*);
template void memFreeArray<MemSpace::Host  ,          int>(         int*);
template void memFreeArray<MemSpace::Host  ,         char>(        char*);
template void memFreeArray<MemSpace::Device,       double>(      double*);
template void memFreeArray<MemSpace::Device,        float>(       float*);
template void memFreeArray<MemSpace::Device, unsigned int>(unsigned int*);
template void memFreeArray<MemSpace::Device,          int>(         int*);
template void memFreeArray<MemSpace::Device,         char>(        char*);

template int          numDistSolSteps(int);
template std::size_t  numDistSolSteps(std::size_t);
template unsigned int numDistSolSteps(unsigned int);


} // namespace cmpk
