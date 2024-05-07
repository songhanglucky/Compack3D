
#ifndef _COMPACK3D_UTILS_KERNELS_CUH
#define _COMPACK3D_UTILS_KERNELS_CUH

#ifdef __CUDA_ARCH__
#define __DEVICE__ __device__
#else
#define __DEVICE__
#endif

namespace cmpk {

template<typename IntType> __DEVICE__
extern IntType log2Ceil(IntType);

template<typename IntType> __DEVICE__
extern IntType locFactIdx(IntType, IntType, IntType, IntType);

template<typename RealType>
void copySlicesFrom3DArrayDimI(RealType*, const RealType*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);

template<typename RealType>
void copySlicesFrom3DArrayDimJ(RealType*, const RealType*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);

template<typename RealType>
void copySlicesFrom3DArrayDimK(RealType*, const RealType*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);

template<typename RealType>
void copySlicesTo3DArrayDimI(RealType*, const RealType*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);

template<typename RealType>
void copySlicesTo3DArrayDimJ(RealType*, const RealType*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);

template<typename RealType>
void copySlicesTo3DArrayDimK(RealType*, const RealType*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);

template<typename RealTypeDst, typename RealTypeSrc>
void copyAndCast2D(RealTypeDst*, const RealTypeSrc*, const unsigned int, const unsigned int, const unsigned int);

} // namespace cmpk

#undef __DEVICE__

#endif // __COMPACK3D_UTILS_KERNELS_CUH
