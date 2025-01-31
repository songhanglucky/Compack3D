
#ifndef _COMPACK3D_TRI_KERNELS_CUH
#define _COMPACK3D_TRI_KERNELS_CUH

namespace cmpk {
namespace tri {

template<typename RealType>
void localSolTriPCRDimI(RealType*, const RealType*, const RealType*, const RealType*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);

template<typename RealType>
void localSolTriPCRDimJ(RealType*, const RealType*, const RealType*, const RealType*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);

template<typename RealType>
void localSolTriPCRDimK(RealType*, const RealType*, const RealType*, const RealType*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);

template<typename RealType, typename RealTypeComm>
void reduceCurrBlockSymm(RealType*, const RealTypeComm*, const RealTypeComm*, const RealType, const RealType, const RealType, const unsigned int);

template<typename RealType, typename RealTypeComm>
void reduceCurrBlockOneSide(RealType*, const RealTypeComm*, const RealType, const RealType, const unsigned int);

template<typename RealType>
void calcReducedSystemRHSLocal(RealType*, const RealType*, const RealType*, const RealType, const RealType, const unsigned int);

template<typename RealType>
void updateLocalSolDimI(RealType*, const RealType*, const RealType*, const RealType*, const RealType*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);

template<typename RealType>
void updateLocalSolDimJ(RealType*, const RealType*, const RealType*, const RealType*, const RealType*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);

template<typename RealType>
void updateLocalSolDimK(RealType*, const RealType*, const RealType*, const RealType*, const RealType*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);

} // namespace tri
} // namespace cmpk


#endif
