
#ifndef _COMPACK3D_PENTA_KERNELS_CUH
#define _COMPACK3D_PENTA_KERNELS_CUH

namespace cmpk {
namespace penta {

template<typename RealType>
void localSolPentaPCRDimI(RealType*, const RealType*, const RealType*, const RealType*, const RealType*, const RealType*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);

template<typename RealType>
void localSolPentaPCRDimJ(RealType*, const RealType*, const RealType*, const RealType*, const RealType*, const RealType*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);

template<typename RealType>
void localSolPentaPCRDimK(RealType*, const RealType*, const RealType*, const RealType*, const RealType*, const RealType*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);

template<typename RealType, typename RealTypeComm>
void reduceCurrBlockSymm(RealType*, const RealTypeComm*, const RealTypeComm*, const RealType*, const RealType*, const RealType*, const unsigned int);

template<typename RealType, typename RealTypeComm>
void reduceCurrBlockOneSide(RealType*, const RealTypeComm*, const RealType*, const RealType*, const unsigned int);

template<typename RealType>
void calcReducedSystemRHSLocal(RealType*, const RealType*, const RealType*, const RealType*, const RealType*, const unsigned int);

template<typename RealType>
void updateLocalSolDimI(RealType*, const RealType*, const RealType*, const RealType*, const RealType*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);

template<typename RealType>
void updateLocalSolDimJ(RealType*, const RealType*, const RealType*, const RealType*, const RealType*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);

template<typename RealType>
void updateLocalSolDimK(RealType*, const RealType*, const RealType*, const RealType*, const RealType*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);

} // namespace penta
} // namespace cmpk

#endif // _COMPACK3D_PENTA_KERNELS_CUH
