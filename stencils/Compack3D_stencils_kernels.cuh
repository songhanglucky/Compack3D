
#ifndef _COMPACK3D_STENCILS_KERNELS_CUH
#define _COMPACK3D_STENCILS_KERNELS_CUH

#include "Compack3D_api.h"

namespace cmpk {
namespace stencil {

template<typename RealType>
void centralDiffCollStencil7DimI(RealType*, const RealType*, const RealType*, const RealType*, const RealType*, const RealType*, const RealType*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);

template<typename RealType>
void centralDiffCollStencil7DimJ(RealType*, const RealType*, const RealType*, const RealType*, const RealType*, const RealType*, const RealType*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);

template<typename RealType>
void centralDiffCollStencil7DimK(RealType*, const RealType*, const RealType*, const RealType*, const RealType*, const RealType*, const RealType*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);

} // namespace stencil
} // namespace cmpk
#endif // _COMPACK3D_STENCILS_KERNELS_CUH
