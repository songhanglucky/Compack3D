
#include "Compack3D_utils_kernels.cuh"
#include <cassert>

namespace cmpk {

template<typename IntType> __device__
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



/*!
 * Index of factorization
 * \param [in]  i                    index of row
 * \param [in]  N                    size of the system
 * \param [in]  stride               current stride
 * \param [in]  max_sub_sys_size     maxium allowed sub-system size that can be solved directly
 * \returns     factorization index at the level
 */
template<typename IntType> __device__
IntType locFactIdx(IntType i, IntType N, IntType stride, IntType max_sub_sys_size) {
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



/*!
 * Copy slices from a 3D array in i-direction and save the data to a flattened 2D array
 * \param[out]    x2d                 flattened 2D array of size Ni * Nj * Nk
 * \param[ in]    x3d                 original 3D array
 * \param[ in]    Ni                  number of entries in i-direction
 * \param[ in]    Nj                  number of entries in j-direction
 * \param[ in]    Nk                  number of entries in k-direction
 * \param[ in]    arr_stride_i        3D array access stride in i-dimension, from i to i+1
 * \param[ in]    arr_stride_j        3D array access stride in j-dimension, from j to j+1
 * \note x2d is flattened in the non-slicing directions which has the size of Ni x (Nj * Nk)
 */
template<typename RealType> __global__
void kernelCopySlicesFrom3DArrayDimI(
              RealType* __restrict__ x2d,
        const RealType* __restrict__ x3d,
        const unsigned int Ni,
        const unsigned int Nj,
        const unsigned int Nk,
        const unsigned int arr_stride_i,
        const unsigned int arr_stride_j
    )
{
    #define IDX_I (blockIdx.z * blockDim.z + threadIdx.z)
    #define IDX_J (blockIdx.y * blockDim.y + threadIdx.y)
    #define IDX_K (blockIdx.x * blockDim.x + threadIdx.x)
    if ((IDX_I < Ni) && (IDX_J < Nj) && (IDX_K < Nk)) {
        x2d[IDX_I * Nj * Nk + IDX_J * Nk + IDX_K] = x3d[IDX_I * arr_stride_i + IDX_J * arr_stride_j + IDX_K];
    }
    #undef IDX_I
    #undef IDX_J
    #undef IDX_K
}



/*!
 * Launch "kernelCopySlicesFrom3DArrayDimI"
 * \param[out]    x2d                 flattened 2D array of size Ni * Nj * Nk
 * \param[ in]    x3d                 original 3D array
 * \param[ in]    Ni                  number of entries in i-direction
 * \param[ in]    Nj                  number of entries in j-direction
 * \param[ in]    Nk                  number of entries in k-direction
 * \param[ in]    arr_stride_i        3D array access stride in i-dimension, from i to i+1
 * \param[ in]    arr_stride_j        3D array access stride in j-dimension, from j to j+1
 * \note x2d is flattened in the non-slicing directions which has the size of Ni x (Nj * Nk)
 */
template<typename RealType>
void copySlicesFrom3DArrayDimI(
              RealType*    x2d,
        const RealType*    x3d,
        const unsigned int Ni,
        const unsigned int Nj,
        const unsigned int Nk,
        const unsigned int arr_stride_i,
        const unsigned int arr_stride_j
    )
{
    unsigned int num_threads_k = 64;
    while (num_threads_k > Nk) num_threads_k >>= 1;
    unsigned int num_threads_j = 4;
    while (num_threads_j > Nj) num_threads_j >>= 1;
    unsigned int num_threads_i = 2;
    while (num_threads_i > Ni) num_threads_i >>= 1;
    dim3 block_size = {num_threads_k, num_threads_j, num_threads_i};
    dim3  grid_size = {(Nk + num_threads_k - 1) / num_threads_k, (Nj + num_threads_j - 1) / num_threads_j, (Ni + num_threads_i - 1) / num_threads_i};
    kernelCopySlicesFrom3DArrayDimI<RealType><<<grid_size, block_size>>>(x2d, x3d, Ni, Nj, Nk, arr_stride_i, arr_stride_j);
}



/*!
 * Copy slices from a 3D array in j-direction and save the data to a flattened 2D array
 * \param[out]    x2d                 flattened 2D array of size Ni * Nj * Nk
 * \param[ in]    x3d                 original 3D array
 * \param[ in]    Ni                  number of entries in i-direction
 * \param[ in]    Nj                  number of entries in j-direction
 * \param[ in]    Nk                  number of entries in k-direction
 * \param[ in]    arr_stride_i        3D array access stride in i-dimension, from i to i+1
 * \param[ in]    arr_stride_j        3D array access stride in j-dimension, from j to j+1
 * \note x2d is flattened in the non-slicing directions which has the size of Nj x (Ni * Nk)
 */
template<typename RealType> __global__
void kernelCopySlicesFrom3DArrayDimJ(
              RealType* __restrict__ x2d,
        const RealType* __restrict__ x3d,
        const unsigned int Ni,
        const unsigned int Nj,
        const unsigned int Nk,
        const unsigned int arr_stride_i,
        const unsigned int arr_stride_j
    )
{
    #define IDX_I (blockIdx.z * blockDim.z + threadIdx.z)
    #define IDX_J (blockIdx.y * blockDim.y + threadIdx.y)
    #define IDX_K (blockIdx.x * blockDim.x + threadIdx.x)
    if ((IDX_I < Ni) && (IDX_J < Nj) && (IDX_K < Nk)) {
        x2d[IDX_J * Ni * Nk + IDX_I * Nk + IDX_K] = x3d[IDX_I * arr_stride_i + IDX_J * arr_stride_j + IDX_K];
    }
    #undef IDX_I
    #undef IDX_J
    #undef IDX_K
}



/*!
 * Launch "kernelCopySlicesFrom3DArrayDimJ"
 * \param[out]    x2d                 flattened 2D array of size Ni * Nj * Nk
 * \param[ in]    x3d                 original 3D array
 * \param[ in]    Ni                  number of entries in i-direction
 * \param[ in]    Nj                  number of entries in j-direction
 * \param[ in]    Nk                  number of entries in k-direction
 * \param[ in]    arr_stride_i        3D array access stride in i-dimension, from i to i+1
 * \param[ in]    arr_stride_j        3D array access stride in j-dimension, from j to j+1
 * \note x2d is flattened in the non-slicing directions which has the size of Nj x (Ni * Nk)
 */
template<typename RealType>
void copySlicesFrom3DArrayDimJ(
              RealType*    x2d,
        const RealType*    x3d,
        const unsigned int Ni,
        const unsigned int Nj,
        const unsigned int Nk,
        const unsigned int arr_stride_i,
        const unsigned int arr_stride_j
    )
{
    unsigned int num_threads_k = 64;
    while (num_threads_k > Nk) num_threads_k >>= 1;
    unsigned int num_threads_j = 2;
    while (num_threads_j > Nj) num_threads_j >>= 1;
    unsigned int num_threads_i = 4;
    while (num_threads_i > Ni) num_threads_i >>= 1;
    dim3 block_size = {num_threads_k, num_threads_j, num_threads_i};
    dim3  grid_size = {(Nk + num_threads_k - 1) / num_threads_k, (Nj + num_threads_j - 1) / num_threads_j, (Ni + num_threads_i - 1) / num_threads_i};
    kernelCopySlicesFrom3DArrayDimJ<RealType><<<grid_size, block_size>>>(x2d, x3d, Ni, Nj, Nk, arr_stride_i, arr_stride_j);
}



/*!
 * Copy slice from a 3D array in k-direction and save the data to a flattened 2D array
 * \param[out]    x2d                 flattened 2D array of size Ni * Nj * Nk
 * \param[ in]    x3d                 original 3D array
 * \param[ in]    Ni                  number of entries in i-direction
 * \param[ in]    Nj                  number of entries in j-direction
 * \param[ in]    Nk                  number of entries in k-direction
 * \param[ in]    arr_stride_i        3D array access stride in i-dimension, from i to i+1
 * \param[ in]    arr_stride_j        3D array access stride in j-dimension, from j to j+1
 * \note x2d is flattened in the non-slicing directions which has the size of Nk x (Ni * Nj)
 */
template<unsigned int NUM_THREADS_I, unsigned int NUM_THREADS_J, unsigned int NUM_THREADS_K, typename RealType> __global__
void kernelCopySlicesFrom3DArrayDimK(
              RealType* __restrict__ x2d,
        const RealType* __restrict__ x3d,
        const unsigned int Ni,
        const unsigned int Nj,
        const unsigned int Nk,
        const unsigned int arr_stride_i,
        const unsigned int arr_stride_j
    )
{
    assert(NUM_THREADS_I * NUM_THREADS_J * NUM_THREADS_K == blockDim.x);
    extern __shared__ char smem_general[];
    RealType* smem_buf = reinterpret_cast<RealType*>(smem_general);
    #define TID_I (threadIdx.x / (NUM_THREADS_J * NUM_THREADS_K))
    #define TID_J ((threadIdx.x / NUM_THREADS_K) % NUM_THREADS_J)
    #define TID_K (threadIdx.x % NUM_THREADS_K)
    #define IDX_I (blockIdx.z * NUM_THREADS_I + TID_I)
    #define IDX_J (blockIdx.y * NUM_THREADS_J + TID_J)
    #define IDX_K (blockIdx.x * NUM_THREADS_K + TID_K)
    if ((IDX_I < Ni) && (IDX_J < Nj) && (IDX_K < Nk)) {
        smem_buf[TID_K * (1 + NUM_THREADS_I * NUM_THREADS_J) + TID_I * NUM_THREADS_J + TID_J] = x3d[IDX_I * arr_stride_i + IDX_J * arr_stride_j + IDX_K];
    }
    #undef TID_I
    #undef TID_J
    #undef TID_K
    #undef IDX_I
    #undef IDX_J
    #undef IDX_K
    __syncthreads();
    #define TID_K (threadIdx.x / (NUM_THREADS_I * NUM_THREADS_J))
    #define TID_I ((threadIdx.x / NUM_THREADS_J) % NUM_THREADS_I)
    #define TID_J (threadIdx.x % NUM_THREADS_J)
    #define IDX_I (blockIdx.z * NUM_THREADS_I + TID_I)
    #define IDX_J (blockIdx.y * NUM_THREADS_J + TID_J)
    #define IDX_K (blockIdx.x * NUM_THREADS_K + TID_K)
    if ((IDX_I < Ni) && (IDX_J < Nj) && (IDX_K < Nk)) {
        x2d[IDX_K * Ni * Nj + IDX_I * Nj + IDX_J] = smem_buf[TID_K * (1 + NUM_THREADS_I * NUM_THREADS_J) + TID_I * NUM_THREADS_J + TID_J];
    }
    #undef TID_I
    #undef TID_J
    #undef TID_K
    #undef IDX_I
    #undef IDX_J
    #undef IDX_K
}



/*!
 * Launch "kernelCopySlicesFrom3DArrayDimK"
 * \param[out]    x2d                 flattened 2D array of size Ni * Nj * Nk
 * \param[ in]    x3d                 original 3D array
 * \param[ in]    Ni                  number of entries in i-direction
 * \param[ in]    Nj                  number of entries in j-direction
 * \param[ in]    Nk                  number of entries in k-direction
 * \param[ in]    arr_stride_i        3D array access stride in i-dimension, from i to i+1
 * \param[ in]    arr_stride_j        3D array access stride in j-dimension, from j to j+1
 * \note x2d is flattened in the non-slicing directions which has the size of Nk x (Ni * Nj)
 */
template<typename RealType>
void copySlicesFrom3DArrayDimK(
              RealType*    x2d,
        const RealType*    x3d,
        const unsigned int Ni,
        const unsigned int Nj,
        const unsigned int Nk,
        const unsigned int arr_stride_i,
        const unsigned int arr_stride_j
    )
{
    constexpr unsigned int NUM_THREADS_I =   1;
    constexpr unsigned int NUM_THREADS_J = 128;
    constexpr unsigned int NUM_THREADS_K =   2;
    constexpr unsigned int BLOCK_SIZE    = NUM_THREADS_I * NUM_THREADS_J * NUM_THREADS_K;
    constexpr unsigned int SMEM_SIZE     = NUM_THREADS_K * (NUM_THREADS_I * NUM_THREADS_J + 1) * sizeof(RealType); // "+1" to avoid bank conflict
    dim3  grid_size = dim3((Nk + NUM_THREADS_K - 1) / NUM_THREADS_K, (Nj + NUM_THREADS_J - 1) / NUM_THREADS_J, (Ni + NUM_THREADS_I - 1) / NUM_THREADS_I);
    kernelCopySlicesFrom3DArrayDimK<NUM_THREADS_I, NUM_THREADS_J, NUM_THREADS_K, RealType><<<grid_size, BLOCK_SIZE, SMEM_SIZE>>>(x2d, x3d, Ni, Nj, Nk, arr_stride_i, arr_stride_j);
}



/*!
 * Copy slices to a 3D array in i-direction and save the data to a flattened 2D array
 * \param[out]    x3d                 original 3D array
 * \param[ in]    x2d                 flattened 2D array of size Ni * Nj * Nk
 * \param[ in]    Ni                  number of entries in i-direction
 * \param[ in]    Nj                  number of entries in j-direction
 * \param[ in]    Nk                  number of entries in k-direction
 * \param[ in]    arr_stride_i        3D array access stride in i-dimension, from i to i+1
 * \param[ in]    arr_stride_j        3D array access stride in j-dimension, from j to j+1
 * \note x2d is flattened in the non-slicing directions which has the size of Ni x (Nj * Nk)
 */
template<typename RealType> __global__
void kernelCopySlicesTo3DArrayDimI(
              RealType* __restrict__ x3d,
        const RealType* __restrict__ x2d,
        const unsigned int Ni,
        const unsigned int Nj,
        const unsigned int Nk,
        const unsigned int arr_stride_i,
        const unsigned int arr_stride_j
    )
{
    #define IDX_I (blockIdx.z * blockDim.z + threadIdx.z)
    #define IDX_J (blockIdx.y * blockDim.y + threadIdx.y)
    #define IDX_K (blockIdx.x * blockDim.x + threadIdx.x)
    if ((IDX_I < Ni) && (IDX_J < Nj) && (IDX_K < Nk)) {
        x3d[IDX_I * arr_stride_i + IDX_J * arr_stride_j + IDX_K] = x2d[IDX_I * Nj * Nk + IDX_J * Nk + IDX_K];
    }
    #undef IDX_I
    #undef IDX_J
    #undef IDX_K
}



/*!
 * Launch "kernelCopySlicesTo3DArrayDimI"
 * \param[out]    x3d                 original 3D array
 * \param[ in]    x2d                 flattened 2D array of size Ni * Nj * Nk
 * \param[ in]    Ni                  number of entries in i-direction
 * \param[ in]    Nj                  number of entries in j-direction
 * \param[ in]    Nk                  number of entries in k-direction
 * \param[ in]    arr_stride_i        3D array access stride in i-dimension, from i to i+1
 * \param[ in]    arr_stride_j        3D array access stride in j-dimension, from j to j+1
 * \note x2d is flattened in the non-slicing directions which has the size of Ni x (Nj * Nk)
 */
template<typename RealType>
void copySlicesTo3DArrayDimI(
              RealType*    x3d,
        const RealType*    x2d,
        const unsigned int Ni,
        const unsigned int Nj,
        const unsigned int Nk,
        const unsigned int arr_stride_i,
        const unsigned int arr_stride_j
    )
{
    unsigned int num_threads_k = 64;
    while (num_threads_k > Nk) num_threads_k >>= 1;
    unsigned int num_threads_j = 4;
    while (num_threads_j > Nj) num_threads_j >>= 1;
    unsigned int num_threads_i = 2;
    while (num_threads_i > Ni) num_threads_i >>= 1;
    dim3 block_size = {num_threads_k, num_threads_j, num_threads_i};
    dim3  grid_size = {(Nk + num_threads_k - 1) / num_threads_k, (Nj + num_threads_j - 1) / num_threads_j, (Ni + num_threads_i - 1) / num_threads_i};
    kernelCopySlicesTo3DArrayDimI<RealType><<<grid_size, block_size>>>(x3d, x2d, Ni, Nj, Nk, arr_stride_i, arr_stride_j);
}



/*!
 * Copy slices to a 3D array in j-direction and save the data to a flattened 2D array
 * \param[out]    x3d                 original 3D array
 * \param[ in]    x2d                 flattened 2D array of size Ni * Nj * Nk
 * \param[ in]    Ni                  number of entries in i-direction
 * \param[ in]    Nj                  number of entries in j-direction
 * \param[ in]    Nk                  number of entries in k-direction
 * \param[ in]    arr_stride_i        3D array access stride in i-dimension, from i to i+1
 * \param[ in]    arr_stride_j        3D array access stride in j-dimension, from j to j+1
 * \note x2d is flattened in the non-slicing directions which has the size of Nj x (Ni * Nk)
 */
template<typename RealType> __global__
void kernelCopySlicesTo3DArrayDimJ(
              RealType* __restrict__ x3d,
        const RealType* __restrict__ x2d,
        const unsigned int Ni,
        const unsigned int Nj,
        const unsigned int Nk,
        const unsigned int arr_stride_i,
        const unsigned int arr_stride_j
    )
{
    #define IDX_I (blockIdx.z * blockDim.z + threadIdx.z)
    #define IDX_J (blockIdx.y * blockDim.y + threadIdx.y)
    #define IDX_K (blockIdx.x * blockDim.x + threadIdx.x)
    if ((IDX_I < Ni) && (IDX_J < Nj) && (IDX_K < Nk)) {
        x3d[IDX_I * arr_stride_i + IDX_J * arr_stride_j + IDX_K] = x2d[IDX_J * Ni * Nk + IDX_I * Nk + IDX_K];
    }
    #undef IDX_I
    #undef IDX_J
    #undef IDX_K
}



/*!
 * Launch "kernelCopySlicesTo3DArrayDimJ"
 * \param[out]    x3d                 original 3D array
 * \param[ in]    x2d                 flattened 2D array of size Ni * Nj * Nk
 * \param[ in]    Ni                  number of entries in i-direction
 * \param[ in]    Nj                  number of entries in j-direction
 * \param[ in]    Nk                  number of entries in k-direction
 * \param[ in]    arr_stride_i        3D array access stride in i-dimension, from i to i+1
 * \param[ in]    arr_stride_j        3D array access stride in j-dimension, from j to j+1
 * \note x2d is flattened in the non-slicing directions which has the size of Nj x (Ni * Nk)
 */
template<typename RealType>
void copySlicesTo3DArrayDimJ(
              RealType*    x3d,
        const RealType*    x2d,
        const unsigned int Ni,
        const unsigned int Nj,
        const unsigned int Nk,
        const unsigned int arr_stride_i,
        const unsigned int arr_stride_j
    )
{
    unsigned int num_threads_k = 64;
    while (num_threads_k > Nk) num_threads_k >>= 1;
    unsigned int num_threads_j = 2;
    while (num_threads_j > Nj) num_threads_j >>= 1;
    unsigned int num_threads_i = 4;
    while (num_threads_i > Ni) num_threads_i >>= 1;
    dim3 block_size = {num_threads_k, num_threads_j, num_threads_i};
    dim3  grid_size = {(Nk + num_threads_k - 1) / num_threads_k, (Nj + num_threads_j - 1) / num_threads_j, (Ni + num_threads_i - 1) / num_threads_i};
    kernelCopySlicesTo3DArrayDimJ<RealType><<<grid_size, block_size>>>(x3d, x2d, Ni, Nj, Nk, arr_stride_i, arr_stride_j);
}



/*!
 * Copy slice to a 3D array in k-direction and save the data to a flattened 2D array
 * \param[out]    x3d                 original 3D array
 * \param[ in]    x2d                 flattened 2D array of size Ni * Nj * Nk
 * \param[ in]    Ni                  number of entries in i-direction
 * \param[ in]    Nj                  number of entries in j-direction
 * \param[ in]    Nk                  number of entries in k-direction
 * \param[ in]    arr_stride_i        3D array access stride in i-dimension, from i to i+1
 * \param[ in]    arr_stride_j        3D array access stride in j-dimension, from j to j+1
 * \note x2d is flattened in the non-slicing directions which has the size of Nk x (Ni * Nj)
 */
template<unsigned int NUM_THREADS_I, unsigned int NUM_THREADS_J, unsigned int NUM_THREADS_K, typename RealType> __global__
void kernelCopySlicesTo3DArrayDimK(
              RealType* __restrict__ x3d,
        const RealType* __restrict__ x2d,
        const unsigned int Ni,
        const unsigned int Nj,
        const unsigned int Nk,
        const unsigned int arr_stride_i,
        const unsigned int arr_stride_j
    )
{
    assert(NUM_THREADS_I * NUM_THREADS_J * NUM_THREADS_K == blockDim.x);
    extern __shared__ char smem_general[];
    RealType* smem_buf = reinterpret_cast<RealType*>(smem_general);
    #define TID_K (threadIdx.x / (NUM_THREADS_I * NUM_THREADS_J))
    #define TID_I ((threadIdx.x / NUM_THREADS_J) % NUM_THREADS_I)
    #define TID_J (threadIdx.x % NUM_THREADS_J)
    #define IDX_I (blockIdx.z * NUM_THREADS_I + TID_I)
    #define IDX_J (blockIdx.y * NUM_THREADS_J + TID_J)
    #define IDX_K (blockIdx.x * NUM_THREADS_K + TID_K)
    if ((IDX_I < Ni) && (IDX_J < Nj) && (IDX_K < Nk)) {
        smem_buf[TID_K * (1 + NUM_THREADS_I * NUM_THREADS_J) + TID_I * NUM_THREADS_J + TID_J] = x2d[IDX_K * Ni * Nj + IDX_I * Nj + IDX_J];
    }
    #undef TID_I
    #undef TID_J
    #undef TID_K
    #undef IDX_I
    #undef IDX_J
    #undef IDX_K
    __syncthreads();
    #define TID_I (threadIdx.x / (NUM_THREADS_J * NUM_THREADS_K))
    #define TID_J ((threadIdx.x / NUM_THREADS_K) % NUM_THREADS_J)
    #define TID_K (threadIdx.x % NUM_THREADS_K)
    #define IDX_I (blockIdx.z * NUM_THREADS_I + TID_I)
    #define IDX_J (blockIdx.y * NUM_THREADS_J + TID_J)
    #define IDX_K (blockIdx.x * NUM_THREADS_K + TID_K)
    if ((IDX_I < Ni) && (IDX_J < Nj) && (IDX_K < Nk)) {
        x3d[IDX_I * arr_stride_i + IDX_J * arr_stride_j + IDX_K] = smem_buf[TID_K * (1 + NUM_THREADS_I * NUM_THREADS_J) + TID_I * NUM_THREADS_J + TID_J];
    }
    #undef TID_I
    #undef TID_J
    #undef TID_K
    #undef IDX_I
    #undef IDX_J
    #undef IDX_K
}



/*!
 * Launch "kernelCopySlicesTo3DArrayDimK"
 * \param[out]    x3d                 original 3D array
 * \param[ in]    x2d                 flattened 2D array of size Ni * Nj * Nk
 * \param[ in]    Ni                  number of entries in i-direction
 * \param[ in]    Nj                  number of entries in j-direction
 * \param[ in]    Nk                  number of entries in k-direction
 * \param[ in]    arr_stride_i        3D array access stride in i-dimension, from i to i+1
 * \param[ in]    arr_stride_j        3D array access stride in j-dimension, from j to j+1
 * \note x2d is flattened in the non-slicing directions which has the size of Nk x (Ni * Nj)
 */
template<typename RealType>
void copySlicesTo3DArrayDimK(
              RealType*    x3d,
        const RealType*    x2d,
        const unsigned int Ni,
        const unsigned int Nj,
        const unsigned int Nk,
        const unsigned int arr_stride_i,
        const unsigned int arr_stride_j
    )
{
    constexpr unsigned int NUM_THREADS_I =   1;
    constexpr unsigned int NUM_THREADS_J = 128;
    constexpr unsigned int NUM_THREADS_K =   2;
    constexpr unsigned int BLOCK_SIZE    = NUM_THREADS_I * NUM_THREADS_J * NUM_THREADS_K;
    constexpr unsigned int SMEM_SIZE     = NUM_THREADS_K * (NUM_THREADS_I * NUM_THREADS_J + 1) * sizeof(RealType); // "+1" to avoid bank conflict
    dim3  grid_size = dim3((Nk + NUM_THREADS_K - 1) / NUM_THREADS_K, (Nj + NUM_THREADS_J - 1) / NUM_THREADS_J, (Ni + NUM_THREADS_I - 1) / NUM_THREADS_I);
    kernelCopySlicesTo3DArrayDimK<NUM_THREADS_I, NUM_THREADS_J, NUM_THREADS_K, RealType><<<grid_size, BLOCK_SIZE, SMEM_SIZE>>>(x3d, x2d, Ni, Nj, Nk, arr_stride_i, arr_stride_j);
}



/*!
 * Copy 2D array and cast to a different type
 * \param[out]    x_out               buffer of 2D array for output
 * \param[ in]    x_in                buffer of 2D array for input
 * \param[ in]    Ni                  number of entries in i-dimension
 * \param[ in]    Nj                  number of entries in j-dimension
 * \param[ in]    arr_stride_i        2D array access stride in i-dimension, from i to i+1
 * \note The memory layout of the 2D array is column-major where the j-dimension is mapped to the aligned memory
 */
template<typename RealTypeDst, typename RealTypeSrc> __global__
void kernelCopyAndCast2D(
              RealTypeDst* __restrict__ x_out,
        const RealTypeSrc* __restrict__ x_in,
        const unsigned int              Ni,
        const unsigned int              Nj,
        const unsigned int              arr_stride_i
    )
{
    const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i < Ni) && (j < Nj)) {
        const unsigned int idx = i * arr_stride_i + j;
        x_out[idx] = static_cast<RealTypeDst>(x_in[idx]);
    }
}



/*!
 * Launch "kernelCopyAndCast2D"
 * \param[out]    x_out               buffer of 2D array for output
 * \param[ in]    x_in                buffer of 2D array for input
 * \param[ in]    Ni                  number of entries in i-dimension
 * \param[ in]    Nj                  number of entries in j-dimension
 * \param[ in]    arr_stride_i        2D array access stride in i-dimension, from i to i+1
 * \note The memory layout of the 2D array is column-major where the j-dimension is mapped to the aligned memory
 */
template<typename RealTypeDst, typename RealTypeSrc>
void copyAndCast2D(
              RealTypeDst* x_out,
        const RealTypeSrc* x_in,
        const unsigned int Ni,
        const unsigned int Nj,
        const unsigned int arr_stride_i
    )
{
    unsigned int num_threads_j = 256;
    while (num_threads_j > Nj) num_threads_j >>= 1;
    unsigned int num_threads_i = 2;
    while (num_threads_i > Ni) num_threads_i >>= 1;
    dim3 block_size = dim3(num_threads_j, num_threads_i);
    dim3  grid_size = dim3((Nj + num_threads_j - 1) / num_threads_j, (Ni + num_threads_i - 1) / num_threads_i);
    kernelCopyAndCast2D<RealTypeDst, RealTypeSrc><<<grid_size, block_size>>>(x_out, x_in, Ni, Nj, arr_stride_i);
}




// EXPLICIT INSTANTIATION

template __device__          int log2Ceil(         int);
template __device__ unsigned int log2Ceil(unsigned int);
template __device__       size_t log2Ceil(      size_t);
template __device__          int locFactIdx(   int,    int,    int,    int);
template __device__ unsigned int locFactIdx(unsigned int, unsigned int, unsigned int, unsigned int);
template __device__       size_t locFactIdx(size_t, size_t, size_t, size_t);

template void copySlicesFrom3DArrayDimI<double>(double*, const double*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
template void copySlicesFrom3DArrayDimI< float>( float*, const  float*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
template void copySlicesFrom3DArrayDimJ<double>(double*, const double*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
template void copySlicesFrom3DArrayDimJ< float>( float*, const  float*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
template void copySlicesFrom3DArrayDimK<double>(double*, const double*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
template void copySlicesFrom3DArrayDimK< float>( float*, const  float*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);

template void copySlicesTo3DArrayDimI<double>(double*, const double*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
template void copySlicesTo3DArrayDimI< float>( float*, const  float*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
template void copySlicesTo3DArrayDimJ<double>(double*, const double*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
template void copySlicesTo3DArrayDimJ< float>( float*, const  float*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
template void copySlicesTo3DArrayDimK<double>(double*, const double*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
template void copySlicesTo3DArrayDimK< float>( float*, const  float*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);

template void copyAndCast2D<double, double>(double*, const double*, const unsigned int, const unsigned int, const unsigned int);
template void copyAndCast2D< float, double>( float*, const double*, const unsigned int, const unsigned int, const unsigned int);


} // namespace cmpk
