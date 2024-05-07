#include "Compack3D_api.h"
#include "Compack3D_stencils_kernels.cuh"

namespace cmpk {
namespace stencil {

/*!
 * Calculate collocated central difference f[i] = a * (f[i+1] - f[i-1]) + b * (f[i+2] - f[i-2]) + c * (f[i+3] - f[i-3])
 * \tparam RealType         real-value data type
 * \tparam NUM_THREADS_I    number of threads in i-dimension
 * \tparam NUM_THREADS_J    number of threads in j-dimension
 * \tparam NUM_THREADS_K    number of threads in k-dimension
 * \param [in]    f_out
 * \param [in]    f_in
 * \param [in]    f_in_halo_prev    buffer of halo exchange in the previous rank
 * \param [in]    f_in_halo_next    buffer of halo exchange in the next rank
 * \param [in]    a                 coefficient of f_in[i+1,:,:] - f_in[i-1,:,:]
 * \param [in]    b                 coefficient of f_in[i+2,:,:] - f_in[i-2,:,:]
 * \param [in]    c                 coefficient of f_in[i+3,:,:] - f_in[i-3,:,:]
 * \param [in]    Ni                number of grid points in the i-dimension within the current rank
 * \param [in]    Nj                number of grid points in the j-dimension within the current rank
 * \param [in]    Nk                number of grid points in the k-dimension within the current rank
 * \param [in]    arr_stride_i      array access stride in i-dimension, from i to i+1
 * \param [in]    arr_stride_j      array access stride in j-dimension, from j to j+1
 */
template<unsigned int NUM_THREADS_I, unsigned int NUM_THREADS_J, unsigned int NUM_THREADS_K, typename RealType> __global__
void kernelCentralDiffCollStencil7DimI(
        RealType* f_out, const RealType* f_in, const RealType* f_in_halo_prev, const RealType* f_in_halo_next,
        const RealType* a, const RealType* b, const RealType* c,
        const unsigned int Ni, const unsigned int Nj, const unsigned int Nk,
        const unsigned int arr_stride_i, const unsigned int arr_stride_j
    )
{
    #define TID_I ( threadIdx.x / NUM_THREADS_K  / NUM_THREADS_J)
    #define TID_J ((threadIdx.x / NUM_THREADS_K) % NUM_THREADS_J)
    #define TID_K ( threadIdx.x                  % NUM_THREADS_K)
    #define IDX_I (blockIdx.z * NUM_THREADS_I + TID_I)
    #define IDX_J (blockIdx.y * NUM_THREADS_J + TID_J)
    #define IDX_K (blockIdx.x * NUM_THREADS_K + TID_K)

    auto SMEM_IDX = [&](int i)->unsigned int { return (3 + i) * (NUM_THREADS_J * NUM_THREADS_K) + TID_J * NUM_THREADS_K + TID_K; };
    auto  ARR_IDX = [&](int i)->unsigned int { return i * arr_stride_i + IDX_J * arr_stride_j + IDX_K; };

    extern __shared__ char smem_general[];
    static_assert(NUM_THREADS_I > 6, "NUM_THREADS_I must be greater than 6 for a 7-point stencil.");
    RealType* f_smem = reinterpret_cast<RealType*>(smem_general);
    if ((IDX_J < Nj) && (IDX_K < Nk)) {
        for (unsigned int step = 0; step < 2; step++) {
            const int idx_i = static_cast<int>(IDX_I - 3 + NUM_THREADS_I * step);
            const int tid_i = static_cast<int>(TID_I - 3 + NUM_THREADS_I * step);
            if (tid_i < static_cast<int>(NUM_THREADS_I + 3)) {
                if (idx_i < 0) {
                    f_smem[SMEM_IDX(tid_i)] = f_in_halo_prev[ARR_IDX(idx_i + 3)];
                } else if (idx_i < static_cast<int>(Ni)) {
                    f_smem[SMEM_IDX(tid_i)] = f_in[ARR_IDX(idx_i)];
                } else if (idx_i < static_cast<int>(Ni + 3)) {
                    f_smem[SMEM_IDX(tid_i)] = f_in_halo_next[ARR_IDX(idx_i - Ni)];
                }
            }
        }
    }
    __syncthreads();
    if ((IDX_I < Ni) && (IDX_J < Nj) && (IDX_K < Nk)) {
        f_out[ARR_IDX(IDX_I)]
            = a[IDX_I] * (f_smem[SMEM_IDX(TID_I+1)] - f_smem[SMEM_IDX(TID_I-1)])
            + b[IDX_I] * (f_smem[SMEM_IDX(TID_I+2)] - f_smem[SMEM_IDX(TID_I-2)])
            + c[IDX_I] * (f_smem[SMEM_IDX(TID_I+3)] - f_smem[SMEM_IDX(TID_I-3)]);
    }
    

    #undef IDX_K
    #undef IDX_J
    #undef IDX_I
    #undef TID_K
    #undef TID_J
    #undef TID_I
}



/*!
 * Launch "kernelCentralDiffCollStencil7DimI"
 * \param [in]    f_out
 * \param [in]    f_in
 * \param [in]    f_in_halo_prev    buffer of halo exchange in the previous rank
 * \param [in]    f_in_halo_next    buffer of halo exchange in the next rank
 * \param [in]    a                 coefficient of f_in[i+1,:,:] - f_in[i-1,:,:]
 * \param [in]    b                 coefficient of f_in[i+2,:,:] - f_in[i-2,:,:]
 * \param [in]    c                 coefficient of f_in[i+3,:,:] - f_in[i-3,:,:]
 * \param [in]    Ni                number of grid points in the i-dimension within the current rank
 * \param [in]    Nj                number of grid points in the j-dimension within the current rank
 * \param [in]    Nk                number of grid points in the k-dimension within the current rank
 * \param [in]    arr_stride_i      array access stride in i-dimension, from i to i+1
 * \param [in]    arr_stride_j      array access stride in j-dimension, from j to j+1
 */
template<typename RealType>
void centralDiffCollStencil7DimI(
        RealType* f_out, const RealType* f_in, const RealType* f_in_halo_prev, const RealType* f_in_halo_next,
        const RealType* a, const RealType* b, const RealType* c,
        const unsigned int Ni, const unsigned int Nj, const unsigned int Nk,
        const unsigned int arr_stride_i, const unsigned int arr_stride_j
    )
{
    constexpr unsigned int NUM_THREADS_I =  8;
    constexpr unsigned int NUM_THREADS_J =  1;
    constexpr unsigned int NUM_THREADS_K = 32;
    constexpr unsigned int BLOCK_SIZE    = NUM_THREADS_I * NUM_THREADS_J * NUM_THREADS_K;
    constexpr unsigned int SMEM_SIZE     = (NUM_THREADS_I + 6) * (NUM_THREADS_J * NUM_THREADS_K) * sizeof(RealType);
    dim3 grid_size = dim3((Nk + NUM_THREADS_K - 1) / NUM_THREADS_K, (Nj + NUM_THREADS_J - 1) / NUM_THREADS_J, (Ni + NUM_THREADS_I - 1) / NUM_THREADS_I);
    kernelCentralDiffCollStencil7DimI<NUM_THREADS_I, NUM_THREADS_J, NUM_THREADS_K, RealType><<<grid_size, BLOCK_SIZE, SMEM_SIZE>>> (
        f_out, f_in, f_in_halo_prev, f_in_halo_next, a, b, c, Ni, Nj, Nk, arr_stride_i, arr_stride_j   
    );

}



/*!
 * Calculate collocated central difference f[j] = a * (f[j+1] - f[j-1]) + b * (f[j+2] - f[j-2]) + c * (f[j+3] - f[j-3])
 * \tparam RealType         real-value data type
 * \tparam NUM_THREADS_I    number of threads in i-dimension
 * \tparam NUM_THREADS_J    number of threads in j-dimension
 * \tparam NUM_THREADS_K    number of threads in k-dimension
 * \param [in]    f_out
 * \param [in]    f_in
 * \param [in]    f_in_halo_prev    buffer of halo exchange in the previous rank
 * \param [in]    f_in_halo_next    buffer of halo exchange in the next rank
 * \param [in]    a                 coefficient of f_in[:,j+1,:] - f_in[:,j-1,:]
 * \param [in]    b                 coefficient of f_in[:,j+2,:] - f_in[:,j-2,:]
 * \param [in]    c                 coefficient of f_in[:,j+3,:] - f_in[:,j-3,:]
 * \param [in]    Ni                number of grid points in the i-dimension within the current rank
 * \param [in]    Nj                number of grid points in the j-dimension within the current rank
 * \param [in]    Nk                number of grid points in the k-dimension within the current rank
 * \param [in]    arr_stride_i      array access stride in i-dimension, from i to i+1
 * \param [in]    arr_stride_j      array access stride in j-dimension, from j to j+1
 */
template<unsigned int NUM_THREADS_I, unsigned int NUM_THREADS_J, unsigned int NUM_THREADS_K, typename RealType> __global__
void kernelCentralDiffCollStencil7DimJ(
        RealType* f_out, const RealType* f_in, const RealType* f_in_halo_prev, const RealType* f_in_halo_next,
        const RealType* a, const RealType* b, const RealType* c,
        const unsigned int Ni, const unsigned int Nj, const unsigned int Nk,
        const unsigned int arr_stride_i, const unsigned int arr_stride_j
    )
{
    #define TID_I ( threadIdx.x / NUM_THREADS_K  / NUM_THREADS_J)
    #define TID_J ((threadIdx.x / NUM_THREADS_K) % NUM_THREADS_J)
    #define TID_K ( threadIdx.x                  % NUM_THREADS_K)
    #define IDX_I (blockIdx.z * NUM_THREADS_I + TID_I)
    #define IDX_J (blockIdx.y * NUM_THREADS_J + TID_J)
    #define IDX_K (blockIdx.x * NUM_THREADS_K + TID_K)

    auto SMEM_IDX = [&](int j)->unsigned int { return TID_I * (6 + NUM_THREADS_J) * NUM_THREADS_K + (j + 3) * NUM_THREADS_K + TID_K; };
    auto  ARR_IDX = [&](int j)->unsigned int { return IDX_I * arr_stride_i + j * arr_stride_j + IDX_K; };
    auto  BUF_IDX = [&](int j)->unsigned int { return IDX_I * 3 * arr_stride_j + j * arr_stride_j + IDX_K; };

    extern __shared__ char smem_general[];
    static_assert(NUM_THREADS_J > 6, "NUM_THREADS_J must be greater than 6 for a 7-point stencil.");
    RealType* f_smem = reinterpret_cast<RealType*>(smem_general);
    if ((IDX_I < Ni) && (IDX_K < Nk)) {
        for (unsigned int step = 0; step < 2; step++) {
            const int idx_j = static_cast<int>(IDX_J - 3 + NUM_THREADS_J * step);
            const int tid_j = static_cast<int>(TID_J - 3 + NUM_THREADS_J * step);
            if (tid_j < static_cast<int>(NUM_THREADS_J + 3)) {
                if (idx_j < 0) {
                    f_smem[SMEM_IDX(tid_j)] = f_in_halo_prev[BUF_IDX(idx_j + 3)];
                } else if (idx_j < static_cast<int>(Nj)) {
                    f_smem[SMEM_IDX(tid_j)] = f_in[ARR_IDX(idx_j)];
                } else if (idx_j < static_cast<int>(Nj + 3)) {
                    f_smem[SMEM_IDX(tid_j)] = f_in_halo_next[BUF_IDX(idx_j - Nj)];
                }
            }
        }
    }
    __syncthreads();
    if ((IDX_I < Ni) && (IDX_J < Nj) && (IDX_K < Nk)) {
        f_out[ARR_IDX(IDX_J)]
            = a[IDX_J] * (f_smem[SMEM_IDX(TID_J+1)] - f_smem[SMEM_IDX(TID_J-1)])
            + b[IDX_J] * (f_smem[SMEM_IDX(TID_J+2)] - f_smem[SMEM_IDX(TID_J-2)])
            + c[IDX_J] * (f_smem[SMEM_IDX(TID_J+3)] - f_smem[SMEM_IDX(TID_J-3)]);
    }
    

    #undef IDX_K
    #undef IDX_J
    #undef IDX_I
    #undef TID_K
    #undef TID_J
    #undef TID_I
}



/*!
 * Launch "kernelCentralDiffCollStencil7DimJ"
 * \param [in]    f_out
 * \param [in]    f_in
 * \param [in]    f_in_halo_prev    buffer of halo exchange in the previous rank
 * \param [in]    f_in_halo_next    buffer of halo exchange in the next rank
 * \param [in]    a                 coefficient of f_in[i+1,:,:] - f_in[i-1,:,:]
 * \param [in]    b                 coefficient of f_in[i+2,:,:] - f_in[i-2,:,:]
 * \param [in]    c                 coefficient of f_in[i+3,:,:] - f_in[i-3,:,:]
 * \param [in]    Ni                number of grid points in the i-dimension within the current rank
 * \param [in]    Nj                number of grid points in the j-dimension within the current rank
 * \param [in]    Nk                number of grid points in the k-dimension within the current rank
 * \param [in]    arr_stride_i      array access stride in i-dimension, from i to i+1
 * \param [in]    arr_stride_j      array access stride in j-dimension, from j to j+1
 */
template<typename RealType>
void centralDiffCollStencil7DimJ(
        RealType* f_out, const RealType* f_in, const RealType* f_in_halo_prev, const RealType* f_in_halo_next,
        const RealType* a, const RealType* b, const RealType* c,
        const unsigned int Ni, const unsigned int Nj, const unsigned int Nk,
        const unsigned int arr_stride_i, const unsigned int arr_stride_j
    )
{
    constexpr unsigned int NUM_THREADS_I =  1;
    constexpr unsigned int NUM_THREADS_J =  8;
    constexpr unsigned int NUM_THREADS_K = 32;
    constexpr unsigned int BLOCK_SIZE    = NUM_THREADS_I * NUM_THREADS_J * NUM_THREADS_K;
    constexpr unsigned int SMEM_SIZE     = NUM_THREADS_I * (NUM_THREADS_J + 6) * NUM_THREADS_K * sizeof(RealType);
    dim3 grid_size = dim3((Nk + NUM_THREADS_K - 1) / NUM_THREADS_K, (Nj + NUM_THREADS_J - 1) / NUM_THREADS_J, (Ni + NUM_THREADS_I - 1) / NUM_THREADS_I);
    kernelCentralDiffCollStencil7DimJ<NUM_THREADS_I, NUM_THREADS_J, NUM_THREADS_K, RealType><<<grid_size, BLOCK_SIZE, SMEM_SIZE>>> (
        f_out, f_in, f_in_halo_prev, f_in_halo_next, a, b, c, Ni, Nj, Nk, arr_stride_i, arr_stride_j   
    );

}



/*!
 * Calculate collocated central difference f[k] = a * (f[k+1] - f[k-1]) + b * (f[k+2] - f[k-2]) + c * (f[k+3] - f[k-3])
 * \tparam RealType         real-value data type
 * \tparam NUM_THREADS_I    number of threads in i-dimension
 * \tparam NUM_THREADS_J    number of threads in j-dimension
 * \tparam NUM_THREADS_K    number of threads in k-dimension
 * \param [in]    f_out
 * \param [in]    f_in
 * \param [in]    f_in_halo_prev    buffer of halo exchange in the previous rank
 * \param [in]    f_in_halo_next    buffer of halo exchange in the next rank
 * \param [in]    a                 coefficient of f_in[:,j+1,:] - f_in[:,j-1,:]
 * \param [in]    b                 coefficient of f_in[:,j+2,:] - f_in[:,j-2,:]
 * \param [in]    c                 coefficient of f_in[:,j+3,:] - f_in[:,j-3,:]
 * \param [in]    Ni                number of grid points in the i-dimension within the current rank
 * \param [in]    Nj                number of grid points in the j-dimension within the current rank
 * \param [in]    Nk                number of grid points in the k-dimension within the current rank
 * \param [in]    arr_stride_i      array access stride in i-dimension, from i to i+1
 * \param [in]    arr_stride_j      array access stride in j-dimension, from j to j+1
 */
template<unsigned int NUM_THREADS_I, unsigned int NUM_THREADS_J, unsigned int NUM_THREADS_K, typename RealType> __global__
void kernelCentralDiffCollStencil7DimK(
        RealType* f_out, const RealType* f_in, const RealType* f_in_halo_prev, const RealType* f_in_halo_next,
        const RealType* a, const RealType* b, const RealType* c,
        const unsigned int Ni, const unsigned int Nj, const unsigned int Nk,
        const unsigned int arr_stride_i, const unsigned int arr_stride_j
    )
{
    #define TID_I ( threadIdx.x / NUM_THREADS_K  / NUM_THREADS_J)
    #define TID_J ((threadIdx.x / NUM_THREADS_K) % NUM_THREADS_J)
    #define TID_K ( threadIdx.x                  % NUM_THREADS_K)
    #define IDX_I (blockIdx.z * NUM_THREADS_I + TID_I)
    #define IDX_J (blockIdx.y * NUM_THREADS_J + TID_J)
    #define IDX_K (blockIdx.x * NUM_THREADS_K + TID_K)

    auto SMEM_IDX = [&](int k)->unsigned int { return TID_I * NUM_THREADS_J * (NUM_THREADS_K + 6) + TID_J * (NUM_THREADS_K + 6) + k + 3; };
    auto  ARR_IDX = [&](int k)->unsigned int { return IDX_I * arr_stride_i + IDX_J * arr_stride_j + k; };
    auto  BUF_IDX = [&](int k)->unsigned int { return IDX_I * arr_stride_i / arr_stride_j * 3 + IDX_J * 3 + k; };

    extern __shared__ char smem_general[];
    static_assert(NUM_THREADS_K > 6, "NUM_THREADS_K must be greater than 6 for a 7-point stencil.");
    RealType* f_smem = reinterpret_cast<RealType*>(smem_general);
    if ((IDX_I < Ni) && (IDX_J < Nj)) {
        for (unsigned int step = 0; step < 2; step++) {
            const int idx_k = static_cast<int>(IDX_K - 3 + NUM_THREADS_K * step);
            const int tid_k = static_cast<int>(TID_K - 3 + NUM_THREADS_K * step);
            if (tid_k < static_cast<int>(NUM_THREADS_K + 3)) {
                if (idx_k < 0) {
                    f_smem[SMEM_IDX(tid_k)] = f_in_halo_prev[BUF_IDX(idx_k + 3)];
                } else if (idx_k < static_cast<int>(Nk)) {
                    f_smem[SMEM_IDX(tid_k)] = f_in[ARR_IDX(idx_k)];
                } else if (idx_k < static_cast<int>(Nk + 3)) {
                    f_smem[SMEM_IDX(tid_k)] = f_in_halo_next[BUF_IDX(idx_k - Nk)];
                }
            }
        }
    }
    __syncthreads();
    if ((IDX_I < Ni) && (IDX_J < Nj) && (IDX_K < Nk)) {
        f_out[ARR_IDX(IDX_K)]
            = a[IDX_K] * (f_smem[SMEM_IDX(TID_K+1)] - f_smem[SMEM_IDX(TID_K-1)])
            + b[IDX_K] * (f_smem[SMEM_IDX(TID_K+2)] - f_smem[SMEM_IDX(TID_K-2)])
            + c[IDX_K] * (f_smem[SMEM_IDX(TID_K+3)] - f_smem[SMEM_IDX(TID_K-3)]);
    }
    

    #undef IDX_K
    #undef IDX_J
    #undef IDX_I
    #undef TID_K
    #undef TID_J
    #undef TID_I
}



/*!
 * Launch "kernelCentralDiffCollStencil7DimK"
 * \param [in]    f_out
 * \param [in]    f_in
 * \param [in]    f_in_halo_prev    buffer of halo exchange in the previous rank
 * \param [in]    f_in_halo_next    buffer of halo exchange in the next rank
 * \param [in]    a                 coefficient of f_in[i+1,:,:] - f_in[i-1,:,:]
 * \param [in]    b                 coefficient of f_in[i+2,:,:] - f_in[i-2,:,:]
 * \param [in]    c                 coefficient of f_in[i+3,:,:] - f_in[i-3,:,:]
 * \param [in]    Ni                number of grid points in the i-dimension within the current rank
 * \param [in]    Nj                number of grid points in the j-dimension within the current rank
 * \param [in]    Nk                number of grid points in the k-dimension within the current rank
 * \param [in]    arr_stride_i      array access stride in i-dimension, from i to i+1
 * \param [in]    arr_stride_j      array access stride in j-dimension, from j to j+1
 */
template<typename RealType>
void centralDiffCollStencil7DimK(
        RealType* f_out, const RealType* f_in, const RealType* f_in_halo_prev, const RealType* f_in_halo_next,
        const RealType* a, const RealType* b, const RealType* c,
        const unsigned int Ni, const unsigned int Nj, const unsigned int Nk,
        const unsigned int arr_stride_i, const unsigned int arr_stride_j
    )
{
    constexpr unsigned int NUM_THREADS_I =   1;
    constexpr unsigned int NUM_THREADS_J =   4;
    constexpr unsigned int NUM_THREADS_K =  64;
    constexpr unsigned int BLOCK_SIZE    = NUM_THREADS_I * NUM_THREADS_J * NUM_THREADS_K;
    constexpr unsigned int SMEM_SIZE     = NUM_THREADS_I * NUM_THREADS_J * (NUM_THREADS_K + 6) * sizeof(RealType);
    dim3 grid_size = dim3((Nk + NUM_THREADS_K - 1) / NUM_THREADS_K, (Nj + NUM_THREADS_J - 1) / NUM_THREADS_J, (Ni + NUM_THREADS_I - 1) / NUM_THREADS_I);
    kernelCentralDiffCollStencil7DimK<NUM_THREADS_I, NUM_THREADS_J, NUM_THREADS_K, RealType><<<grid_size, BLOCK_SIZE, SMEM_SIZE>>> (
        f_out, f_in, f_in_halo_prev, f_in_halo_next, a, b, c, Ni, Nj, Nk, arr_stride_i, arr_stride_j   
    );

}


template void centralDiffCollStencil7DimI<double>(double*, const double*, const double*, const double*, const double*, const double*, const double*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
template void centralDiffCollStencil7DimI< float>( float*, const  float*, const  float*, const  float*, const  float*, const  float*, const  float*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
template void centralDiffCollStencil7DimJ<double>(double*, const double*, const double*, const double*, const double*, const double*, const double*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
template void centralDiffCollStencil7DimJ< float>( float*, const  float*, const  float*, const  float*, const  float*, const  float*, const  float*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
template void centralDiffCollStencil7DimK<double>(double*, const double*, const double*, const double*, const double*, const double*, const double*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
template void centralDiffCollStencil7DimK< float>( float*, const  float*, const  float*, const  float*, const  float*, const  float*, const  float*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);

} // stencil
} //namespace cmpk
