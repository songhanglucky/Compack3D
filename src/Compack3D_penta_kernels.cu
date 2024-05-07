
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <stdexcept>
#include <cassert>
#include <cstdio>
#include "Compack3D_penta_kernels.cuh"
#include "Compack3D_utils_kernels.cuh"

namespace cmpk {
namespace penta {

/*!
 * Solve the local system using PCR along i-dimension
 * \param [in, out] x                   right-hand side of the system and the solution as the output
 * \param [in]      fact_prev_2         factorization coefficients of the second previous row
 * \param [in]      fact_prev_1         factorization coefficients of the first previous row
 * \param [in]      fact_curr           factorization coefficients of the current row
 * \param [in]      fact_next_1         factorization coefficients of the first next row
 * \param [in]      fact_next_2         factorization coefficients of the second next row
 * \param [in]      max_sub_sys_size    maximum sub-system size that can be solved with one memory load
 * \param [in]      Ni                  total number of entries in the i-dimension
 * \param [in]      Nj                  total number of entries in the j-dimension
 * \param [in]      Nk                  total number of entries in the k-dimension
 * \param [in]      arr_stride_i        array access stride in i-dimension, from i to i+1
 * \param [in]      arr_stride_j        array access stride in j-dimension, from j to j+1
 * \note array access in k-dimension is contiguous
 */
template<typename RealType, unsigned int NUM_THREADS_I, unsigned int NUM_THREADS_K> __global__
void kernelLocalSolPentaPCRDimI(
              RealType* __restrict__           x,
        const RealType* __restrict__ fact_prev_2,
        const RealType* __restrict__ fact_prev_1,
        const RealType* __restrict__ fact_curr  ,
        const RealType* __restrict__ fact_next_1,
        const RealType* __restrict__ fact_next_2,
        const unsigned int      max_sub_sys_size,
        const unsigned int Ni, const unsigned int Nj, const unsigned int Nk,
        const unsigned int arr_stride_i, const unsigned int arr_stride_j
) {
    #define IDX_K            (blockIdx.x * NUM_THREADS_K + (threadIdx.x % NUM_THREADS_K))
    #define IDX_J             blockIdx.y
    #define TID_K            (threadIdx.x % NUM_THREADS_K)
    #define TID_I            (threadIdx.x / NUM_THREADS_K)
    #define NUM_SYS          Ni
    #define SMEM_IDX(I, K)     ((I) * NUM_THREADS_K + (K))
    #define ARR_IDX(I, J, K) ((I) * arr_stride_i + (J) * arr_stride_j + (K))

    assert(blockIdx.z == 0);
    assert((NUM_THREADS_I * NUM_THREADS_K) == blockDim.x);

    extern __shared__ char smem_general[];
    RealType* smem_realtype = reinterpret_cast<RealType*>(smem_general);
    RealType* smem_buf_0 = &smem_realtype[0];
    RealType* smem_buf_1 = &smem_realtype[NUM_SYS * NUM_THREADS_K];
    RealType* x_read  = smem_buf_0;
    RealType* x_write = smem_buf_1;

    const bool VALID_JK = (IDX_J < Nj) && (IDX_K < Nk);

    // Load data to shared memory from global memory
    unsigned int idx_i = TID_I;
    while (idx_i < Ni) {
        if (VALID_JK) x_read[SMEM_IDX(idx_i, TID_K)] = x[ARR_IDX(idx_i, IDX_J, IDX_K)];
        idx_i += NUM_THREADS_I;
    }
    __syncthreads();

    // Parallel cyclic-reduction on shared memory
    int stride = 1;
    int level  = 0;
    while (stride < NUM_SYS) {
        x_read  = (level & 0b1) ? smem_buf_1 : smem_buf_0;
        x_write = (level & 0b1) ? smem_buf_0 : smem_buf_1;

        idx_i = TID_I;
        while (idx_i < Ni) {
            if (VALID_JK) {
                const int i_sub = idx_i >> level;
                const int n_sub = (NUM_SYS + stride - 1 - (idx_i & (stride-1))) >> level;
                const int row_label = 0b00100
                                    + ((i_sub >        2 ) << 4)
                                    + ((i_sub >        0 ) << 3)
                                    + ((i_sub < (n_sub-1)) << 1)
                                    + ( i_sub < (n_sub-3)      );
                const unsigned int idx_fact = level * NUM_SYS + locFactIdx<unsigned int>(idx_i, NUM_SYS, stride, max_sub_sys_size);
                const unsigned int SMEM_IDX_CURR = SMEM_IDX(idx_i, TID_K);
                x_write[SMEM_IDX_CURR] = fact_curr[idx_fact] * x_read[SMEM_IDX_CURR];
                if (row_label & 0b10000) x_write[SMEM_IDX_CURR] += fact_prev_2[idx_fact] * x_read[SMEM_IDX(idx_i - (stride << 1), TID_K)];
                if (row_label & 0b01000) x_write[SMEM_IDX_CURR] += fact_prev_1[idx_fact] * x_read[SMEM_IDX(idx_i -  stride      , TID_K)];
                if (row_label & 0b00010) x_write[SMEM_IDX_CURR] += fact_next_1[idx_fact] * x_read[SMEM_IDX(idx_i +  stride      , TID_K)];
                if (row_label & 0b00001) x_write[SMEM_IDX_CURR] += fact_next_2[idx_fact] * x_read[SMEM_IDX(idx_i + (stride << 1), TID_K)];
            }
            idx_i += NUM_THREADS_I;
        }
        __syncthreads();
        stride <<= 1;
        level ++;
    }
     
    // Write result to global memeory
    idx_i = TID_I;
    while (idx_i < Ni) {
        if (VALID_JK) x[ARR_IDX(idx_i, IDX_J, IDX_K)] = x_write[SMEM_IDX(idx_i, TID_K)];
        idx_i += NUM_THREADS_I;
    }
     
    #undef IDX_K
    #undef IDX_J
    #undef TID_K
    #undef TID_I
    #undef NUM_SYS
    #undef SMEM_IDX
    #undef ARR_IDX
}



/*!
 * Launch "kernelLocalSolPentaPCRDimI<RealType, unsigned int, unsigned int>"
 * \param [in, out] x                   right-hand side of the system and the solution as the output
 * \param [in]      fact_prev_2         factorization coefficients of the second previous row
 * \param [in]      fact_prev_1         factorization coefficients of the first previous row
 * \param [in]      fact_curr           factorization coefficients of the current row
 * \param [in]      fact_next_1         factorization coefficients of the first next row
 * \param [in]      fact_next_2         factorization coefficients of the second next row
 * \param [in]      max_sub_sys_size    maximum sub-system size that can be solved with one memory load
 * \param [in]      Ni                  total number of entries in the i-dimension
 * \param [in]      Nj                  total number of entries in the j-dimension
 * \param [in]      Nk                  total number of entries in the k-dimension
 * \param [in]      arr_stride_i        array access stride in i-dimension, from i to i+1
 * \param [in]      arr_stride_j        array access stride in j-dimension, from j to j+1
 * \note array access in k-dimension is contiguous
 */
template<typename RealType>
void localSolPentaPCRDimI(
              RealType*           x,
        const RealType* fact_prev_2,
        const RealType* fact_prev_1,
        const RealType* fact_curr  ,
        const RealType* fact_next_1,
        const RealType* fact_next_2,
        const unsigned int  max_sub_sys_size,
        const unsigned int Ni, unsigned const int Nj, const unsigned int Nk,
        const unsigned int arr_stride_i, const unsigned int arr_stride_j
) {
    if (Ni <= 256) { // limited by the size of shared memory (16kB)
        constexpr unsigned int BLOCK_SIZE    = 512;
        constexpr unsigned int DATA_SEG      = 64;
        constexpr unsigned int NUM_THREADS_K = DATA_SEG / sizeof(RealType);
        constexpr unsigned int NUM_THREADS_I = BLOCK_SIZE / NUM_THREADS_K;
        const unsigned int smem_size = 2 * DATA_SEG * Ni;
        dim3 grid_size  = dim3((Nk + NUM_THREADS_K - 1) / NUM_THREADS_K, Nj, 1);
        //cudaFuncSetAttribute(kernelLocalSolPentaPCRDimI<RealType, NUM_THREADS_I, NUM_THREADS_K>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        kernelLocalSolPentaPCRDimI<RealType, NUM_THREADS_I, NUM_THREADS_K><<<grid_size, BLOCK_SIZE, smem_size>>>
            (x, fact_prev_2, fact_prev_1, fact_curr, fact_next_1, fact_next_2, max_sub_sys_size, Ni, Nj, Nk, arr_stride_i, arr_stride_j);
    } else if (Ni <= 512) { // limited by the size of shared memory (16kB)
        constexpr unsigned int BLOCK_SIZE    = 1024;
        constexpr unsigned int DATA_SEG      = 32;
        constexpr unsigned int NUM_THREADS_K = DATA_SEG / sizeof(RealType);
        constexpr unsigned int NUM_THREADS_I = BLOCK_SIZE / NUM_THREADS_K;
        const unsigned int smem_size = 2 * DATA_SEG * Ni;
        dim3 grid_size  = dim3((Nk + NUM_THREADS_K - 1) / NUM_THREADS_K, Nj, 1);
        //cudaFuncSetAttribute(kernelLocalSolPentaPCRDimI<RealType, NUM_THREADS_I, NUM_THREADS_K>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        kernelLocalSolPentaPCRDimI<RealType, NUM_THREADS_I, NUM_THREADS_K><<<grid_size, BLOCK_SIZE, smem_size>>>
            (x, fact_prev_2, fact_prev_1, fact_curr, fact_next_1, fact_next_2, max_sub_sys_size, Ni, Nj, Nk, arr_stride_i, arr_stride_j);
    } else {
        constexpr unsigned int BLOCK_SIZE    = 1024;
        constexpr unsigned int DATA_SEG      = 8;
        constexpr unsigned int NUM_THREADS_K = DATA_SEG / sizeof(RealType);
        constexpr unsigned int NUM_THREADS_I = BLOCK_SIZE / NUM_THREADS_K;
        const unsigned int smem_size = 2 * DATA_SEG * Ni;
        dim3 grid_size  = dim3((Nk + NUM_THREADS_K - 1) / NUM_THREADS_K, Nj, 1);
        //cudaFuncSetAttribute(kernelLocalSolPentaPCRDimI<RealType, NUM_THREADS_I, NUM_THREADS_K>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        kernelLocalSolPentaPCRDimI<RealType, NUM_THREADS_I, NUM_THREADS_K><<<grid_size, BLOCK_SIZE, smem_size>>>
            (x, fact_prev_2, fact_prev_1, fact_curr, fact_next_1, fact_next_2, max_sub_sys_size, Ni, Nj, Nk, arr_stride_i, arr_stride_j);
    }
}



/*!
 * Solve the local system using PCR along j-dimension
 * \param [in, out] x                   right-hand side of the system and the solution as the output
 * \param [in]      fact_prev_2         factorization coefficients of the second previous row
 * \param [in]      fact_prev_1         factorization coefficients of the first previous row
 * \param [in]      fact_curr           factorization coefficients of the current row
 * \param [in]      fact_next_1         factorization coefficients of the first next row
 * \param [in]      fact_next_2         factorization coefficients of the second next row
 * \param [in]      max_sub_sys_size    maximum sub-system size that can be solved with one memory load
 * \param [in]      Ni                  total number of entries in the i-dimension
 * \param [in]      Nj                  total number of entries in the j-dimension
 * \param [in]      Nk                  total number of entries in the k-dimension
 * \param [in]      arr_stride_i        array access stride in i-dimension, from i to i+1
 * \param [in]      arr_stride_j        array access stride in j-dimension, from j to j+1
 * \note array access in k-dimension is contiguous
 */
template<typename RealType, unsigned int NUM_THREADS_J, unsigned int NUM_THREADS_K> __global__
void kernelLocalSolPentaPCRDimJ(
              RealType* __restrict__           x,
        const RealType* __restrict__ fact_prev_2,
        const RealType* __restrict__ fact_prev_1,
        const RealType* __restrict__ fact_curr  ,
        const RealType* __restrict__ fact_next_1,
        const RealType* __restrict__ fact_next_2,
        const unsigned int      max_sub_sys_size,
        const unsigned int Ni, const unsigned int Nj, const unsigned int Nk,
        const unsigned int arr_stride_i, const unsigned int arr_stride_j
) {
    #define IDX_K            (blockIdx.x * NUM_THREADS_K + (threadIdx.x % NUM_THREADS_K))
    #define IDX_I             blockIdx.z
    #define TID_K            (threadIdx.x % NUM_THREADS_K)
    #define TID_J            (threadIdx.x / NUM_THREADS_K)
    #define NUM_SYS           Nj
    #define SMEM_IDX(J, K)   ((J) * NUM_THREADS_K + (K))
    #define ARR_IDX(I, J, K) ((I) * arr_stride_i + (J) * arr_stride_j + (K))

    assert(blockIdx.y == 0);
    assert((NUM_THREADS_J * NUM_THREADS_K) == blockDim.x);

    extern __shared__ char smem_general[];
    RealType* smem_realtype = reinterpret_cast<RealType*>(smem_general);
    RealType* smem_buf_0 = &smem_realtype[0];
    RealType* smem_buf_1 = &smem_realtype[NUM_SYS * NUM_THREADS_K];
    RealType* x_read  = smem_buf_0;
    RealType* x_write = smem_buf_1;

    const bool VALID_IK = (IDX_K < Nk) && (IDX_I < Ni);

    // Load data to shared memory from global memory
    unsigned int idx_j = TID_J;
    while (idx_j < Nj) {
        if (VALID_IK) x_read[SMEM_IDX(idx_j, TID_K)] = x[ARR_IDX(IDX_I, idx_j, IDX_K)];
        idx_j += NUM_THREADS_J;
    }
    __syncthreads();

    // Parallel cyclic-reduction on shared memory
    int stride = 1;
    int level  = 0;
    while (stride < NUM_SYS) {
        x_read  = (level & 0b1) ? smem_buf_1 : smem_buf_0;
        x_write = (level & 0b1) ? smem_buf_0 : smem_buf_1;

        idx_j = TID_J;
        while (idx_j < Nj) {
            if (VALID_IK) {
                const int j_sub = idx_j >> level;
                const int n_sub = (NUM_SYS + stride - 1 - (idx_j & (stride-1))) >> level;
                const int row_label = 0b00100
                                    + ((j_sub >        2 ) << 4)
                                    + ((j_sub >        0 ) << 3)
                                    + ((j_sub < (n_sub-1)) << 1)
                                    + ( j_sub < (n_sub-3)      );
                const unsigned int idx_fact = level * NUM_SYS + locFactIdx<unsigned int>(idx_j, NUM_SYS, stride, max_sub_sys_size);
                const unsigned int SMEM_IDX_CURR = SMEM_IDX(idx_j, TID_K);
                x_write[SMEM_IDX_CURR] = fact_curr[idx_fact] * x_read[SMEM_IDX_CURR];
                if (row_label & 0b10000) x_write[SMEM_IDX_CURR] += fact_prev_2[idx_fact] * x_read[SMEM_IDX(idx_j - (stride << 1), TID_K)];
                if (row_label & 0b01000) x_write[SMEM_IDX_CURR] += fact_prev_1[idx_fact] * x_read[SMEM_IDX(idx_j -  stride      , TID_K)];
                if (row_label & 0b00010) x_write[SMEM_IDX_CURR] += fact_next_1[idx_fact] * x_read[SMEM_IDX(idx_j +  stride      , TID_K)];
                if (row_label & 0b00001) x_write[SMEM_IDX_CURR] += fact_next_2[idx_fact] * x_read[SMEM_IDX(idx_j + (stride << 1), TID_K)];
            }
            idx_j += NUM_THREADS_J;
        }
        __syncthreads();
        stride <<= 1;
        level ++;
    }
     
    // Write result to global memeory
    idx_j = TID_J;
    while (idx_j < Nj) {
        if (VALID_IK) x[ARR_IDX(IDX_I, idx_j, IDX_K)] = x_write[SMEM_IDX(idx_j, TID_K)];
        idx_j += NUM_THREADS_J;
    }
     
    #undef IDX_K
    #undef IDX_I
    #undef TID_K
    #undef TID_J
    #undef NUM_SYS
    #undef SMEM_IDX
    #undef ARR_IDX
}



/*!
 * Launch "kernelLocalSolPentaPCRDimJ<RealType, unsigned int, unsigned int>"
 * \param [in, out] x                   right-hand side of the system and the solution as the output
 * \param [in]      fact_prev_2         factorization coefficients of the second previous row
 * \param [in]      fact_prev_1         factorization coefficients of the first previous row
 * \param [in]      fact_curr           factorization coefficients of the current row
 * \param [in]      fact_next_1         factorization coefficients of the first next row
 * \param [in]      fact_next_2         factorization coefficients of the second next row
 * \param [in]      max_sub_sys_size    maximum sub-system size that can be solved with one memory load
 * \param [in]      Ni                  total number of entries in the i-dimension
 * \param [in]      Nj                  total number of entries in the j-dimension
 * \param [in]      Nk                  total number of entries in the k-dimension
 * \param [in]      arr_stride_i        array access stride in i-dimension, from i to i+1
 * \param [in]      arr_stride_j        array access stride in j-dimension, from j to j+1
 * \note array access in k-dimension is contiguous
 */
template<typename RealType>
void localSolPentaPCRDimJ(
              RealType*           x,
        const RealType* fact_prev_2,
        const RealType* fact_prev_1,
        const RealType* fact_curr  ,
        const RealType* fact_next_1,
        const RealType* fact_next_2,
        const unsigned int  max_sub_sys_size,
        const unsigned int Ni, unsigned const int Nj, const unsigned int Nk,
        const unsigned int arr_stride_i, const unsigned int arr_stride_j
) {
    if (Ni <= 256) { // limited by the size of shared memory (16kB)
        constexpr unsigned int BLOCK_SIZE    = 512;
        constexpr unsigned int DATA_SEG      = 64;
        constexpr unsigned int NUM_THREADS_K = DATA_SEG / sizeof(RealType);
        constexpr unsigned int NUM_THREADS_J = BLOCK_SIZE / NUM_THREADS_K;
        const unsigned int smem_size = 2 * DATA_SEG * Nj;
        dim3 grid_size  = {(Nk + NUM_THREADS_K - 1) / NUM_THREADS_K, 1, Ni};
        //cudaFuncSetAttribute(kernelLocalSolPentaPCRDimJ<RealType, NUM_THREADS_J, NUM_THREADS_K>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        kernelLocalSolPentaPCRDimJ<RealType, NUM_THREADS_J, NUM_THREADS_K><<<grid_size, BLOCK_SIZE, smem_size>>>
            (x, fact_prev_2, fact_prev_1, fact_curr, fact_next_1, fact_next_2, max_sub_sys_size, Ni, Nj, Nk, arr_stride_i, arr_stride_j);
    } else if (Ni <= 512) {
        constexpr unsigned int BLOCK_SIZE    = 1024;
        constexpr unsigned int DATA_SEG      = 32;
        constexpr unsigned int NUM_THREADS_K = DATA_SEG / sizeof(RealType);
        constexpr unsigned int NUM_THREADS_J = BLOCK_SIZE / NUM_THREADS_K;
        const unsigned int smem_size = 2 * DATA_SEG * Nj;
        dim3 grid_size  = {(Nk + NUM_THREADS_K - 1) / NUM_THREADS_K, 1, Ni};
        //cudaFuncSetAttribute(kernelLocalSolPentaPCRDimJ<RealType, NUM_THREADS_J, NUM_THREADS_K>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        kernelLocalSolPentaPCRDimJ<RealType, NUM_THREADS_J, NUM_THREADS_K><<<grid_size, BLOCK_SIZE, smem_size>>>
            (x, fact_prev_2, fact_prev_1, fact_curr, fact_next_1, fact_next_2, max_sub_sys_size, Ni, Nj, Nk, arr_stride_i, arr_stride_j);
    } else {
        constexpr unsigned int BLOCK_SIZE    = 1024;
        constexpr unsigned int DATA_SEG      = 8;
        constexpr unsigned int NUM_THREADS_K = DATA_SEG / sizeof(RealType);
        constexpr unsigned int NUM_THREADS_J = BLOCK_SIZE / NUM_THREADS_K;
        const unsigned int smem_size = 2 * DATA_SEG * Nj;
        dim3 grid_size  = {(Nk + NUM_THREADS_K - 1) / NUM_THREADS_K, 1, Ni};
        //cudaFuncSetAttribute(kernelLocalSolPentaPCRDimJ<RealType, NUM_THREADS_J, NUM_THREADS_K>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        kernelLocalSolPentaPCRDimJ<RealType, NUM_THREADS_J, NUM_THREADS_K><<<grid_size, BLOCK_SIZE, smem_size>>>
            (x, fact_prev_2, fact_prev_1, fact_curr, fact_next_1, fact_next_2, max_sub_sys_size, Ni, Nj, Nk, arr_stride_i, arr_stride_j);
    }
}



/*!
 * Solve the local system using PCR along k-dimension
 * \param [in, out] x                   right-hand side of the system and the solution as the output
 * \param [in]      fact_prev_2         factorization coefficients of the second previous row
 * \param [in]      fact_prev_1         factorization coefficients of the first previous row
 * \param [in]      fact_curr           factorization coefficients of the current row
 * \param [in]      fact_next_1         factorization coefficients of the first next row
 * \param [in]      fact_next_2         factorization coefficients of the second next row
 * \param [in]      max_sub_sys_size    maximum sub-system size that can be solved with one memory load
 * \param [in]      Ni                  total number of entries in the i-dimension
 * \param [in]      Nj                  total number of entries in the j-dimension
 * \param [in]      Nk                  total number of entries in the k-dimension
 * \param [in]      arr_stride_i        array access stride in i-dimension, from i to i+1
 * \param [in]      arr_stride_j        array access stride in j-dimension, from j to j+1
 * \note array access in k-dimension is contiguous
 */
template<typename RealType, unsigned int NUM_THREADS_K, unsigned int NUM_THREADS_J> __global__
void kernelLocalSolPentaPCRDimK(
              RealType* __restrict__           x,
        const RealType* __restrict__ fact_prev_2,
        const RealType* __restrict__ fact_prev_1,
        const RealType* __restrict__ fact_curr  ,
        const RealType* __restrict__ fact_next_1,
        const RealType* __restrict__ fact_next_2,
        const unsigned int      max_sub_sys_size,
        const unsigned int Ni, const unsigned int Nj, const unsigned int Nk,
        const unsigned int arr_stride_i, const unsigned int arr_stride_j
) {
    #define IDX_I             blockIdx.z
    #define IDX_J            ((blockIdx.y * NUM_THREADS_J) + (threadIdx.x / NUM_THREADS_K))
    #define TID_J            (threadIdx.x / NUM_THREADS_K) 
    #define TID_K            (threadIdx.x % NUM_THREADS_K) 
    #define NUM_SYS           Nk
    #define SMEM_IDX(K, J)   ((K) + (J) * Nk)
    #define ARR_IDX(I, J, K) ((I) * arr_stride_i + (J) * arr_stride_j + (K))

    assert(blockIdx.x == 0);
    assert((NUM_THREADS_K * NUM_THREADS_J) == blockDim.x);

    extern __shared__ char smem_general[];
    RealType* smem_realtype = reinterpret_cast<RealType*>(smem_general);
    RealType* smem_buf_0 = &smem_realtype[0];
    RealType* smem_buf_1 = &smem_realtype[NUM_SYS * NUM_THREADS_J];
    RealType* x_read  = smem_buf_0;
    RealType* x_write = smem_buf_1;
    const bool VALID_IJ = (IDX_I < Ni) && (IDX_J < Nj);

    // Load data to shared memory from global memory
    unsigned int idx_k = TID_K;
    while (idx_k < Nk) {
        if (VALID_IJ) x_read[SMEM_IDX(idx_k, TID_J)] = x[ARR_IDX(IDX_I, IDX_J, idx_k)];
        idx_k += NUM_THREADS_K;
    }
    __syncthreads();

    // Parallel cyclic-reduction on shared memory
    int stride = 1;
    int level  = 0;
    while (stride < NUM_SYS) {
        x_read  = (level & 0b1) ? smem_buf_1 : smem_buf_0;
        x_write = (level & 0b1) ? smem_buf_0 : smem_buf_1;

        idx_k = TID_K;
        while (idx_k < Nk) {
            if (VALID_IJ) {
                const int k_sub = idx_k >> level;
                const int n_sub = (NUM_SYS + stride - 1 - (idx_k & (stride-1))) >> level;
                const int row_label = 0b00100
                                    + ((k_sub >        2 ) << 4)
                                    + ((k_sub >        0 ) << 3)
                                    + ((k_sub < (n_sub-1)) << 1)
                                    + ( k_sub < (n_sub-3)      );
                const unsigned int idx_fact = level * NUM_SYS + locFactIdx<unsigned int>(idx_k, NUM_SYS, stride, max_sub_sys_size);
                const unsigned int SMEM_IDX_CURR = SMEM_IDX(idx_k, TID_J);
                x_write[SMEM_IDX_CURR] = fact_curr[idx_fact] * x_read[SMEM_IDX_CURR];
                if (row_label & 0b10000) x_write[SMEM_IDX_CURR] += fact_prev_2[idx_fact] * x_read[SMEM_IDX(idx_k - (stride << 1), TID_J)];
                if (row_label & 0b01000) x_write[SMEM_IDX_CURR] += fact_prev_1[idx_fact] * x_read[SMEM_IDX(idx_k -  stride      , TID_J)];
                if (row_label & 0b00010) x_write[SMEM_IDX_CURR] += fact_next_1[idx_fact] * x_read[SMEM_IDX(idx_k +  stride      , TID_J)];
                if (row_label & 0b00001) x_write[SMEM_IDX_CURR] += fact_next_2[idx_fact] * x_read[SMEM_IDX(idx_k + (stride << 1), TID_J)];
            }
            idx_k += NUM_THREADS_K;
        }
        __syncthreads();
        stride <<= 1;
        level ++;
    }
     
    // Write result to global memeory
    idx_k = TID_K;
    while (idx_k < Nk) {
        if (VALID_IJ) x[ARR_IDX(IDX_I, IDX_J, idx_k)] = x_write[SMEM_IDX(idx_k, TID_J)];
        idx_k += NUM_THREADS_K;
    }
     
    #undef IDX_I
    #undef IDX_J
    #undef TID_J
    #undef TID_K
    #undef NUM_SYS
    #undef SMEM_IDX
    #undef ARR_IDX
}



/*!
 * Launch "kernelLocalSolPentaPCRDimK<RealType, unsigned int, unsigned int>"
 * \param [in, out] x                   right-hand side of the system and the solution as the output
 * \param [in]      fact_prev_2         factorization coefficients of the second previous row
 * \param [in]      fact_prev_1         factorization coefficients of the first previous row
 * \param [in]      fact_curr           factorization coefficients of the current row
 * \param [in]      fact_next_1         factorization coefficients of the first next row
 * \param [in]      fact_next_2         factorization coefficients of the second next row
 * \param [in]      max_sub_sys_size    maximum sub-system size that can be solved with one memory load
 * \param [in]      Ni                  total number of entries in the i-dimension
 * \param [in]      Nj                  total number of entries in the j-dimension
 * \param [in]      Nk                  total number of entries in the k-dimension
 * \param [in]      arr_stride_i        array access stride in i-dimension, from i to i+1
 * \param [in]      arr_stride_j        array access stride in j-dimension, from j to j+1
 * \note array access in k-dimension is contiguous
 */
template<typename RealType>
void localSolPentaPCRDimK(
              RealType*           x,
        const RealType* fact_prev_2,
        const RealType* fact_prev_1,
        const RealType* fact_curr  ,
        const RealType* fact_next_1,
        const RealType* fact_next_2,
        const unsigned int  max_sub_sys_size,
        const unsigned int Ni, unsigned const int Nj, const unsigned int Nk,
        const unsigned int arr_stride_i, const unsigned int arr_stride_j
) {
    if (Nk <= 256) {
        constexpr unsigned int NUM_THREADS_K = 128;
        constexpr unsigned int NUM_THREADS_J =  2 * sizeof(double) / sizeof(RealType); 
        constexpr unsigned int BLOCK_SIZE    = NUM_THREADS_K * NUM_THREADS_J;
        const unsigned int smem_size = 2 * Nk * NUM_THREADS_J * sizeof(RealType);
        dim3 grid_size  = dim3(1, (Nj + NUM_THREADS_J - 1) / NUM_THREADS_J, Ni);
        //cudaFuncSetAttribute(kernelLocalSolPentaPCRDimK<RealType, NUM_THREADS_K, NUM_THREADS_J>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        kernelLocalSolPentaPCRDimK<RealType, NUM_THREADS_K, NUM_THREADS_J><<<grid_size, BLOCK_SIZE, smem_size>>>
            (x, fact_prev_2, fact_prev_1, fact_curr, fact_next_1, fact_next_2, max_sub_sys_size, Ni, Nj, Nk, arr_stride_i, arr_stride_j);
    } else if (Nk <= 512) {
        constexpr unsigned int NUM_THREADS_K = 256;
        constexpr unsigned int NUM_THREADS_J = 2 * sizeof(double) / sizeof(RealType); 
        constexpr unsigned int BLOCK_SIZE    = NUM_THREADS_K * NUM_THREADS_J;
        const unsigned int smem_size = 2 * Nk * NUM_THREADS_J * sizeof(RealType);
        dim3 grid_size  = dim3(1, (Nj + NUM_THREADS_J - 1) / NUM_THREADS_J, Ni);
        //cudaFuncSetAttribute(kernelLocalSolPentaPCRDimK<RealType, NUM_THREADS_K, NUM_THREADS_J>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        kernelLocalSolPentaPCRDimK<RealType, NUM_THREADS_K, NUM_THREADS_J><<<grid_size, BLOCK_SIZE, smem_size>>>
            (x, fact_prev_2, fact_prev_1, fact_curr, fact_next_1, fact_next_2, max_sub_sys_size, Ni, Nj, Nk, arr_stride_i, arr_stride_j);
    } else {
        constexpr unsigned int NUM_THREADS_K = 1024;
        constexpr unsigned int NUM_THREADS_J = sizeof(double) / sizeof(RealType); 
        constexpr unsigned int BLOCK_SIZE    = NUM_THREADS_K * NUM_THREADS_J;
        const unsigned int smem_size = 2 * Nk * NUM_THREADS_J * sizeof(RealType);
        dim3 grid_size  = dim3(1, (Nj + NUM_THREADS_J - 1) / NUM_THREADS_J, Ni);
        //cudaFuncSetAttribute(kernelLocalSolPentaPCRDimK<RealType, NUM_THREADS_K, NUM_THREADS_J>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        kernelLocalSolPentaPCRDimK<RealType, NUM_THREADS_K, NUM_THREADS_J><<<grid_size, BLOCK_SIZE, smem_size>>>
            (x, fact_prev_2, fact_prev_1, fact_curr, fact_next_1, fact_next_2, max_sub_sys_size, Ni, Nj, Nk, arr_stride_i, arr_stride_j);
    }
}



/*!
 * Reduce the current distributed block using blocks from both sides
 * \tparam RealType                 real-value type of flattened local block 
 * \tparam RealTypeComm             real-value type of communicated neighboring block
 * \param [in, out] x_curr          local block as input and eliminated local block as output
 * \param [in]      x_prev_buf      communicated neighboring block from the previous (strided) rank
 * \param [in]      x_next_buf      communicated neighboring block from the next (strided) rank
 * \param [in]      fact_curr       distributed factorization coefficients of the current block
 * \param [in]      fact_prev       distributed factorization coefficients of the previous block
 * \param [in]      fact_next       distributed factorization coefficients of the next block
 * \param [in]      N_batch         number of entries in the non-solve dimension
 * \note solution is stored in column-major with size 2 x N_batch
 */
template<typename RealType, typename RealTypeComm = RealType> __global__
void kernelReduceCurrBlockSymm(
              RealType*     __restrict__ x_curr,
        const RealTypeComm* __restrict__ x_prev_buf,
        const RealTypeComm* __restrict__ x_next_buf,
        const RealType*     __restrict__ fact_curr,
        const RealType*     __restrict__ fact_prev,
        const RealType*     __restrict__ fact_next,
        const unsigned int N_batch)
{
    // The index of the system is (i,j) and the solution is in column-major
    #define IDX_J            (blockIdx.x * blockDim.x + threadIdx.x)
    #define IDX_I                                       threadIdx.y
    #define TID_J                                       threadIdx.x
    #define TID_I                                       threadIdx.y
    #define NUM_THREADS       blockDim.x * blockDim.y
    #define  SM_IDX(I, J)   ((I) * blockDim.x + (J))
    #define ARR_IDX(I, J)   ((I) * N_batch    + (J))

    assert( blockIdx.y  == 0);
    assert( blockDim.y  == 2);
    const bool VALID_ENTRY = IDX_J < N_batch;
    extern __shared__ char sm_general[];
    RealType*     sm_curr = reinterpret_cast<RealType*    >(&sm_general[0]);
    RealTypeComm* sm_prev = reinterpret_cast<RealTypeComm*>(&sm_general[NUM_THREADS *  sizeof(RealType)]);
    RealTypeComm* sm_next = reinterpret_cast<RealTypeComm*>(&sm_general[NUM_THREADS * (sizeof(RealType) + sizeof(RealTypeComm))]);

    if (VALID_ENTRY) {
        const unsigned int  sm_ij =  SM_IDX(TID_I, TID_J);
        const unsigned int arr_ij = ARR_IDX(IDX_I, IDX_J);
        sm_curr[sm_ij] = x_curr    [arr_ij];
        sm_prev[sm_ij] = x_prev_buf[arr_ij];
        sm_next[sm_ij] = x_next_buf[arr_ij];
    }

    __syncthreads();
    if (VALID_ENTRY) {
        const unsigned char fact_idx_0 = (TID_I << 1);
        const unsigned char fact_idx_1 = fact_idx_0 + 1;
        const unsigned int    sm_idx_0 = SM_IDX(0, TID_J);
        const unsigned int    sm_idx_1 = SM_IDX(1, TID_J);
        x_curr[ARR_IDX(IDX_I, IDX_J)]  = fact_curr[fact_idx_0] * sm_curr[sm_idx_0] + fact_curr[fact_idx_1] * sm_curr[sm_idx_1]
                                       + fact_prev[fact_idx_0] * sm_prev[sm_idx_0] + fact_prev[fact_idx_1] * sm_prev[sm_idx_1]
                                       + fact_next[fact_idx_0] * sm_next[sm_idx_0] + fact_next[fact_idx_1] * sm_next[sm_idx_1];
    }

    #undef IDX_J
    #undef IDX_I
    #undef TID_J
    #undef TID_I
    #undef NUM_THREADS
    #undef  SM_IDX
    #undef ARR_IDX
}



/*!
 * Launch kernel "kernelReduceCurrBlockSymm"
 * \tparam RealType                 real-value type of flattened local block 
 * \tparam RealTypeComm             real-value type of communicated neighboring block
 * \param [in, out] x_curr          local block as input and eliminated local block as output
 * \param [in]      x_prev_buf      communicated neighboring block from the previous (strided) rank
 * \param [in]      x_next_buf      communicated neighboring block from the next (strided) rank
 * \param [in]      fact_curr       distributed factorization coefficients of the current block
 * \param [in]      fact_prev       distributed factorization coefficients of the previous block
 * \param [in]      fact_next       distributed factorization coefficients of the next block
 * \param [in]      N_batch         number of entries in the non-solve dimension
 * \note solution is stored in column-major with size 2 x N_batch
 */
template<typename RealType, typename RealTypeComm = RealType>
void reduceCurrBlockSymm(
              RealType*     x_curr,
        const RealTypeComm* x_prev_buf,
        const RealTypeComm* x_next_buf,
        const RealType*     fact_curr,
        const RealType*     fact_prev,
        const RealType*     fact_next,
        const unsigned int N_batch)
{
    unsigned int num_threads_j = 256;
    while (num_threads_j > N_batch) num_threads_j >>= 1;
    dim3 block_size = dim3(num_threads_j, 2);
    dim3  grid_size = dim3((N_batch + num_threads_j - 1) / num_threads_j, 1);
    unsigned int sm_size = 2 * num_threads_j * (sizeof(RealType) + 2 * sizeof(RealTypeComm));
    kernelReduceCurrBlockSymm<RealType, RealTypeComm><<<grid_size, block_size, sm_size>>>(x_curr, x_prev_buf, x_next_buf, fact_curr, fact_prev, fact_next, N_batch);
}



/*!
 * Reduce the current distributed block using blocks from one side
 * \tparam RealType                 real-value type of flattened local block 
 * \tparam RealTypeComm             real-value type of communicated neighboring block
 * \param [in, out] x_cur           local block as input and eliminated local block as output
 * \param [in]      x_nbr_buf       communicated neighboring block from the previous (strided) rank
 * \param [in]      fact_cur        distributed factorization coefficients of the current block
 * \param [in]      fact_nbr        distributed factorization coefficients of the previous block
 * \param [in]      N_batch         number of entries in the non-solve dimension
 * \note solution is stored in column-major with size 2 x N_batch
 */
template<typename RealType, typename RealTypeComm = RealType> __global__
void kernelReduceCurrBlockOneSide(
              RealType*     __restrict__ x_cur,
        const RealTypeComm* __restrict__ x_nbr_buf,
        const RealType*     __restrict__ fact_cur,
        const RealType*     __restrict__ fact_nbr,
        const unsigned int N_batch)
{
    // The index of the system is (i,j) and the solution is in column-major
    #define IDX_J            (blockIdx.x * blockDim.x + threadIdx.x)
    #define IDX_I                                       threadIdx.y
    #define TID_J                                       threadIdx.x
    #define TID_I                                       threadIdx.y
    #define NUM_THREADS       blockDim.x * blockDim.y
    #define  SM_IDX(I, J)   ((I) * blockDim.x + (J))
    #define ARR_IDX(I, J)   ((I) * N_batch    + (J))

    assert( blockIdx.y  == 0);
    assert( blockDim.y  == 2);
    const bool VALID_ENTRY = IDX_J < N_batch;
    extern __shared__ char sm_general[];
    RealType*     sm_cur = reinterpret_cast<RealType*    >(&sm_general[0]);
    RealTypeComm* sm_nbr = reinterpret_cast<RealTypeComm*>(&sm_general[NUM_THREADS * sizeof(RealType)]);

    if (VALID_ENTRY) {
        const unsigned int  sm_ij =  SM_IDX(TID_I, TID_J);
        const unsigned int arr_ij = ARR_IDX(IDX_I, IDX_J);
        sm_cur[sm_ij] = x_cur    [arr_ij];
        sm_nbr[sm_ij] = x_nbr_buf[arr_ij];
    }

    __syncthreads();
    if (VALID_ENTRY) {
        const unsigned char fact_idx_0 = (TID_I << 1);
        const unsigned char fact_idx_1 = fact_idx_0 + 1;
        const unsigned int    sm_idx_0 = SM_IDX(0, TID_J);
        const unsigned int    sm_idx_1 = SM_IDX(1, TID_J);
        x_cur[ARR_IDX(IDX_I, IDX_J)]  = fact_cur[fact_idx_0] * sm_cur[sm_idx_0] + fact_cur[fact_idx_1] * sm_cur[sm_idx_1]
                                      + fact_nbr[fact_idx_0] * sm_nbr[sm_idx_0] + fact_nbr[fact_idx_1] * sm_nbr[sm_idx_1];
    }

    #undef IDX_J
    #undef IDX_I
    #undef TID_J
    #undef TID_I
    #undef NUM_THREADS
    #undef  SM_IDX
    #undef ARR_IDX
}



/*!
 * Launch kernel "kernelReduceCurrBlockOneSide"
 * \tparam RealType                 real-value type of flattened local block 
 * \tparam RealTypeComm             real-value type of communicated neighboring block
 * \param [in, out] x_cur           local block as input and eliminated local block as output
 * \param [in]      x_nbr_buf       communicated neighboring block from the previous (strided) rank
 * \param [in]      fact_cur        distributed factorization coefficients of the current block
 * \param [in]      fact_nbr        distributed factorization coefficients of the previous block
 * \param [in]      N_batch         number of entries in the non-solve dimension
 * \note solution is stored in column-major with size 2 x N_batch
 */
template<typename RealType, typename RealTypeComm = RealType>
void reduceCurrBlockOneSide(
              RealType*     x_cur,
        const RealTypeComm* x_nbr_buf,
        const RealType*     fact_cur,
        const RealType*     fact_nbr,
        const unsigned int N_batch)
{
    unsigned int num_threads_j = 256;
    while (num_threads_j > N_batch) num_threads_j >>= 1;
    dim3 block_size = dim3(num_threads_j, 2);
    dim3  grid_size = dim3((N_batch + num_threads_j - 1) / num_threads_j, 1);
    unsigned int sm_size = 2 * num_threads_j * (sizeof(RealType) + sizeof(RealTypeComm));
    kernelReduceCurrBlockOneSide<RealType, RealTypeComm><<<grid_size, block_size, sm_size>>>(x_cur, x_nbr_buf, fact_cur, fact_nbr, N_batch);
}



/*!
 * Locally calcualte the right-hand side of the reduced system, b_hat.
 * \param [in, out]    x_tilde        b_tilde as input and b_hat as output (see Eq (21) in Song et al. JCP (2022) 111443)
 * \param [in]         y_prev_tail    The last two rows of y_{i-1} defined in Eq(17) in Song et al. JCP (2022) 111443
 * \param [in]         y_curr_head    The first two rows of y_{i} defined in Eq(17) in Song et al. JCP (2022) 111443
 * \param [in]         Li_tilde_tail  The non-zero entries in Li_tilde stored in row-major (see Fig.4 in Song et al. JCP (2022) 111443)
 * \param [in]         Ui_tilde_head  The non-zero entries in Ui_tilde stored in row-major (see Fig.4 in Song et al. JCP (2022) 111443)
 * \param [in]         N_batch        Number of entries in the non-solve direction
 * \note The non-solve direction of x_tilde, y_prev_tail and y_curr_head are mapped to contiguous memory layout
 */
template<typename RealType> __global__
void kernelCalcReducedSystemRHSLocal(
              RealType* __restrict__ x_tilde,
        const RealType* __restrict__ y_prev_tail,
        const RealType* __restrict__ y_curr_head,
        const RealType* __restrict__ Li_tilde_tail,
        const RealType* __restrict__ Ui_tilde_head,
        const unsigned int N_batch
    )
{
    #define  SM_IDX(I, J)   ((I) * (blockDim.x+1) + (J))
    #define ARR_IDX(I, J)   ((I) * N_batch    + (J))
    assert(blockIdx.y == 0 && blockDim.y == 2);
    const unsigned int      j  = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int arr_idx = ARR_IDX(threadIdx.y, j);
    extern __shared__ char sm_general[];
    RealType* sm_y_prev = reinterpret_cast<RealType*>(&sm_general[0]);
    RealType* sm_y_curr = reinterpret_cast<RealType*>(&sm_general[SM_IDX(2, 0) * sizeof(RealType)]);

    if (j < N_batch) {
        sm_y_prev[SM_IDX(threadIdx.y, threadIdx.x)] = y_prev_tail[arr_idx];
        sm_y_curr[SM_IDX(threadIdx.y, threadIdx.x)] = y_curr_head[arr_idx];
    }

    __syncthreads();

    const unsigned int sm_idx_0 = SM_IDX(0, threadIdx.x);
    const unsigned int sm_idx_1 = SM_IDX(1, threadIdx.x);
    if ((j < N_batch) && (threadIdx.y == 0)) {
        x_tilde[arr_idx] -= Li_tilde_tail[0] * sm_y_prev[sm_idx_0] + Li_tilde_tail[1] * sm_y_prev[sm_idx_1] + Ui_tilde_head[0] * sm_y_curr[sm_idx_0];
    }
    if ((j < N_batch) && (threadIdx.y == 1)) {
        x_tilde[arr_idx] -= Li_tilde_tail[2] * sm_y_prev[sm_idx_1] + Ui_tilde_head[1] * sm_y_curr[sm_idx_0] + Ui_tilde_head[2] * sm_y_curr[sm_idx_1];
    }

    #undef SM_IDX
    #undef ARR_IDX
}



/*!
 * Launch "kernelCalcReducedSystemRHSLocal"
 * \param [in, out]    x_tilde        b_tilde as input and b_hat as output (see Eq (21) in Song et al. JCP (2022) 111443)
 * \param [in]         y_prev_tail    The last two rows of y_{i-1} defined in Eq(17) in Song et al. JCP (2022) 111443
 * \param [in]         y_curr_head    The first two rows of y_{i} defined in Eq(17) in Song et al. JCP (2022) 111443
 * \param [in]         Li_tilde_tail  The non-zero entries in Li_tilde stored in row-major (see Fig.4 in Song et al. JCP (2022) 111443)
 * \param [in]         Ui_tilde_head  The non-zero entries in Ui_tilde stored in row-major (see Fig.4 in Song et al. JCP (2022) 111443)
 * \param [in]         N_batch        Number of entries in the non-solve direction
 * \note The non-solve direction of x_tilde, y_prev_tail and y_curr_head are mapped to contiguous memory layout
 */
template<typename RealType>
void calcReducedSystemRHSLocal(
              RealType* x_tilde,
        const RealType* y_prev_tail,
        const RealType* y_curr_head,
        const RealType* Li_tilde_tail,
        const RealType* Ui_tilde_head,
        const unsigned int N_batch
    )
{
    unsigned int num_threads_x = 128;
    while(num_threads_x > N_batch) num_threads_x >>= 1;
    unsigned int sm_size = 4 * (num_threads_x + 1) * sizeof(RealType);
    dim3 block_size = dim3(num_threads_x, 2);
    dim3  grid_size = dim3((N_batch + num_threads_x - 1) / num_threads_x, 1);
    kernelCalcReducedSystemRHSLocal<RealType><<<grid_size, block_size, sm_size>>>(x_tilde, y_prev_tail, y_curr_head, Li_tilde_tail, Ui_tilde_head, N_batch);
}



/*!
 * Calculate local solution from y_{rank} in i-dimension using x_tilde from current and next partitions (see Eq.(22) in Song et al. JCP (2022) 111443)
 * \param [in, out]    x_loc            y_{rank} as input and x_{rank} as output (see Fig.4 and Eq.(17) in Song et al. JCP (2022) 111443)
 * \param [in]         x_tilde_curr     The solution to the reduced system in the current partition (see Fig.4 in Song et al. JCP (2022) 11143)
 * \param [in]         x_tilde_next     The solution to the reduced system in the next partition (see Fig.4 in Song et al. JCP (2022) 11143)
 * \param [in]         S                Preprocessed coefficients stored in column-major defined in Eq.(15) in Song et al. JCP (2022) 11143
 * \param [in]         R                Preprocessed coefficients stored in column-major defined in Eq.(16) in Song et al. JCP (2022) 11143
 * \param [in]         Ni               Number of entries in x_loc in i-dimension
 * \param [in]         Nj               Number of entries in x_loc in j-dimension
 * \param [in]         Nk               Number of entries in x_loc in k-dimension
 * \param [in]      arr_stride_i        array access stride of x_loc in i-dimension, from i to i+1
 * \param [in]      arr_stride_j        array access stride of x_loc in j-dimension, from j to j+1
 */
template<unsigned int NUM_THREADS_I, unsigned int NUM_THREADS_J, unsigned int NUM_THREADS_K, typename RealType> __global__
void kernelUpdateLocalSolDimI(
              RealType* __restrict__ x_loc,
        const RealType* __restrict__ x_tilde_curr,
        const RealType* __restrict__ x_tilde_next,
        const RealType* __restrict__ S,
        const RealType* __restrict__ R,
        const unsigned int Ni, const unsigned int Nj, const unsigned int Nk,
        const unsigned int arr_stride_i, const unsigned int arr_stride_j
    )
{
    #define TID_I (threadIdx.x / (NUM_THREADS_J * NUM_THREADS_K))
    #define TID_J ((threadIdx.x / NUM_THREADS_K) % NUM_THREADS_J)
    #define TID_K (threadIdx.x % NUM_THREADS_K)
    #define SMEM_IDX(I, J, K) ((I) * (NUM_THREADS_J * NUM_THREADS_K) + (J) * NUM_THREADS_J + (K)) 

    #define IDX_I (blockIdx.z * NUM_THREADS_I + TID_I)
    #define IDX_J (blockIdx.y * NUM_THREADS_J + TID_J)
    #define IDX_K (blockIdx.x * NUM_THREADS_K + TID_K)

    extern __shared__ char smem_general[];
    RealType* smem_x_tilde_curr = reinterpret_cast<RealType*>(&smem_general[0]);
    RealType* smem_x_tilde_next = reinterpret_cast<RealType*>(&smem_general[2 * NUM_THREADS_J * NUM_THREADS_K * sizeof(RealType)]);

    if ((TID_I < 2) && (IDX_J < Nj) && (IDX_K < Nk)) {
        smem_x_tilde_curr[SMEM_IDX(TID_I, TID_J, TID_K)] = x_tilde_curr[TID_I * Nj * Nk + IDX_J * Nk + IDX_K];
        smem_x_tilde_next[SMEM_IDX(TID_I, TID_J, TID_K)] = x_tilde_next[TID_I * Nj * Nk + IDX_J * Nk + IDX_K];
    }
    __syncthreads();

    if ((IDX_I < Ni) && (IDX_J < Nj) && (IDX_K < Nk)) {
        x_loc[IDX_I * arr_stride_i + IDX_J * arr_stride_j + IDX_K]
            -= S[IDX_I] * smem_x_tilde_curr[SMEM_IDX(0, TID_J, TID_K)] + S[IDX_I + Ni] * smem_x_tilde_curr[SMEM_IDX(1, TID_J, TID_K)]
             + R[IDX_I] * smem_x_tilde_next[SMEM_IDX(0, TID_J, TID_K)] + R[IDX_I + Ni] * smem_x_tilde_next[SMEM_IDX(1, TID_J, TID_K)];
    }
    
    #undef IDX_I
    #undef IDX_J
    #undef IDX_K
    #undef TID_I
    #undef TID_J
    #undef TID_K
    #undef SMEM_IDX
}



/*!
 * Launch "kernelUpdateLocalSolDimI"
 * \param [in, out]    x_loc            y_{rank} as input and x_{rank} as output (see Fig.4 and Eq.(17) in Song et al. JCP (2022) 111443)
 * \param [in]         x_tilde_curr     The solution to the reduced system in the current partition (see Fig.4 in Song et al. JCP (2022) 11143)
 * \param [in]         x_tilde_next     The solution to the reduced system in the next partition (see Fig.4 in Song et al. JCP (2022) 11143)
 * \param [in]         S                Preprocessed coefficients stored in column-major defined in Eq.(15) in Song et al. JCP (2022) 11143
 * \param [in]         R                Preprocessed coefficients stored in column-major defined in Eq.(16) in Song et al. JCP (2022) 11143
 * \param [in]         Ni               Number of entries in x_loc in i-dimension
 * \param [in]         Nj               Number of entries in x_loc in j-dimension
 * \param [in]         Nk               Number of entries in x_loc in k-dimension
 * \param [in]      arr_stride_i        array access stride of x_loc in i-dimension, from i to i+1
 * \param [in]      arr_stride_j        array access stride of x_loc in j-dimension, from j to j+1
 */
template<typename RealType>
void updateLocalSolDimI(
              RealType* x_loc,
        const RealType* x_tilde_curr,
        const RealType* x_tilde_next,
        const RealType* S,
        const RealType* R,
        const unsigned int Ni, const unsigned int Nj, const unsigned int Nk,
        const unsigned int arr_stride_i, const unsigned int arr_stride_j
    )
{
    constexpr unsigned int NUM_THREADS_I = 32;
    constexpr unsigned int NUM_THREADS_J =  1;
    constexpr unsigned int NUM_THREADS_K = 16;
    constexpr unsigned int SMEM_SIZE = 4 * NUM_THREADS_J * NUM_THREADS_K * sizeof(RealType);
    constexpr unsigned int BLOCK_SIZE = NUM_THREADS_I * NUM_THREADS_J * NUM_THREADS_K;
    dim3  grid_size = dim3((Nk + NUM_THREADS_K - 1) / NUM_THREADS_K, (Nj + NUM_THREADS_J - 1) / NUM_THREADS_J, (Ni + NUM_THREADS_I - 1) / NUM_THREADS_I);
    kernelUpdateLocalSolDimI<NUM_THREADS_I, NUM_THREADS_J, NUM_THREADS_K, RealType><<<grid_size, BLOCK_SIZE, SMEM_SIZE>>>
        (x_loc, x_tilde_curr, x_tilde_next, S, R, Ni, Nj, Nk, arr_stride_i, arr_stride_j);
}



/*!
 * Calculate local solution from y_{rank} in j-dimension using x_tilde from current and next partitions (see Eq.(22) in Song et al. JCP (2022) 111443)
 * \param [in, out]    x_loc            y_{rank} as input and x_{rank} as output (see Fig.4 and Eq.(17) in Song et al. JCP (2022) 111443)
 * \param [in]         x_tilde_curr     The solution to the reduced system in the current partition (see Fig.4 in Song et al. JCP (2022) 11143)
 * \param [in]         x_tilde_next     The solution to the reduced system in the next partition (see Fig.4 in Song et al. JCP (2022) 11143)
 * \param [in]         S                Preprocessed coefficients stored in column-major defined in Eq.(15) in Song et al. JCP (2022) 11143
 * \param [in]         R                Preprocessed coefficients stored in column-major defined in Eq.(16) in Song et al. JCP (2022) 11143
 * \param [in]         Ni               Number of entries in x_loc in i-dimension
 * \param [in]         Nj               Number of entries in x_loc in j-dimension
 * \param [in]         Nk               Number of entries in x_loc in k-dimension
 * \param [in]      arr_stride_i        array access stride of x_loc in i-dimension, from i to i+1
 * \param [in]      arr_stride_j        array access stride of x_loc in j-dimension, from j to j+1
 */
template<unsigned int NUM_THREADS_I, unsigned int NUM_THREADS_J, unsigned int NUM_THREADS_K, typename RealType> __global__
void kernelUpdateLocalSolDimJ(
              RealType* __restrict__ x_loc,
        const RealType* __restrict__ x_tilde_curr,
        const RealType* __restrict__ x_tilde_next,
        const RealType* __restrict__ S,
        const RealType* __restrict__ R,
        const unsigned int Ni, const unsigned int Nj, const unsigned int Nk,
        const unsigned int arr_stride_i, const unsigned int arr_stride_j
    )
{
    #define SMEM_IDX(I, J, K) ((J) * (NUM_THREADS_K * NUM_THREADS_I) + (I) * NUM_THREADS_K + (K)) 
    extern __shared__ char smem_general[];
    RealType* smem_x_tilde_curr = reinterpret_cast<RealType*>(&smem_general[0]);
    RealType* smem_x_tilde_next = reinterpret_cast<RealType*>(&smem_general[2 * NUM_THREADS_I * NUM_THREADS_K * sizeof(RealType)]);

    #define TID_J (threadIdx.x / (NUM_THREADS_I * NUM_THREADS_K))
    #define TID_I ((threadIdx.x / NUM_THREADS_K) % NUM_THREADS_I)
    #define TID_K (threadIdx.x % NUM_THREADS_K)
    #define IDX_I (blockIdx.z * NUM_THREADS_I + TID_I)
    #define IDX_J (blockIdx.y * NUM_THREADS_J + TID_J)
    #define IDX_K (blockIdx.x * NUM_THREADS_K + TID_K)
    if ((TID_J < 2) && (IDX_I < Ni) && (IDX_K < Nk)) {
        smem_x_tilde_curr[SMEM_IDX(TID_I, TID_J, TID_K)] = x_tilde_curr[TID_J * Ni * Nk + IDX_I * Nk + IDX_K];
        smem_x_tilde_next[SMEM_IDX(TID_I, TID_J, TID_K)] = x_tilde_next[TID_J * Ni * Nk + IDX_I * Nk + IDX_K];
    }
    #undef IDX_I
    #undef IDX_J
    #undef IDX_K
    #undef TID_I
    #undef TID_J
    #undef TID_K
    __syncthreads();

    #define TID_I (threadIdx.x / (NUM_THREADS_J * NUM_THREADS_K))
    #define TID_J ((threadIdx.x / NUM_THREADS_K) % NUM_THREADS_J)
    #define TID_K (threadIdx.x % NUM_THREADS_K)
    #define IDX_I (blockIdx.z * NUM_THREADS_I + TID_I)
    #define IDX_J (blockIdx.y * NUM_THREADS_J + TID_J)
    #define IDX_K (blockIdx.x * NUM_THREADS_K + TID_K)

    if ((IDX_I < Ni) && (IDX_J < Nj) && (IDX_K < Nk)) {
        x_loc[IDX_I * arr_stride_i + IDX_J * arr_stride_j + IDX_K]
            -= S[IDX_J] * smem_x_tilde_curr[SMEM_IDX(TID_I, 0, TID_K)] + S[IDX_J + Nj] * smem_x_tilde_curr[SMEM_IDX(TID_I, 1, TID_K)]
             + R[IDX_J] * smem_x_tilde_next[SMEM_IDX(TID_I, 0, TID_K)] + R[IDX_J + Nj] * smem_x_tilde_next[SMEM_IDX(TID_I, 1, TID_K)];
    }
    
    #undef IDX_I
    #undef IDX_J
    #undef IDX_K
    #undef TID_I
    #undef TID_J
    #undef TID_K
    #undef SMEM_IDX
}



/*!
 * Launch "kernelUpdateLocalSolDimJ"
 * \param [in, out]    x_loc            y_{rank} as input and x_{rank} as output (see Fig.4 and Eq.(17) in Song et al. JCP (2022) 111443)
 * \param [in]         x_tilde_curr     The solution to the reduced system in the current partition (see Fig.4 in Song et al. JCP (2022) 11143)
 * \param [in]         x_tilde_next     The solution to the reduced system in the next partition (see Fig.4 in Song et al. JCP (2022) 11143)
 * \param [in]         S                Preprocessed coefficients stored in column-major defined in Eq.(15) in Song et al. JCP (2022) 11143
 * \param [in]         R                Preprocessed coefficients stored in column-major defined in Eq.(16) in Song et al. JCP (2022) 11143
 * \param [in]         Ni               Number of entries in x_loc in i-dimension
 * \param [in]         Nj               Number of entries in x_loc in j-dimension
 * \param [in]         Nk               Number of entries in x_loc in k-dimension
 * \param [in]      arr_stride_i        array access stride of x_loc in i-dimension, from i to i+1
 * \param [in]      arr_stride_j        array access stride of x_loc in j-dimension, from j to j+1
 */
template<typename RealType>
void updateLocalSolDimJ(
              RealType* x_loc,
        const RealType* x_tilde_curr,
        const RealType* x_tilde_next,
        const RealType* S,
        const RealType* R,
        const unsigned int Ni, const unsigned int Nj, const unsigned int Nk,
        const unsigned int arr_stride_i, const unsigned int arr_stride_j
    )
{
    constexpr unsigned int NUM_THREADS_I =  1;
    constexpr unsigned int NUM_THREADS_J = 32;
    constexpr unsigned int NUM_THREADS_K = 16;
    constexpr unsigned int SMEM_SIZE     = 4 * NUM_THREADS_I * NUM_THREADS_K * sizeof(RealType);
    constexpr unsigned int BLOCK_SIZE    = NUM_THREADS_I * NUM_THREADS_J * NUM_THREADS_K;
    dim3  grid_size = dim3((Nk + NUM_THREADS_K - 1) / NUM_THREADS_K, (Nj + NUM_THREADS_J - 1) / NUM_THREADS_J, (Ni + NUM_THREADS_I - 1) / NUM_THREADS_I);
    kernelUpdateLocalSolDimJ<NUM_THREADS_I, NUM_THREADS_J, NUM_THREADS_K, RealType><<<grid_size, BLOCK_SIZE, SMEM_SIZE>>>
        (x_loc, x_tilde_curr, x_tilde_next, S, R, Ni, Nj, Nk, arr_stride_i, arr_stride_j);
}



/*!
 * Calculate local solution from y_{rank} in k-dimension using x_tilde from current and next partitions (see Eq.(22) in Song et al. JCP (2022) 111443)
 * \param [in, out]    x_loc            y_{rank} as input and x_{rank} as output (see Fig.4 and Eq.(17) in Song et al. JCP (2022) 111443)
 * \param [in]         x_tilde_curr     The solution to the reduced system in the current partition (see Fig.4 in Song et al. JCP (2022) 11143)
 * \param [in]         x_tilde_next     The solution to the reduced system in the next partition (see Fig.4 in Song et al. JCP (2022) 11143)
 * \param [in]         S                Preprocessed coefficients stored in column-major defined in Eq.(15) in Song et al. JCP (2022) 11143
 * \param [in]         R                Preprocessed coefficients stored in column-major defined in Eq.(16) in Song et al. JCP (2022) 11143
 * \param [in]         Ni               Number of entries in x_loc in i-dimension
 * \param [in]         Nj               Number of entries in x_loc in j-dimension
 * \param [in]         Nk               Number of entries in x_loc in k-dimension
 * \param [in]      arr_stride_i        array access stride of x_loc in i-dimension, from i to i+1
 * \param [in]      arr_stride_j        array access stride of x_loc in j-dimension, from j to j+1
 */
template<unsigned int NUM_THREADS_I, unsigned int NUM_THREADS_J, unsigned int NUM_THREADS_K, typename RealType> __global__
void kernelUpdateLocalSolDimK(
              RealType* __restrict__ x_loc,
        const RealType* __restrict__ x_tilde_curr,
        const RealType* __restrict__ x_tilde_next,
        const RealType* __restrict__ S,
        const RealType* __restrict__ R,
        const unsigned int Ni, const unsigned int Nj, const unsigned int Nk,
        const unsigned int arr_stride_i, const unsigned int arr_stride_j
    )
{

    #define SMEM_IDX(I, J, K) ((K) * NUM_THREADS_I * NUM_THREADS_J + (I) * NUM_THREADS_J + (J)) 

    extern __shared__ char smem_general[];
    RealType* smem_x_tilde_curr = reinterpret_cast<RealType*>(&smem_general[0]);
    RealType* smem_x_tilde_next = reinterpret_cast<RealType*>(&smem_general[2 * NUM_THREADS_I * NUM_THREADS_J * sizeof(RealType)]);

    #define TID_K (threadIdx.x / (NUM_THREADS_I * NUM_THREADS_J))
    #define TID_I ((threadIdx.x / NUM_THREADS_J) % NUM_THREADS_I)
    #define TID_J (threadIdx.x % NUM_THREADS_J)
    #define IDX_I (blockIdx.z * NUM_THREADS_I + TID_I)
    #define IDX_J (blockIdx.y * NUM_THREADS_J + TID_J)
    #define IDX_K (blockIdx.x * NUM_THREADS_K + TID_K)

    if ((TID_K < 2) && (IDX_I < Ni) && (IDX_J < Nj)) {
        smem_x_tilde_curr[SMEM_IDX(TID_I, TID_J, TID_K)] = x_tilde_curr[TID_K * Ni * Nj + IDX_I * Nj + IDX_J];
        smem_x_tilde_next[SMEM_IDX(TID_I, TID_J, TID_K)] = x_tilde_next[TID_K * Ni * Nj + IDX_I * Nj + IDX_J];
    }
    #undef IDX_I
    #undef IDX_J
    #undef IDX_K
    #undef TID_I
    #undef TID_J
    #undef TID_K
    __syncthreads();

    #define TID_I (threadIdx.x / (NUM_THREADS_J * NUM_THREADS_K))
    #define TID_J ((threadIdx.x / NUM_THREADS_K) % NUM_THREADS_J)
    #define TID_K (threadIdx.x % NUM_THREADS_K)
    #define IDX_I (blockIdx.z * NUM_THREADS_I + TID_I)
    #define IDX_J (blockIdx.y * NUM_THREADS_J + TID_J)
    #define IDX_K (blockIdx.x * NUM_THREADS_K + TID_K)
    if ((IDX_I < Ni) && (IDX_J < Nj) && (IDX_K < Nk)) {
        x_loc[IDX_I * arr_stride_i + IDX_J * arr_stride_j + IDX_K]
            -= S[IDX_K] * smem_x_tilde_curr[SMEM_IDX(TID_I, TID_J, 0)] + S[IDX_K + Nk] * smem_x_tilde_curr[SMEM_IDX(TID_I, TID_J, 1)]
             + R[IDX_K] * smem_x_tilde_next[SMEM_IDX(TID_I, TID_J, 0)] + R[IDX_K + Nk] * smem_x_tilde_next[SMEM_IDX(TID_I, TID_J, 1)];
    }
    
    #undef IDX_I
    #undef IDX_J
    #undef IDX_K
    #undef TID_I
    #undef TID_J
    #undef TID_K
    #undef SMEM_IDX
}



/*!
 * Launch "kernelUpdateLocalSolDimK"
 * \param [in, out]    x_loc            y_{rank} as input and x_{rank} as output (see Fig.4 and Eq.(17) in Song et al. JCP (2022) 111443)
 * \param [in]         x_tilde_curr     The solution to the reduced system in the current partition (see Fig.4 in Song et al. JCP (2022) 11143)
 * \param [in]         x_tilde_next     The solution to the reduced system in the next partition (see Fig.4 in Song et al. JCP (2022) 11143)
 * \param [in]         S                Preprocessed coefficients stored in column-major defined in Eq.(15) in Song et al. JCP (2022) 11143
 * \param [in]         R                Preprocessed coefficients stored in column-major defined in Eq.(16) in Song et al. JCP (2022) 11143
 * \param [in]         Ni               Number of entries in x_loc in i-dimension
 * \param [in]         Nj               Number of entries in x_loc in j-dimension
 * \param [in]         Nk               Number of entries in x_loc in k-dimension
 * \param [in]      arr_stride_i        array access stride of x_loc in i-dimension, from i to i+1
 * \param [in]      arr_stride_j        array access stride of x_loc in j-dimension, from j to j+1
 */
template<typename RealType>
void updateLocalSolDimK(
              RealType* x_loc,
        const RealType* x_tilde_curr,
        const RealType* x_tilde_next,
        const RealType* S,
        const RealType* R,
        const unsigned int Ni, const unsigned int Nj, const unsigned int Nk,
        const unsigned int arr_stride_i, const unsigned int arr_stride_j
    )
{
    constexpr unsigned int NUM_THREADS_I =  1;
    constexpr unsigned int NUM_THREADS_J = 16;
    constexpr unsigned int NUM_THREADS_K = 32;
    constexpr unsigned int SMEM_SIZE     = 4 * NUM_THREADS_I * NUM_THREADS_J * sizeof(RealType);
    constexpr unsigned int BLOCK_SIZE    = NUM_THREADS_I * NUM_THREADS_J * NUM_THREADS_K;
    dim3  grid_size = dim3((Nk + NUM_THREADS_K - 1) / NUM_THREADS_K, (Nj + NUM_THREADS_J - 1) / NUM_THREADS_J, (Ni + NUM_THREADS_I - 1) / NUM_THREADS_I);
    kernelUpdateLocalSolDimK<NUM_THREADS_I, NUM_THREADS_J, NUM_THREADS_K, RealType><<<grid_size, BLOCK_SIZE, SMEM_SIZE>>>
        (x_loc, x_tilde_curr, x_tilde_next, S, R, Ni, Nj, Nk, arr_stride_i, arr_stride_j);
}





// EXPLICIT INSTANTIATION
template void localSolPentaPCRDimI<double>(double*, const double*, const double*, const double*, const double*, const double*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
template void localSolPentaPCRDimI< float>( float*, const  float*, const  float*, const  float*, const  float*, const  float*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
template void localSolPentaPCRDimJ<double>(double*, const double*, const double*, const double*, const double*, const double*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
template void localSolPentaPCRDimJ< float>( float*, const  float*, const  float*, const  float*, const  float*, const  float*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
template void localSolPentaPCRDimK<double>(double*, const double*, const double*, const double*, const double*, const double*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
template void localSolPentaPCRDimK< float>( float*, const  float*, const  float*, const  float*, const  float*, const  float*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);

template void reduceCurrBlockSymm<double, double>(double*, const double*, const double*, const double*, const double*, const double*, const unsigned int);
template void reduceCurrBlockSymm<double,  float>(double*, const  float*, const  float*, const double*, const double*, const double*, const unsigned int);

template void reduceCurrBlockOneSide<double, double>(double*, const double*, const double*, const double*, const unsigned int);
template void reduceCurrBlockOneSide<double,  float>(double*, const  float*, const double*, const double*, const unsigned int);

template void calcReducedSystemRHSLocal<double>(double*, const double*, const double*, const double*, const double*, const unsigned int);
template void calcReducedSystemRHSLocal< float>( float*, const  float*, const  float*, const  float*, const  float*, const unsigned int);

template void updateLocalSolDimI<double>(double*, const double*, const double*, const double*, const double*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
template void updateLocalSolDimI< float>( float*, const  float*, const  float*, const  float*, const  float*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
template void updateLocalSolDimJ<double>(double*, const double*, const double*, const double*, const double*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
template void updateLocalSolDimJ< float>( float*, const  float*, const  float*, const  float*, const  float*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
template void updateLocalSolDimK<double>(double*, const double*, const double*, const double*, const double*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
template void updateLocalSolDimK< float>( float*, const  float*, const  float*, const  float*, const  float*, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int);

} // namespace penta
} // namspace cmpk
