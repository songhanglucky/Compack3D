
#include <cassert>
#include <stdexcept>
#include "Compack3D_utils.h"
#include "Compack3D_penta.h"
#ifdef COMPACK3D_DEVICE_ENABLED_CUDA
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "Compack3D_utils_kernels.cuh"
#include "Compack3D_penta_kernels.cuh"
#endif


namespace cmpk {
namespace penta {


/*!
 * AXPY operation of 2x2 row-major matrices
 * C <- alpha * C + beta * AB
 * \param[out] C    matrix product
 * \param[in]  A    left multiplier
 * \param[in]  B    right multiplier
 * \param[in]  alpha    1st coefficient
 * \param[in]  beta     2nd coefficient
 */
template<typename RealType>
void mat2x2AXPY(RealType* C, const RealType* A, const RealType* B, const RealType alpha, const RealType beta) {
    RealType C_tmp[4];
    C_tmp[0] = alpha * C[0] + beta * (A[0] * B[0] + A[1] * B[2]);
    C_tmp[1] = alpha * C[1] + beta * (A[0] * B[1] + A[1] * B[3]);
    C_tmp[2] = alpha * C[2] + beta * (A[2] * B[0] + A[3] * B[2]);
    C_tmp[3] = alpha * C[3] + beta * (A[2] * B[1] + A[3] * B[3]);
    for (int i = 0; i < 4; i++) C[i] = C_tmp[i];
}



/*!
 * Invert a 2x2 row-major matrix
 * \param[out] A_inv    inverse result
 * \param[in]  A        input matrix
 */
template<typename RealType>
void mat2x2Inv(RealType* A_inv, const RealType* A) {
    const RealType det_inv = 1.0 / (A[0] * A[3] - A[1] * A[2]);
    A_inv[0] =  det_inv * A[3];
    A_inv[1] = -det_inv * A[1];
    A_inv[2] = -det_inv * A[2];
    A_inv[3] =  det_inv * A[0];
}



/*!
 * Calculate A * inv(B) of 2x2 matrices
 */
template<typename RealType>
void mat2x2AMultInvB(RealType* C, const RealType* A, const RealType* B) {
    const RealType det_inv = 1.0 / (B[0] * B[3] - B[1] * B[2]);
    C[0] = det_inv * ( A[0] * B[3] - A[1] * B[2]);
    C[1] = det_inv * (-A[0] * B[1] + A[1] * B[0]);
    C[2] = det_inv * ( A[2] * B[3] - A[3] * B[2]);
    C[3] = det_inv * (-A[2] * B[1] + A[3] * B[0]);
}




/*!
 * Solve a 5x5 system for factorization
 * See the linear system above. The right-hand side is fixed to be [0, 0, 1, 0, 0].
 *
 * 0 b x x x x x
 *     | | | | |-- row next 2
 *     | | | |---- row next 1
 *     | | |------ row curr
 *     | |-------- row prev 1
 *     |---------- row prev 2
 *
 * \param[out]     a_prev2     The coefficient for parallel cyclic reduction using row i-2
 * \param[out]     a_prev1     The coefficient for parallel cyclic reduction using row i-1
 * \param[out]     a_curr      The coefficient for parallel cyclic reduction using row i
 * \param[out]     a_next1     The coefficient for parallel cyclic reduction using row i+1
 * \param[out]     a_next2     The coefficient for parallel cyclic reduction using row i+2
 * \param[in,out]  A           The left-hand side 5x5 system (identical zero entries are fixed). The values are not preserved after the function is called.
 * \param[in]      row_label   The validity of each row evaluated bit-wised
 */
template<typename RealType> 
void solFact(RealType& a_prev2, RealType& a_prev1, RealType& a_curr, RealType& a_next1, RealType& a_next2, FactSysPenta<RealType>& A, const int row_label) {
    assert( (row_label & 0b00100) == 0b00100 );

    if ( (row_label & 0b10000) == 0b00000 ) {
        A.prev2[0] = 1.0;
        A.prev2[1] = 0.0;
        A.prev2[2] = 0.0;
        A.prev1[0] = 0.0;
    }

    if ( (row_label & 0b01000) == 0b00000 ) {
        assert( (row_label & 0b10000) == 0b00000 );
        A.prev1[1] = 1.0;
        A.prev1[2] = 0.0;
        A.prev1[3] = 0.0;
        A.curr [0] = 0.0;
        A.next1[0] = 0.0;
    }

    if ( (row_label & 0b00001) == 0b00000 ) {
        A.next2[0] = 0.0;
        A.next2[1] = 0.0;
        A.next2[2] = 1.0;
        A.next1[3] = 0.0;

    }

    if ( (row_label & 0b00010) == 0b00000 ) {
        assert( (row_label & 0b00001) == 0b00000 );
        A.next1[0] = 0.0;
        A.next1[1] = 0.0;
        A.next1[2] = 1.0;
        A. curr[2] = 0.0;
        A.prev1[3] = 0.0;
    }

    A.prev1[1] -= A.prev1[0] * A.prev2[1] / A.prev2[0];
    A.prev1[2] -= A.prev1[0] * A.prev2[2] / A.prev2[0];
    A.next1[2] -= A.next1[3] * A.next2[1] / A.next2[2];
    A.next1[1] -= A.next1[3] * A.next2[0] / A.next2[2];

    A.curr [1] -= A.curr [0] * A.prev1[2] / A.prev1[1];
    A.curr [2] -= A.curr [0] * A.prev1[3] / A.prev1[1];
    A.next1[1] -= A.next1[0] * A.prev1[2] / A.prev1[1];
    A.next1[2] -= A.next1[0] * A.prev1[3] / A.prev1[1];

    A.curr[1] -= A.curr[2] * A.next1[1] / A.next1[2];
    A.curr[0] -= A.curr[2] * A.next1[0] / A.next1[2];

    a_curr  = 1.0 / A.curr[1];
    a_prev1 = -A.curr [0] * a_curr  / A.prev1[1];
    a_next1 = -A.curr [2] * a_curr  / A.next1[2];
    a_prev2 = -A.prev1[0] * a_prev1 / A.prev2[0];
    a_next2 = -A.next1[3] * a_next1 / A.next2[2];
}



/*!
 * Factorize a locally stored acyclic penta-diagonal system
 * \param[in] fact_prev_2       coefficients of factorization of each row i using row i - 2
 * \param[in] fact_prev_1       coefficients of factorization of each row i using row i - 1
 * \param[in] fact_curr         coefficients of factorization of each row i using row i
 * \param[in] fact_next_1       coefficients of factorization of each row i using row i + 1
 * \param[in] fact_next_2       coefficients of factorization of each row i using row i + 2
 * \param[in] N                 size of total system
 * \param[in] sub_sys_size_max  size of maximum allowed sub-system that can be solved directly (need to be consistent with maximum shared memory allocated)
 */
template<typename RealType>
void localFactPenta(
        RealType* fact_prev_2,
        RealType* fact_prev_1,
        RealType* fact_curr,
        RealType* fact_next_1,
        RealType* fact_next_2,
        RealType* l2,
        RealType* l1,
        RealType*  d,
        RealType* u1,
        RealType* u2,
        const int N,
        const int sub_sys_size_max
) {

    const int num_stages = log2Ceil(N);

#ifdef COMPACK3D_DEBUG_MODE_ENABLED
    {
        const int fact_buffer_size = num_stages * N;
        const RealType norm = 1.0 / fact_buffer_size;
        RealType check_buffer = 0.0;
        try { // Check fact_prev_2 buffer
            for (int i = 0; i < fact_buffer_size; i++) check_buffer += norm * fact_prev_2[i] * fact_prev_2[i];
        } catch (...) {
            throw std::invalid_argument("Buffer fact_prev_2 encounters memory access error.");
        }

        try { // Check fact_prev_1 buffer
            for (int i = 0; i < fact_buffer_size; i++) check_buffer += norm * fact_prev_1[i] * fact_prev_1[i];
        } catch (...) {
            throw std::invalid_argument("Buffer fact_prev_1 encounters memory access error.");
        }

        try { // Check fact_curr buffer
            for (int i = 0; i < fact_buffer_size; i++) check_buffer += norm * fact_curr[i] * fact_curr[i];
        } catch (...) {
            throw std::invalid_argument("Buffer fact_curr encounters memory access error.");
        }

        try { // Check fact_next_1 buffer
            for (int i = 0; i < fact_buffer_size; i++) check_buffer += norm * fact_next_1[i] * fact_next_1[i];
        } catch (...) {
            throw std::invalid_argument("Buffer fact_next_1 encounters memory access error.");
        }

        try { // Check fact_next_2 buffer
            for (int i = 0; i < fact_buffer_size; i++) check_buffer += norm * fact_next_2[i] * fact_next_2[i];
        } catch (...) {
            throw std::invalid_argument("Buffer fact_next_2 encounters memory access error.");
        }

        try { // Check l2 buffer
            for (int i = 0; i < N; i++) check_buffer += norm * l2[i] * l2[i];
        } catch (...) {
            throw std::invalid_argument("Buffer l2 encounters a memory access error.");
        }

        try { // Check l1 buffer
            for (int i = 0; i < N; i++) check_buffer += norm * l1[i] * l1[i];
        } catch (...) {
            throw std::invalid_argument("Buffer l1 encounters a memory access error.");
        }

        try { // Check d buffer
            for (int i = 0; i < N; i++) check_buffer += norm * d[i] * d[i];
        } catch (...) {
            throw std::invalid_argument("Buffer d encounters a memory access error.");
        }

        try { // Check u1 buffer
            for (int i = 0; i < N; i++) check_buffer += norm * u1[i] * u1[i];
        } catch (...) {
            throw std::invalid_argument("Buffer u1 encounters a memory access error.");
        }

        try { // Check u2 buffer
            for (int i = 0; i < N; i++) check_buffer += norm * u2[i] * u2[i];
        } catch (...) {
            throw std::invalid_argument("Buffer u2 encounters a memory access error.");
        }

        if (check_buffer < 1e-32) throw std::invalid_argument("Bad system");
    }
#endif

    RealType* l2_tmp = new RealType [N];
    RealType* l1_tmp = new RealType [N];
    RealType*  d_tmp = new RealType [N];
    RealType* u1_tmp = new RealType [N];
    RealType* u2_tmp = new RealType [N];

    int stride = 1;
    int level  = 0;
    FactSysPenta<RealType> A;

    while (stride < N) {

        RealType *l2_read, *l2_write;
        RealType *l1_read, *l1_write;
        RealType * d_read, * d_write;
        RealType *u1_read, *u1_write;
        RealType *u2_read, *u2_write;

        if (level & 0b1) {
            l2_read = l2_tmp; l2_write = l2;
            l1_read = l1_tmp; l1_write = l1;
             d_read =  d_tmp;  d_write =  d;
            u1_read = u1_tmp; u1_write = u1;
            u2_read = u2_tmp; u2_write = u2;
        } else {
            l2_read = l2; l2_write = l2_tmp;
            l1_read = l1; l1_write = l1_tmp;
             d_read =  d;  d_write =  d_tmp;
            u1_read = u1; u1_write = u1_tmp;
            u2_read = u2; u2_write = u2_tmp;
        } 

        for (int i = 0; i < N; i++) {
            const int i_sub = i / stride;
            const int n_sub = (N + stride - 1 - (i % stride)) / stride;

            const int i_prev_2 = i - (stride << 1);
            const int i_prev_1 = i -  stride;
            const int i_next_1 = i +  stride;
            const int i_next_2 = i + (stride << 1);

            const int row_label = 0b00100
                                + ((i_sub >        2 ) << 4)
                                + ((i_sub >        0 ) << 3)
                                + ((i_sub < (n_sub-1)) << 1)
                                + ( i_sub < (n_sub-3)      );

            if (row_label & 0b10000) {
                A.prev2[0] = l1_read[i_prev_2] * ((i_sub - 2) > 0);
                A.prev2[1] = u1_read[i_prev_2] * ((i_sub - 2) < (n_sub - 1));
                A.prev2[2] = u2_read[i_prev_2] * ((i_sub - 2) < (n_sub - 2));
            }

            if (row_label & 0b01000) {
                A.prev1[0] = l2_read[i_prev_1] * ((i_sub - 1) > 1);
                A.prev1[1] =  d_read[i_prev_1];
                A.prev1[2] = u1_read[i_prev_1] * ((i_sub - 1) < (n_sub - 1));
                A.prev1[3] = u2_read[i_prev_1] * ((i_sub - 1) < (n_sub - 2));
            }

            if (row_label & 0b00010) {
                A.next1[0] = l2_read[i_next_1] * ((i_sub + 1) > 1);
                A.next1[1] = l1_read[i_next_1] * ((i_sub + 1) > 0);
                A.next1[2] =  d_read[i_next_1];
                A.next1[3] = u2_read[i_next_1] * ((i_sub + 1) < (n_sub - 2));
            }

            if (row_label & 0b00001) {
                A.next2[0] = l2_read[i_next_2] * ((i_sub + 2) > 1);
                A.next2[1] = l1_read[i_next_2] * ((i_sub + 2) > 0);
                A.next2[2] = u1_read[i_next_2] * ((i_sub + 2) < (n_sub - 1));
            }

            A.curr[0] = (i_sub >          0 ) * l1_read[i];
            A.curr[1] =                          d_read[i];
            A.curr[2] = (i_sub < (n_sub - 1)) * u1_read[i];

            const int idx_fact = level * N + locFactIdx(i, N, stride, sub_sys_size_max);
            RealType& a_prev_2 = fact_prev_2[idx_fact];
            RealType& a_prev_1 = fact_prev_1[idx_fact];
            RealType& a_curr   = fact_curr  [idx_fact];
            RealType& a_next_1 = fact_next_1[idx_fact];
            RealType& a_next_2 = fact_next_2[idx_fact];

            solFact(a_prev_2, a_prev_1, a_curr, a_next_1, a_next_2, A, row_label);

            l2_write[i] = 0.0;
            l1_write[i] = a_curr * l2_read[i] * (i_sub > 0);
             d_write[i] = a_curr *  d_read[i];
            u1_write[i] = a_curr * u2_read[i] * (i_sub < (n_sub - 1));
            u2_write[i] = 0.0;

            if (row_label & 0b10000) {
                l2_write[i] += a_prev_2 * l2_read[i_prev_2];
                l1_write[i] += a_prev_2 *  d_read[i_prev_2];
                 d_write[i] += a_prev_2 * u2_read[i_prev_2];
            }
            if (row_label & 0b01000) {
                l1_write[i] += a_prev_1 * l1_read[i_prev_1];
                 d_write[i] += a_prev_1 * u1_read[i_prev_1];
            }
            if (row_label & 0b00010) {
                 d_write[i] += a_next_1 * l1_read[i_next_1];
                u1_write[i] += a_next_1 * u1_read[i_next_1];
            }
            if (row_label & 0b00001) {
                 d_write[i] += a_next_2 * l2_read[i_next_2];
                u1_write[i] += a_next_2 *  d_read[i_next_2];
                u2_write[i] += a_next_2 * u2_read[i_next_2];
            }
        } // for

        stride <<= 1;
        level ++;
    }

    if (l2_tmp) { delete [] l2_tmp; l2_tmp = nullptr; }
    if (l1_tmp) { delete [] l1_tmp; l1_tmp = nullptr; }
    if ( d_tmp) { delete []  d_tmp;  d_tmp = nullptr; }
    if (u1_tmp) { delete [] u1_tmp; u1_tmp = nullptr; }
    if (u2_tmp) { delete [] u2_tmp; u2_tmp = nullptr; }
}



/*!
 * Simulation of local solve process (for development use only)
 * \param [in, out] x                   right-hand side of the system and the solution as the output
 * \param [in]      fact_prev_2         factorization coefficients of the second previous row
 * \param [in]      fact_prev_1         factorization coefficients of the first previous row
 * \param [in]      fact_curr           factorization coefficients of the current row
 * \param [in]      fact_next_1         factorization coefficients of the first next row
 * \param [in]      fact_next_2         factorization coefficients of the second next row
 * \param [in]      N                   size (total number of rows) of the local system
 * \param [in]      max_sub_sys_size    maximum sub-system size that can be solved with one memory load
 */
template<typename RealType>
void vanillaLocalSolPentaPCR(
        RealType*                 x,
        const RealType* fact_prev_2,
        const RealType* fact_prev_1,
        const RealType* fact_curr  ,
        const RealType* fact_next_1,
        const RealType* fact_next_2,
        const int N,
        const int max_sub_sys_size
) {
    RealType* x_buf = new RealType [N];
    int stride = 1;
    int level  = 0;
    while (stride < N) {
        RealType* x_read  = (level & 0b1) ? x_buf : x;
        RealType* x_write = (level & 0b1) ? x : x_buf;
        for (int i = 0; i < N; i++) {
            const int i_sub = i / stride;
            const int n_sub = (N + stride - 1 - (i % stride)) / stride;

            const int row_label = 0b00100
                                + ((i_sub >        2 ) << 4)
                                + ((i_sub >        0 ) << 3)
                                + ((i_sub < (n_sub-1)) << 1)
                                + ( i_sub < (n_sub-3)      );


            const int idx_fact = level * N + locFactIdx(i, N, stride, max_sub_sys_size);
            x_write[i] = fact_curr[idx_fact] * x_read[i];
            if (row_label & 0b10000) x_write[i] += fact_prev_2[idx_fact] * x_read[i - (stride << 1)];
            if (row_label & 0b01000) x_write[i] += fact_prev_1[idx_fact] * x_read[i -  stride      ];
            if (row_label & 0b00010) x_write[i] += fact_next_1[idx_fact] * x_read[i +  stride      ];
            if (row_label & 0b00001) x_write[i] += fact_next_2[idx_fact] * x_read[i + (stride << 1)];
            
        }
        stride <<= 1;
        level ++;
    }
    if (level & 0b1) for (int i = 0; i < N; i++) x[i] = x_buf[i];
    delete [] x_buf;
}



/*!
 * Simulation of local solve process (solve along aligned memory)
 * \param [in, out] x_batch             right-hand side of the system and the solution as the output
 * \param [in]      fact_prev_2         factorization coefficients of the second previous row
 * \param [in]      fact_prev_1         factorization coefficients of the first previous row
 * \param [in]      fact_curr           factorization coefficients of the current row
 * \param [in]      fact_next_1         factorization coefficients of the first next row
 * \param [in]      fact_next_2         factorization coefficients of the second next row
 * \param [in]      N                   size (total number of rows) of the local system
 * \param [in]      batch_size          batch_size
 * \param [in]      max_sub_sys_size    maximum sub-system size that can be solved with one memory load
 */
template<typename RealType>
void vanillaLocalSolPentaPCRBatch(
        RealType*           x_batch,
        const RealType* fact_prev_2,
        const RealType* fact_prev_1,
        const RealType* fact_curr  ,
        const RealType* fact_next_1,
        const RealType* fact_next_2,
        const int N,
        const int batch_size,
        const int max_sub_sys_size
) {
    RealType* x_batch_buf = new RealType [N * batch_size];
    int stride = 1;
    int level  = 0;
    while (stride < N) {
        RealType* x_read  = (level & 0b1) ? x_batch_buf : x_batch;
        RealType* x_write = (level & 0b1) ? x_batch : x_batch_buf;
        for (int i = 0; i < N; i++) {
            const int i_sub = i / stride;
            const int n_sub = (N + stride - 1 - (i % stride)) / stride;
            const int row_label = 0b00100
                                + ((i_sub >        2 ) << 4)
                                + ((i_sub >        0 ) << 3)
                                + ((i_sub < (n_sub-1)) << 1)
                                + ( i_sub < (n_sub-3)      );
            const int idx_fact = level * N + locFactIdx(i, N, stride, max_sub_sys_size);
            for (int batch_offset = 0; batch_offset < batch_size; batch_offset ++) {
                const int i_batch = i + batch_offset * N;
                x_write[i_batch] = fact_curr[idx_fact] * x_read[i_batch];
                if (row_label & 0b10000) x_write[i_batch] += fact_prev_2[idx_fact] * x_read[i_batch - (stride << 1)];
                if (row_label & 0b01000) x_write[i_batch] += fact_prev_1[idx_fact] * x_read[i_batch -  stride      ];
                if (row_label & 0b00010) x_write[i_batch] += fact_next_1[idx_fact] * x_read[i_batch +  stride      ];
                if (row_label & 0b00001) x_write[i_batch] += fact_next_2[idx_fact] * x_read[i_batch + (stride << 1)];
            }
        }
        stride <<= 1;
        level ++;
    }
    if (level & 0b1) for (int i = 0; i < N; i++) x_batch[i] = x_batch_buf[i];
    delete [] x_batch_buf;
}



/*!
 * Factorize a block tridiagonal system (2x2) on distributed memory
 * \param [out] block_fact_prev    factorization coefficients of previous block
 * \param [out] block_fact_curr    factorization coefficients of current block
 * \param [out] block_fact_next    factorization coefficients of next block
 * \param [in]  LDU                row of distributed system stored in the current rank (size of 12 elements)
 * \param [in]  Np                 number of distributed memory partitions involved in the system
 * \param [in]  comm_sub           MPI sub-communicator that managed the system
 * 
 * \note The memory layout of a block structure follows the row-major order, i.e., 00, 01, 10, 11.
 * \note Np must be no greater than the MPI_Comm_size(comm_sub)
 */
template<typename RealType>
void distFactPenta(RealType* block_fact_prev, RealType* block_fact_curr, RealType* block_fact_next, RealType* LDU, const int Np, MPI_Comm comm_sub) {
    const int num_stages = numDistSolSteps(Np);

#ifdef COMPACK3D_DEBUG_MODE_ENABLED
    {
        int mpi_size;
        MPI_Comm_size(comm_sub, &mpi_size);
        if (Np > mpi_size) {
            throw std::invalid_argument("Np is greater than maximum number of partitions involved in the reduced system.");
        }

        const int fact_buffer_size = 4 * num_stages;

        try { // Check block_fact_prev buffer
            for (int i = 0; i < fact_buffer_size; i++) block_fact_prev[i];
        } catch (...) {
            throw std::invalid_argument("Buffer block_fact_prev encounters a memory access error.");
        }

        try { // Check block_fact_curr buffer
            for (int i = 0; i < fact_buffer_size; i++) block_fact_curr[i];
        } catch (...) {
            throw std::invalid_argument("Buffer block_fact_curr encounters a memory access error.");
        }

        try { // Check block_fact_next buffer
            for (int i = 0; i < fact_buffer_size; i++) block_fact_next[i];
        } catch (...) {
            throw std::invalid_argument("Buffer block_fact_next encounters a memory access error.");
        }

        try { // Check LDU buffer
            for (int i = 0; i < 12; i++) LDU[i];
        } catch (...) {
            throw std::invalid_argument("Buffer LDU encounters a memory access error.");
        }
    }
#endif

    int rank_curr;
    MPI_Comm_rank(comm_sub, &rank_curr);

    RealType LDU_prev[12];
    RealType LDU_next[12];

    RealType* L_prev = &LDU_prev[0];
    RealType* D_prev = &LDU_prev[4];
    RealType* U_prev = &LDU_prev[8];
    RealType* L_curr = &LDU     [0];
    RealType* D_curr = &LDU     [4];
    RealType* U_curr = &LDU     [8];
    RealType* L_next = &LDU_next[0];
    RealType* D_next = &LDU_next[4];
    RealType* U_next = &LDU_next[8];

    MPI_Request mpi_reqs [4];
    MPI_Status  mpi_stats[4];

    int stride = 1; // NOTE: stride is also equal to the number of sub-systems
    int N_sub  = Np;
    int fact_idx_orig = 0;
    while (N_sub) {
        if ((N_sub > 1) && (N_sub & 0b1)) {
            N_sub -= 1;
            const int detach_row_lo = N_sub * stride;
            const int detach_row_hi = detach_row_lo + stride;
            const int rank_prev = (rank_curr - stride + detach_row_hi) % detach_row_hi;
            const int rank_next = (rank_curr + stride                ) % detach_row_hi;

            RealType* fact_prev_step = &block_fact_prev[fact_idx_orig];
            RealType* fact_curr_step = &block_fact_curr[fact_idx_orig];
            RealType* fact_next_step = &block_fact_next[fact_idx_orig];
            for (int i = 0; i < 4; i++) {
                fact_prev_step[i] = 0.0;
                fact_curr_step[i] = 0.0;
                fact_next_step[i] = 0.0;
            }

            if(detach_row_lo <= rank_prev && rank_prev < detach_row_hi) {
                // Recv data from rank_prev
                MPI_Irecv(LDU_prev, 12, MPIDataType<RealType>::value, rank_prev, 451, comm_sub, &mpi_reqs[2]);
                MPI_Waitall(1, &mpi_reqs[2], &mpi_stats[2]);

                mat2x2AMultInvB<RealType>(fact_prev_step, L_curr, D_prev);
                mat2x2AXPY<RealType>(L_curr, fact_prev_step, L_prev, 0.0, -1.0); // L_curr =        - fact_prev_step * L_prev
                mat2x2AXPY<RealType>(D_curr, fact_prev_step, U_prev, 1.0, -1.0); // D_curr = D_curr - fact_prev_step * U_prev
                mat2x2Inv<RealType>(fact_curr_step, D_curr);
                for (int j = 0; j < 4; j++) D_curr[j] = (j == 0) + (j == 3); // set D_curr to identity
                mat2x2AXPY<RealType>(L_curr, fact_curr_step, L_curr, 0.0, 1.0);
                mat2x2AXPY<RealType>(U_curr, fact_curr_step, U_curr, 0.0, 1.0);
                mat2x2AXPY<RealType>(fact_prev_step, fact_curr_step, fact_prev_step, 0.0, -1.0);
            }

            if(detach_row_lo <= rank_next && rank_next < detach_row_hi) {
                // Recv data from rank_next
                MPI_Irecv(LDU_next, 12, MPIDataType<RealType>::value, rank_next, 461, comm_sub, &mpi_reqs[3]);
                MPI_Waitall(1, &mpi_reqs[3], &mpi_stats[3]);

                mat2x2AMultInvB<RealType>(fact_next_step, U_curr, D_next);
                mat2x2AXPY<RealType>(U_curr, fact_next_step, U_next, 0.0, -1.0); // U_curr =        - fact_next_step * U_next
                mat2x2AXPY<RealType>(D_curr, fact_next_step, L_next, 1.0, -1.0); // D_curr = D_curr - fact_next_step * L_next
                mat2x2Inv<RealType>(fact_curr_step, D_curr);
                for (int j = 0; j < 4; j++) D_curr[j] = (j == 0) + (j == 3); // set D_curr to identity
                mat2x2AXPY<RealType>(L_curr, fact_curr_step, L_curr, 0.0, 1.0);
                mat2x2AXPY<RealType>(U_curr, fact_curr_step, U_curr, 0.0, 1.0);
                mat2x2AXPY<RealType>(fact_next_step, fact_curr_step, fact_next_step, 0.0, -1.0);
            }

            if(detach_row_lo <= rank_curr && rank_curr < detach_row_hi) {
                // Send data to rank_prev and rank_next
                MPI_Isend(LDU, 12, MPIDataType<RealType>::value, rank_prev, 461, comm_sub, &mpi_reqs[0]);
                MPI_Isend(LDU, 12, MPIDataType<RealType>::value, rank_next, 451, comm_sub, &mpi_reqs[1]);
                MPI_Waitall(2, &mpi_reqs[0], &mpi_stats[0]);
            }

            fact_idx_orig += 4;
        } // if ((N_sub > 1) && (N_sub & 0b1))

        // Regular PCR
        const int N_attached = N_sub * stride;
        const int rank_prev = (rank_curr - stride + N_attached) % N_attached;
        const int rank_next = (rank_curr + stride             ) % N_attached;
        RealType* fact_prev_step = &block_fact_prev[fact_idx_orig];
        RealType* fact_curr_step = &block_fact_curr[fact_idx_orig];
        RealType* fact_next_step = &block_fact_next[fact_idx_orig];
        for (int i = 0; i < 4; i++) {
            fact_prev_step[i] = 0.0;
            fact_curr_step[i] = 0.0;
            fact_next_step[i] = 0.0;
        }

        if (rank_curr < N_attached) {
            MPI_Irecv(LDU_prev, 12, MPIDataType<RealType>::value, rank_prev, 451, comm_sub, &mpi_reqs[2]);
            MPI_Irecv(LDU_next, 12, MPIDataType<RealType>::value, rank_next, 461, comm_sub, &mpi_reqs[3]);
            MPI_Isend(LDU     , 12, MPIDataType<RealType>::value, rank_prev, 461, comm_sub, &mpi_reqs[0]);
            MPI_Isend(LDU     , 12, MPIDataType<RealType>::value, rank_next, 451, comm_sub, &mpi_reqs[1]);
            MPI_Waitall(4, mpi_reqs, mpi_stats);
            // Calculate factorization coefficients
            mat2x2AMultInvB<RealType>(fact_prev_step, L_curr, D_prev);
            mat2x2AXPY<RealType>(L_curr, fact_prev_step, L_prev, 0.0, -1.0); // L_curr =        - fact_prev_step * L_prev
            mat2x2AXPY<RealType>(D_curr, fact_prev_step, U_prev, 1.0, -1.0); // D_curr = D_curr - fact_prev_step * U_prev
            mat2x2AMultInvB<RealType>(fact_next_step, U_curr, D_next);
            mat2x2AXPY<RealType>(U_curr, fact_next_step, U_next, 0.0, -1.0); // U_curr =        - fact_next_step * U_next
            mat2x2AXPY<RealType>(D_curr, fact_next_step, L_next, 1.0, -1.0); // D_curr = D_curr - fact_next_step * L_next
            mat2x2Inv<RealType>(fact_curr_step, D_curr);
            for (int j = 0; j < 4; j++) D_curr[j] = (j == 0) + (j == 3); // set D_curr to identity
            mat2x2AXPY<RealType>(L_curr, fact_curr_step, L_curr, 0.0, 1.0);
            mat2x2AXPY<RealType>(U_curr, fact_curr_step, U_curr, 0.0, 1.0);
            mat2x2AXPY<RealType>(fact_prev_step, fact_curr_step, fact_prev_step, 0.0, -1.0);
            mat2x2AXPY<RealType>(fact_next_step, fact_curr_step, fact_next_step, 0.0, -1.0);
        } // if (rank_curr < N_attached)

        stride <<= 1;
        N_sub  >>= 1;
        fact_idx_orig += 4;
    } // while ((stride << 1) <= Np)

    // reattach
    stride >>= 2;
    while (stride) {
        if ((Np > stride) && (Np & stride)) {
            // Reattach row floor(Np / (stride<<1)) * (stride<<1) + j for j in [0, stride)
            const int attach_row_lo = (Np / (stride << 1)) * (stride << 1);
            const int attach_row_hi = attach_row_lo + stride;
            const int rank_prev = (rank_curr - stride + attach_row_hi) % attach_row_hi;
            const int rank_next = (rank_curr + stride                ) % attach_row_hi;
            RealType* fact_prev_step = &block_fact_prev[fact_idx_orig];
            RealType* fact_curr_step = &block_fact_curr[fact_idx_orig];
            RealType* fact_next_step = &block_fact_next[fact_idx_orig];
            for (int i = 0; i < 4; i++) {
                fact_prev_step[i] = 0.0;
                fact_curr_step[i] = 0.0;
                fact_next_step[i] = 0.0;
            }

            if(attach_row_lo <= rank_curr && rank_curr < attach_row_hi) {
                MPI_Irecv(LDU_prev, 12, MPIDataType<RealType>::value, rank_prev, 451, comm_sub, &mpi_reqs[2]);
                MPI_Irecv(LDU_next, 12, MPIDataType<RealType>::value, rank_next, 461, comm_sub, &mpi_reqs[3]);
                MPI_Waitall(2, &mpi_reqs[2], &mpi_stats[2]);
                // Calculate factorization coefficients
                mat2x2AMultInvB<RealType>(fact_prev_step, L_curr, D_prev);
                mat2x2AXPY<RealType>(L_curr, fact_prev_step, L_prev, 0.0, -1.0); // L_curr =        - fact_prev_step * L_prev
                mat2x2AXPY<RealType>(D_curr, fact_prev_step, U_prev, 1.0, -1.0); // D_curr = D_curr - fact_prev_step * U_prev
                mat2x2AMultInvB<RealType>(fact_next_step, U_curr, D_next);
                mat2x2AXPY<RealType>(U_curr, fact_next_step, U_next, 0.0, -1.0); // U_curr =        - fact_next_step * U_next
                mat2x2AXPY<RealType>(D_curr, fact_next_step, L_next, 1.0, -1.0); // D_curr = D_curr - fact_next_step * L_next
                mat2x2Inv<RealType>(fact_curr_step, D_curr);
                for (int j = 0; j < 4; j++) D_curr[j] = (j == 0) + (j == 3); // set D_curr to identity
                mat2x2AXPY<RealType>(L_curr, fact_curr_step, L_curr, 0.0, 1.0);
                mat2x2AXPY<RealType>(U_curr, fact_curr_step, U_curr, 0.0, 1.0);
                mat2x2AXPY<RealType>(fact_prev_step, fact_curr_step, fact_prev_step, 0.0, -1.0);
                mat2x2AXPY<RealType>(fact_next_step, fact_curr_step, fact_next_step, 0.0, -1.0);
            }

            if(attach_row_lo <= rank_prev && rank_prev < attach_row_hi) {
                MPI_Isend(LDU, 12, MPIDataType<RealType>::value, rank_prev, 461, comm_sub, &mpi_reqs[0]);
                MPI_Waitall(1, &mpi_reqs[0], &mpi_stats[0]);
            }

            if(attach_row_lo <= rank_next && rank_next < attach_row_hi) {
                MPI_Isend(LDU, 12, MPIDataType<RealType>::value, rank_next, 451, comm_sub, &mpi_reqs[1]);
                MPI_Waitall(1, &mpi_reqs[1], &mpi_stats[1]);
            }

            fact_idx_orig += 4;
        } // if (Np & stride)
        stride >>= 1;
    } // while (stride)
}


template<typename RealType, typename RealTypeComm = RealType>
inline void vanillaReduceCurrBlockOneSide(RealType* x_curr, RealTypeComm* x_prev_buf, RealType* fact_curr, RealType* fact_prev) {
    RealType x_curr_buf[2];
    x_curr_buf[0] = x_curr[0];
    x_curr_buf[1] = x_curr[1];

    x_curr[0] = fact_curr[0b00] * x_curr_buf[0] + fact_curr[0b01] * x_curr_buf[1]
              + fact_prev[0b00] * x_prev_buf[0] + fact_prev[0b01] * x_prev_buf[1];
    x_curr[1] = fact_curr[0b10] * x_curr_buf[0] + fact_curr[0b11] * x_curr_buf[1]
              + fact_prev[0b10] * x_prev_buf[0] + fact_prev[0b11] * x_prev_buf[1];
}


template<typename RealType, typename RealTypeComm = RealType>
inline void vanillaReduceCurrBlockSymm(RealType* x_curr, RealTypeComm* x_prev_buf, RealTypeComm* x_next_buf,
        RealType* fact_curr, RealType* fact_prev, RealType* fact_next) {
    RealType x_curr_buf[2];
    x_curr_buf[0] = x_curr[0];
    x_curr_buf[1] = x_curr[1];

    x_curr[0] = fact_curr[0b00] * x_curr_buf[0] + fact_curr[0b01] * x_curr_buf[1]
              + fact_prev[0b00] * x_prev_buf[0] + fact_prev[0b01] * x_prev_buf[1]
              + fact_next[0b00] * x_next_buf[0] + fact_next[0b01] * x_next_buf[1];
    x_curr[1] = fact_curr[0b10] * x_curr_buf[0] + fact_curr[0b11] * x_curr_buf[1]
              + fact_prev[0b10] * x_prev_buf[0] + fact_prev[0b11] * x_prev_buf[1]
              + fact_next[0b10] * x_next_buf[0] + fact_next[0b11] * x_next_buf[1];
}


/*!
 * Vanilla distributed solve
 * \param [in, out]    x_curr             right-hand side (b_hat) as input and solution (x_tilde) as output (see Fig.4 and Eq.(17) in Song et al. JCP (2022) 111443)
 * \param [in]         x_prev_buf         receive buffer for x_tilde from the previous rank
 * \param [in]         x_curr_buf         send buffer for x_tilde in the current rank
 * \param [in]         x_next_buf         receive buffer for x_tilde from the next rank
 * \param [in]         block_fact_prev    distributed factorization coefficients for the lower-diagonal block
 * \param [in]         block_fact_curr    distributed factorization coefficients for the diagonal block
 * \param [in]         block_fact_next    distributed factorization coefficients for the upper-diagonal block
 * \param [in]         Np                 total number of partitions in the solve direction
 * \param [in]         comm_sub           MPI sub-communicator of partitions in the solve direction for each reduced system
 */
template<typename RealType, typename RealTypeComm = RealType>
void vanillaDistSolve (RealType* x_curr, RealTypeComm* x_prev_buf, RealTypeComm* x_curr_buf, RealTypeComm* x_next_buf,
        RealType* block_fact_prev, RealType* block_fact_curr, RealType* block_fact_next,
        const int Np, MPI_Comm comm_sub)
{
#ifdef COMPACK3D_DEBUG_MODE_ENABLED
    {
        int mpi_size;
        MPI_Comm_size(comm_sub, &mpi_size);
        if (Np > mpi_size) {
            throw std::invalid_argument("Np is greater than maximum number of partitions involved in the reduced system.");
        }
    }
#endif

    int rank_curr;
    MPI_Comm_rank(comm_sub, &rank_curr);

    MPI_Request mpi_reqs [4];
    MPI_Status  mpi_stats[4];

    int stride = 1; // NOTE: stride is also equal to the number of sub-systems
    int N_sub  = Np;
    int fact_idx_orig = 0;

    while (N_sub) {
        // Detach process
        if ((N_sub > 1) && (N_sub & 0b1)) {
            N_sub -= 1;
            const int detach_row_lo = N_sub * stride;
            const int detach_row_hi = detach_row_lo + stride;
            const int rank_prev = (rank_curr - stride + detach_row_hi) % detach_row_hi;
            const int rank_next = (rank_curr + stride                ) % detach_row_hi;
            RealType* fact_prev_step = &block_fact_prev[fact_idx_orig];
            RealType* fact_curr_step = &block_fact_curr[fact_idx_orig];
            RealType* fact_next_step = &block_fact_next[fact_idx_orig];

            if(detach_row_lo <= rank_prev && rank_prev < detach_row_hi) {
                MPI_Recv(x_prev_buf, 2, MPIDataType<RealTypeComm>::value, rank_prev, 451, comm_sub, &mpi_stats[2]);
                vanillaReduceCurrBlockOneSide<RealType, RealTypeComm>(x_curr, x_prev_buf, fact_curr_step, fact_prev_step);
            }

            if(detach_row_lo <= rank_next && rank_next < detach_row_hi) {
                MPI_Recv(x_next_buf, 2, MPIDataType<RealTypeComm>::value, rank_next, 461, comm_sub, &mpi_stats[3]);
                vanillaReduceCurrBlockOneSide<RealType, RealTypeComm>(x_curr, x_next_buf, fact_curr_step, fact_next_step);
            }

            if(detach_row_lo <= rank_curr && rank_curr < detach_row_hi) {
                for (int i = 0; i < 2; i++) x_curr_buf[i] = static_cast<RealTypeComm>(x_curr[i]);
                MPI_Isend(x_curr_buf, 2, MPIDataType<RealTypeComm>::value, rank_prev, 461, comm_sub, &mpi_reqs[0]);
                MPI_Isend(x_curr_buf, 2, MPIDataType<RealTypeComm>::value, rank_next, 451, comm_sub, &mpi_reqs[1]);
                MPI_Waitall(2, &mpi_reqs[0], &mpi_stats[0]);
            }

            fact_idx_orig += 4;
        } // if (Np & stride)

        // Regular PCR
        const int N_attached = N_sub * stride;
        const int rank_prev = (rank_curr - stride + N_attached) % N_attached;
        const int rank_next = (rank_curr + stride             ) % N_attached;
        RealType* fact_prev_step = &block_fact_prev[fact_idx_orig];
        RealType* fact_curr_step = &block_fact_curr[fact_idx_orig];
        RealType* fact_next_step = &block_fact_next[fact_idx_orig];

        if (rank_curr < N_attached) {
            MPI_Irecv(x_prev_buf, 2, MPIDataType<RealTypeComm>::value, rank_prev, 451, comm_sub, &mpi_reqs[0]);
            MPI_Irecv(x_next_buf, 2, MPIDataType<RealTypeComm>::value, rank_next, 461, comm_sub, &mpi_reqs[1]);
            for (int i = 0; i < 2; i++) x_curr_buf[i] = static_cast<RealTypeComm>(x_curr[i]);
            MPI_Isend(x_curr_buf, 2, MPIDataType<RealTypeComm>::value, rank_next, 451, comm_sub, &mpi_reqs[2]);
            MPI_Isend(x_curr_buf, 2, MPIDataType<RealTypeComm>::value, rank_prev, 461, comm_sub, &mpi_reqs[3]);
            MPI_Waitall(4, mpi_reqs, mpi_stats);
            vanillaReduceCurrBlockSymm<RealType, RealTypeComm>(x_curr, x_prev_buf, x_next_buf, fact_curr_step, fact_prev_step, fact_next_step);
        } // if (rank_curr < N_attached)

        stride <<= 1;
        N_sub  >>= 1;
        fact_idx_orig += 4;
    } // while ((stride << 1) <= Np)

    stride >>= 2;
    while (stride) {
        if ((Np > stride) && (Np & stride)) {
            // Reattach row floor(Np / (stride<<1)) * (stride<<1) + j for j in [0, stride)
            const int attach_row_lo = (Np / (stride << 1)) * (stride << 1);
            const int N_attached = attach_row_lo + stride;
            const int rank_prev = (rank_curr - stride + N_attached) % N_attached;
            const int rank_next = (rank_curr + stride             ) % N_attached;
            RealType* fact_prev_step = &block_fact_prev[fact_idx_orig];
            RealType* fact_curr_step = &block_fact_curr[fact_idx_orig];
            RealType* fact_next_step = &block_fact_next[fact_idx_orig];

            if(attach_row_lo <= rank_curr && rank_curr < N_attached) {
                MPI_Irecv(x_prev_buf, 2, MPIDataType<RealTypeComm>::value, rank_prev, 451, comm_sub, &mpi_reqs[0]);
                MPI_Irecv(x_next_buf, 2, MPIDataType<RealTypeComm>::value, rank_next, 461, comm_sub, &mpi_reqs[1]);
                MPI_Waitall(2, &mpi_reqs[0], &mpi_stats[0]);
                vanillaReduceCurrBlockSymm<RealType, RealTypeComm>(x_curr, x_prev_buf, x_next_buf, fact_curr_step, fact_prev_step, fact_next_step);
            }

            if(attach_row_lo <= rank_prev && rank_prev < N_attached) {
                for (int i = 0; i < 2; i++) x_curr_buf[i] = static_cast<RealTypeComm>(x_curr[i]);
                MPI_Send(x_curr_buf, 2, MPIDataType<RealTypeComm>::value, rank_prev, 461, comm_sub);
            }

            if(attach_row_lo <= rank_next && rank_next < N_attached) {
                for (int i = 0; i < 2; i++) x_curr_buf[i] = static_cast<RealTypeComm>(x_curr[i]);
                MPI_Send(x_curr_buf, 2, MPIDataType<RealTypeComm>::value, rank_next, 451, comm_sub);
            }

            fact_idx_orig += 4;
        } // if (Np & stride)
        stride >>= 1;
    } // while (stride)
}



/*!
 * Factorize a penta-diagonal and store the factorization coefficients on host memory within each MPI rank
 * \param [out] fact_local_prev_2    factorization coefficients of the 2nd-previous rows in the local diagonal block solve in each step
 * \param [out] fact_local_prev_1    factorization coefficients of the 1st-previous rows in the local diagonal block solve in each step
 * \param [out] fact_local_curr      factorization coefficients of the current rows in the local diagonal block solve in each step
 * \param [out] fact_local_next_1    factorization coefficients of the 1st-next rows in the local diagonal block solve in each step
 * \param [out] fact_local_next_2    factorization coefficients of the 2nd-next rows in the local diagonal block solve in each step
 * \param [out] fact_dist_prev       factorization coefficients of the previous block in the distributed solve in each step
 * \param [out] fact_dist_curr       factorization coefficients of the current block in the distributed solve in each step
 * \param [out] fact_dist_next       factorization coefficients of the next block in the distributed solve in each step
 * \param [out] Si                   inv(Di) * Li_tilde see Eq.(15) in H. Song, K.V. Matsuno, J.R. West et al., JCP, 2022
 * \param [out] Ri                   inv(Di) * Ui_tilde see Eq.(16) in H. Song, K.V. Matsuno, J.R. West et al., JCP, 2022
 * \param [in]  part_L2              the 2nd-lower diagonal elements in the penta-diagonal system within the distributed partition
 * \param [in]  part_L1              the 1st-lower diagonal elements in the penta-diagonal system within the distributed partition
 * \param [in]  part_D               the diagonal elements in the penta-diagonal system within the distributed partition
 * \param [in]  part_U1              the 1st-upper diagonal elements in the penta-diagonal system within the distributed partition
 * \param [in]  part_U2              the 2nd-upper diagonal elements in the penta-diagonal system within the distributed partition
 * \param [in]  N_sub                number of rows in the distributed partition
 * \param [in]  Np                   total number of partitions
 * \param [in]  max_local_fact_size  maximum local factorization size on shared memory
 * \param [in]  comm_sub             MPI sub-communicator for the partitioned system
 */
template<typename RealType>
void factPartitionedPentaHost(
        RealType* fact_local_prev_2, RealType* fact_local_prev_1, RealType* fact_local_curr, RealType* fact_local_next_1, RealType* fact_local_next_2,
        RealType* fact_dist_prev, RealType* fact_dist_curr, RealType* fact_dist_next, RealType* Si, RealType* Ri,
        RealType* part_L2, RealType* part_L1, RealType* part_D, RealType* part_U1, RealType* part_U2,
        const int N_sub, const int Np, const int max_local_fact_size, MPI_Comm comm_sub
) {
    const int N_local = N_sub - 2;

    RealType LDU[12];
    RealType* L     = &LDU[0];
    RealType* D     = &LDU[4];
    RealType* U     = &LDU[8];
    L[0b00] = part_L2[0];
    L[0b01] = part_L1[0];
    L[0b10] = 0.0;
    L[0b11] = part_L2[1];
    D[0b00] = part_D [0];
    D[0b01] = part_U1[0];
    D[0b10] = part_L1[1];
    D[0b11] = part_D [1];
    U[0b00] = part_U2[0];
    U[0b01] = 0.0;
    U[0b10] = part_U1[1];
    U[0b11] = part_U2[1];

    RealType* local_L2 = &part_L2[2];
    RealType* local_L1 = &part_L1[2];
    RealType* local_D  = &part_D [2];
    RealType* local_U1 = &part_U1[2];
    RealType* local_U2 = &part_U2[2];

    RealType Li[3];
    Li[0] = local_L2[0];
    Li[1] = local_L1[0];
    Li[2] = local_L2[1];
    local_L2[0] = 0.0;
    local_L1[0] = 0.0;
    local_L2[1] = 0.0;

    RealType Ui[3];
    Ui[0] = local_U2[N_local - 2];
    Ui[1] = local_U1[N_local - 1];
    Ui[2] = local_U2[N_local - 1];
    local_U2[N_local - 2] = 0.0;
    local_U1[N_local - 1] = 0.0;
    local_U2[N_local - 1] = 0.0;

    localFactPenta(fact_local_prev_2, fact_local_prev_1, fact_local_curr, fact_local_next_1, fact_local_next_2, local_L2, local_L1, local_D, local_U1, local_U2, N_local, max_local_fact_size);

    // Set Si to Li see H. Song, K.V. Matsuno, J.R. West et al., JCP, 2022
    for (int j = 0; j < (2 * N_local); j ++) Si[j] = 0.0;
    Si[0          ] = Li[0];
    Si[    N_local] = Li[1];
    Si[1 + N_local] = Li[2];
    vanillaLocalSolPentaPCRBatch(Si, fact_local_prev_2, fact_local_prev_1, fact_local_curr, fact_local_next_1, fact_local_next_2, N_local, 2, max_local_fact_size);

    RealType Si_top[4], Si_bot[4];
    Si_top[0b00] = Si[0          ];
    Si_top[0b10] = Si[1          ];
    Si_top[0b01] = Si[    N_local];
    Si_top[0b11] = Si[1 + N_local];
    Si_bot[0b00] = Si[N_local - 2];
    Si_bot[0b10] = Si[N_local - 1];
    Si_bot[0b01] = Si[N_local - 2 + N_local];
    Si_bot[0b11] = Si[N_local - 1 + N_local];

    // Set Ri to Ui see H. Song, K.V. Matsuno, J.R. West et al., JCP, 2022
    for (int j = 0; j < (2 * N_local); j ++) Ri[j] = 0.0;
    Ri[N_local     - 2] = Ui[0]; // local_U2[N_local - 2];
    Ri[N_local     - 1] = Ui[1]; // local_U1[N_local - 1];
    Ri[N_local * 2 - 1] = Ui[2]; // local_U2[N_local - 1];
    vanillaLocalSolPentaPCRBatch<RealType>(Ri, fact_local_prev_2, fact_local_prev_1, fact_local_curr, fact_local_next_1, fact_local_next_2, N_local, 2, max_local_fact_size);

    RealType Ri_top[4], Ri_bot[4];
    Ri_top[0b00] = Ri[0          ];
    Ri_top[0b10] = Ri[1          ];
    Ri_top[0b01] = Ri[    N_local];
    Ri_top[0b11] = Ri[1 + N_local];
    Ri_bot[0b00] = Ri[N_local - 2];
    Ri_bot[0b10] = Ri[N_local - 1];
    Ri_bot[0b01] = Ri[N_local - 2 + N_local];
    Ri_bot[0b11] = Ri[N_local - 1 + N_local];

    // Assemble reduced system
    RealType Si_prev_bot[4], Ri_prev_bot[4];
    int rank_curr;
    MPI_Comm_rank(comm_sub, &rank_curr);
    MPI_Request mpi_reqs [4];
    MPI_Status  mpi_stats[4];
    MPI_Irecv(Si_prev_bot, 4, MPIDataType<RealType>::value, (rank_curr + Np - 1) % Np, 133, comm_sub, &mpi_reqs[0]);
    MPI_Irecv(Ri_prev_bot, 4, MPIDataType<RealType>::value, (rank_curr + Np - 1) % Np, 113, comm_sub, &mpi_reqs[1]);
    MPI_Isend(Si_bot, 4, MPIDataType<RealType>::value, (rank_curr + 1) % Np, 133, comm_sub, &mpi_reqs[2]);
    MPI_Isend(Ri_bot, 4, MPIDataType<RealType>::value, (rank_curr + 1) % Np, 113, comm_sub, &mpi_reqs[3]);
    MPI_Waitall(4, &mpi_reqs[0], &mpi_stats[0]);

    RealType LDU_hat [12];
    RealType* L_hat = &LDU_hat[0];
    RealType* D_hat = &LDU_hat[4];
    RealType* U_hat = &LDU_hat[8];
    for (int j = 0; j < 4; j++) D_hat[j] = D[j];
    mat2x2AXPY<RealType>(L_hat, L, Si_prev_bot, 0.0, -1.0);
    mat2x2AXPY<RealType>(U_hat, U, Ri_top     , 0.0, -1.0);
    mat2x2AXPY<RealType>(D_hat, L, Ri_prev_bot, 1.0, -1.0);
    mat2x2AXPY<RealType>(D_hat, U, Si_top     , 1.0, -1.0);

    distFactPenta<RealType>(fact_dist_prev, fact_dist_curr, fact_dist_next, LDU_hat, Np, comm_sub);
}



/*!
 * Allocate factorization buffers
 * \tparam    MemSpaceType    type of memory space either cmpk::MemSpace::Host or cmpk::MemSpaceDevice
 * \tparam    RealType        real type of factorization either double or float
 * \param [out]    fact_local_prev_2_buf    pointer to the array of factorization coefficients of the 2nd previous row for each step in local PCR  \see locFactIdx
 * \param [out]    fact_local_prev_1_buf    pointer to the array of factorization coefficients of the 1st previous row for each step in local PCR  \see locFactIdx
 * \param [out]    fact_local_curr_buf      pointer to the array of factorization coefficients of the currrent row for each step in local PCR for normalization
 * \param [out]    fact_local_next_1_buf    pointer to the array of factorization coefficients of the 1st next row for each step in local PCR \see locFactIdx
 * \param [out]    fact_local_next_2_buf    pointer to the array of factorization coefficients of the 2nd next row for each step in local PCR \see locFactIdx
 * \param [out]    fact_dist_prev_buf       pointer to the array of factorization coefficients of the previous block for each step in the distributed solve
 * \param [out]    fact_dist_curr_buf       pointer to the array of factorization coefficients of the current block for each step in the distributed solve
 * \param [out]    fact_dist_next_buf       pointer to the array of factorization coefficients of the next block for each step in the distributed solve
 * \param [out]    Si_buf                   pointer to the S_{i} vector defined in Eq.(15) in Song et al. JCP (2022) 111443
 * \param [out]    Ri_buf                   pointer to the R_{i} vector defined in Eq.(16) in Song et al. JCP (2022) 111443
 * \param [out]    Li_tilde_tail_buf        pointer to the row-major array of the last three entries in \tilde{L}_{i} defined in Fig.4 in Song et al. JCP (2022) 111443
 * \param [out]    Ui_tilde_head_buf        pointer to the row-major array of the first three entries in \tilde{U}_{i} defined in Fig.4 in Song et al. JCP (2022) 111443
 * \param  [in]    N_sub                    number of entries in the solve direction in the current partition
 * \param  [in]    Np                       total number of partitions in the solve direction
 *
 * \note All output array buffers must be initially nullptr
 */
template<typename MemSpaceType, typename RealType>
void allocFactBuffers(
        RealType** fact_local_prev_2_buf, RealType** fact_local_prev_1_buf, RealType** fact_local_curr_buf, RealType** fact_local_next_1_buf, RealType** fact_local_next_2_buf,
        RealType** fact_dist_prev_buf, RealType** fact_dist_curr_buf, RealType** fact_dist_next_buf,
        RealType** Si_buf, RealType** Ri_buf, RealType** Li_tilde_tail_buf, RealType** Ui_tilde_head_buf,
        const unsigned int N_sub, const unsigned int Np
) {
    static_assert(std::is_same<MemSpaceType, MemSpace::Host>::value || std::is_same<MemSpaceType, MemSpace::Device>::value, "Valid MemSpace typenames are MemSpace::Host and MemSpace::Device.");

    if ((*fact_local_prev_2_buf) != nullptr) throw std::invalid_argument("\"*fact_local_prev_2_buf\" must be nullptr initially to be allocated safely.");
    if ((*fact_local_prev_1_buf) != nullptr) throw std::invalid_argument("\"*fact_local_prev_1_buf\" must be nullptr initially to be allocated safely.");
    if ((*fact_local_next_1_buf) != nullptr) throw std::invalid_argument("\"*fact_local_next_1_buf\" must be nullptr initially to be allocated safely.");
    if ((*fact_local_next_2_buf) != nullptr) throw std::invalid_argument("\"*fact_local_next_2_buf\" must be nullptr initially to be allocated safely.");
    if ((*fact_dist_prev_buf)    != nullptr) throw std::invalid_argument("\"*fact_dist_prev_buf\" must be nullptr initially to be allocated safely.");
    if ((*fact_dist_curr_buf)    != nullptr) throw std::invalid_argument("\"*fact_dist_curr_buf\" must be nullptr initially to be allocated safely.");
    if ((*fact_dist_next_buf)    != nullptr) throw std::invalid_argument("\"*fact_dist_next_buf\" must be nullptr initially to be allocated safely.");
    if ((*Si_buf)                != nullptr) throw std::invalid_argument("\"*Si_buf\" must be nullptr initially to be allocated safely.");
    if ((*Ri_buf)                != nullptr) throw std::invalid_argument("\"*Si_buf\" must be nullptr initially to be allocated safely.");
    if ((*Li_tilde_tail_buf)     != nullptr) throw std::invalid_argument("\"*Li_tilde_tail_buf\" must be nullptr initially to be allocated safely.");
    if ((*Ui_tilde_head_buf)     != nullptr) throw std::invalid_argument("\"*Ui_tilde_head_buf\" must be nullptr initially to be allocated safely.");

    const unsigned int local_fact_size = (N_sub - 2) * log2Ceil(N_sub - 2);
    const unsigned int dist_fact_size = 4 * numDistSolSteps<unsigned int>(Np);
    const int local_soln_size = 2 * (N_sub - 2);
    memAllocArray<MemSpaceType, RealType>(fact_local_prev_2_buf, local_fact_size);
    memAllocArray<MemSpaceType, RealType>(fact_local_prev_1_buf, local_fact_size);
    memAllocArray<MemSpaceType, RealType>(fact_local_curr_buf  , local_fact_size);
    memAllocArray<MemSpaceType, RealType>(fact_local_next_1_buf, local_fact_size);
    memAllocArray<MemSpaceType, RealType>(fact_local_next_2_buf, local_fact_size);
    memAllocArray<MemSpaceType, RealType>(fact_dist_prev_buf   , dist_fact_size);
    memAllocArray<MemSpaceType, RealType>(fact_dist_curr_buf   , dist_fact_size);
    memAllocArray<MemSpaceType, RealType>(fact_dist_next_buf   , dist_fact_size);
    memAllocArray<MemSpaceType, RealType>(Si_buf               , local_soln_size);
    memAllocArray<MemSpaceType, RealType>(Ri_buf               , local_soln_size);
    memAllocArray<MemSpaceType, RealType>(Li_tilde_tail_buf    , 3);
    memAllocArray<MemSpaceType, RealType>(Ui_tilde_head_buf    , 3);
}



/*!
 * Destroy the factorization buffers and free out the memory
 */
template<typename MemSpaceType, typename RealType>
void freeFactBuffers(
        RealType* fact_local_prev_2, RealType* fact_local_prev_1, RealType* fact_local_curr, RealType* fact_local_next_1, RealType* fact_local_next_2,
        RealType* fact_dist_prev, RealType* fact_dist_curr, RealType* fact_dist_next, RealType* Si, RealType* Ri, RealType* Li_tilde_tail, RealType* Ui_tilde_head)
{
    static_assert(std::is_same<MemSpaceType, MemSpace::Host>::value || std::is_same<MemSpaceType, MemSpace::Device>::value, "Valid MemSpace typenames are MemSpace::Host and MemSpace::Device.");

    if (fact_local_prev_2) memFreeArray<MemSpaceType, RealType>(fact_local_prev_2);
    if (fact_local_prev_1) memFreeArray<MemSpaceType, RealType>(fact_local_prev_1);
    if (fact_local_curr  ) memFreeArray<MemSpaceType, RealType>(fact_local_curr  );
    if (fact_local_next_1) memFreeArray<MemSpaceType, RealType>(fact_local_next_1);
    if (fact_local_next_2) memFreeArray<MemSpaceType, RealType>(fact_local_next_2);
    if (fact_dist_prev)    memFreeArray<MemSpaceType, RealType>(fact_dist_prev);
    if (fact_dist_curr)    memFreeArray<MemSpaceType, RealType>(fact_dist_curr);
    if (fact_dist_next)    memFreeArray<MemSpaceType, RealType>(fact_dist_next);
    if (Si)                memFreeArray<MemSpaceType, RealType>(Si);
    if (Ri)                memFreeArray<MemSpaceType, RealType>(Ri);
    if (Li_tilde_tail)     memFreeArray<MemSpaceType, RealType>(Li_tilde_tail);
    if (Ui_tilde_head)     memFreeArray<MemSpaceType, RealType>(Ui_tilde_head);
    
}




/*!
 * Factorize a penta-diagonal and store the factorization coefficients on host memory within each MPI rank
 * \param [out] fact_local_prev_2    factorization coefficients of the 2nd-previous rows in the local diagonal block solve in each step
 * \param [out] fact_local_prev_1    factorization coefficients of the 1st-previous rows in the local diagonal block solve in each step
 * \param [out] fact_local_curr      factorization coefficients of the current rows in the local diagonal block solve in each step
 * \param [out] fact_local_next_1    factorization coefficients of the 1st-next rows in the local diagonal block solve in each step
 * \param [out] fact_local_next_2    factorization coefficients of the 2nd-next rows in the local diagonal block solve in each step
 * \param [out] fact_dist_prev       factorization coefficients of the previous block in the distributed solve in each step
 * \param [out] fact_dist_curr       factorization coefficients of the current block in the distributed solve in each step
 * \param [out] fact_dist_next       factorization coefficients of the next block in the distributed solve in each step
 * \param [out] Si                   inv(Di) * Li_tilde see Eq.(15) in H. Song, K.V. Matsuno, J.R. West et al., JCP, 2022
 * \param [out] Ri                   inv(Di) * Ui_tilde see Eq.(16) in H. Song, K.V. Matsuno, J.R. West et al., JCP, 2022
 * \param [out] Li_tilde_tail        non-zero entries in the Li_tilde
 * \param [out] Ui_tilde_head        non-zero entries in the Ui_tilde
 * \param [in]  part_L2              the 2nd-lower diagonal elements in the penta-diagonal system within the distributed partition
 * \param [in]  part_L1              the 1st-lower diagonal elements in the penta-diagonal system within the distributed partition
 * \param [in]  part_D               the diagonal elements in the penta-diagonal system within the distributed partition
 * \param [in]  part_U1              the 1st-upper diagonal elements in the penta-diagonal system within the distributed partition
 * \param [in]  part_U2              the 2nd-upper diagonal elements in the penta-diagonal system within the distributed partition
 * \param [in]  N_sub                number of rows in the distributed partition
 * \param [in]  Np                   total number of partitions
 * \param [in]  max_local_fact_size  maximum local factorization size on shared memory
 * \param [in]  comm_sub             MPI sub-communicator for the partitioned system
 */
template<typename MemSpaceType, typename RealType>
void factPartitionedPenta(
        RealType* fact_local_prev_2, RealType* fact_local_prev_1, RealType* fact_local_curr, RealType* fact_local_next_1, RealType* fact_local_next_2,
        RealType* fact_dist_prev, RealType* fact_dist_curr, RealType* fact_dist_next, RealType* Si, RealType* Ri, RealType* Li_tilde_tail, RealType* Ui_tilde_head,
        RealType* part_L2, RealType* part_L1, RealType* part_D, RealType* part_U1, RealType* part_U2,
        const unsigned int N_sub, const unsigned int Np, const unsigned int max_local_fact_size, MPI_Comm comm_sub
) {
    static_assert(std::is_same<MemSpaceType, MemSpace::Host>::value || std::is_same<MemSpaceType, MemSpace::Device>::value, "Valid MemSpace typenames are MemSpace::Host and MemSpace::Device.");

    if (N_sub < 4) throw std::invalid_argument("N_sub must be no less than 4.");

    RealType* fact_local_prev_2_host = nullptr;
    RealType* fact_local_prev_1_host = nullptr;
    RealType* fact_local_curr_host   = nullptr;
    RealType* fact_local_next_1_host = nullptr;
    RealType* fact_local_next_2_host = nullptr;
    RealType* fact_dist_prev_host    = nullptr;
    RealType* fact_dist_curr_host    = nullptr;
    RealType* fact_dist_next_host    = nullptr;
    RealType* Si_host                = nullptr;
    RealType* Ri_host                = nullptr;
    RealType* Li_tilde_tail_host     = nullptr;
    RealType* Ui_tilde_head_host     = nullptr;

    const unsigned int local_fact_size = (N_sub - 2) * log2Ceil(N_sub - 2);
    const unsigned int dist_fact_size = 4 * numDistSolSteps<unsigned int>(Np);
    const unsigned int local_soln_size = 2 * (N_sub - 2);

    if constexpr (std::is_same<MemSpaceType, MemSpace::Device>::value) {
        memAllocArray<MemSpace::Host, RealType>(&fact_local_prev_2_host, local_fact_size);
        memAllocArray<MemSpace::Host, RealType>(&fact_local_prev_1_host, local_fact_size);
        memAllocArray<MemSpace::Host, RealType>(&fact_local_curr_host  , local_fact_size);
        memAllocArray<MemSpace::Host, RealType>(&fact_local_next_1_host, local_fact_size);
        memAllocArray<MemSpace::Host, RealType>(&fact_local_next_2_host, local_fact_size);
        memAllocArray<MemSpace::Host, RealType>(&fact_dist_prev_host, dist_fact_size);
        memAllocArray<MemSpace::Host, RealType>(&fact_dist_curr_host, dist_fact_size);
        memAllocArray<MemSpace::Host, RealType>(&fact_dist_next_host, dist_fact_size);
        memAllocArray<MemSpace::Host, RealType>(&Si_host, local_soln_size);
        memAllocArray<MemSpace::Host, RealType>(&Ri_host, local_soln_size);
        memAllocArray<MemSpace::Host, RealType>(&Li_tilde_tail_host, 3);
        memAllocArray<MemSpace::Host, RealType>(&Ui_tilde_head_host, 3);
    } else {
        fact_local_prev_2_host = fact_local_prev_2;
        fact_local_prev_1_host = fact_local_prev_1;
        fact_local_curr_host   = fact_local_curr;
        fact_local_next_1_host = fact_local_next_1;
        fact_local_next_2_host = fact_local_next_2;
        fact_dist_prev_host    = fact_dist_prev;
        fact_dist_curr_host    = fact_dist_curr;
        fact_dist_next_host    = fact_dist_next;
        Si_host                = Si;
        Ri_host                = Ri;
        Li_tilde_tail_host     = Li_tilde_tail;
        Ui_tilde_head_host     = Ui_tilde_head;
    }

    Li_tilde_tail_host[0] = part_L2[0];
    Li_tilde_tail_host[1] = part_L1[0];
    Li_tilde_tail_host[2] = part_L2[1];
    Ui_tilde_head_host[0] = part_U2[0];
    Ui_tilde_head_host[1] = part_U1[1];
    Ui_tilde_head_host[2] = part_U2[1];

    factPartitionedPentaHost(
        fact_local_prev_2_host, fact_local_prev_1_host, fact_local_curr_host, fact_local_next_1_host, fact_local_next_2_host,
        fact_dist_prev_host, fact_dist_curr_host, fact_dist_next_host, Si_host, Ri_host,
        part_L2, part_L1, part_D, part_U1, part_U2,
        N_sub, Np, max_local_fact_size, comm_sub
    );

    if constexpr (std::is_same<MemSpaceType, MemSpace::Device>::value) {
        deepCopy<MemSpace::Device, MemSpace::Host, RealType>(fact_local_prev_2, fact_local_prev_2_host, local_fact_size);
        deepCopy<MemSpace::Device, MemSpace::Host, RealType>(fact_local_prev_1, fact_local_prev_1_host, local_fact_size);
        deepCopy<MemSpace::Device, MemSpace::Host, RealType>(fact_local_curr  , fact_local_curr_host  , local_fact_size);
        deepCopy<MemSpace::Device, MemSpace::Host, RealType>(fact_local_next_1, fact_local_next_1_host, local_fact_size);
        deepCopy<MemSpace::Device, MemSpace::Host, RealType>(fact_local_next_2, fact_local_next_2_host, local_fact_size);
        deepCopy<MemSpace::Device, MemSpace::Host, RealType>(fact_dist_prev, fact_dist_prev_host, dist_fact_size);
        deepCopy<MemSpace::Device, MemSpace::Host, RealType>(fact_dist_curr, fact_dist_curr_host, dist_fact_size);
        deepCopy<MemSpace::Device, MemSpace::Host, RealType>(fact_dist_next, fact_dist_next_host, dist_fact_size);
        deepCopy<MemSpace::Device, MemSpace::Host, RealType>(Si, Si_host, local_soln_size);
        deepCopy<MemSpace::Device, MemSpace::Host, RealType>(Ri, Ri_host, local_soln_size);
        deepCopy<MemSpace::Device, MemSpace::Host, RealType>(Li_tilde_tail, Li_tilde_tail_host, 3);
        deepCopy<MemSpace::Device, MemSpace::Host, RealType>(Ui_tilde_head, Ui_tilde_head_host, 3);
        memFreeArray<MemSpace::Host, RealType>(fact_local_prev_2_host);
        memFreeArray<MemSpace::Host, RealType>(fact_local_prev_1_host);
        memFreeArray<MemSpace::Host, RealType>(fact_local_curr_host  );
        memFreeArray<MemSpace::Host, RealType>(fact_local_next_1_host);
        memFreeArray<MemSpace::Host, RealType>(fact_local_next_2_host);
        memFreeArray<MemSpace::Host, RealType>(fact_dist_prev_host);
        memFreeArray<MemSpace::Host, RealType>(fact_dist_curr_host);
        memFreeArray<MemSpace::Host, RealType>(fact_dist_next_host);
        memFreeArray<MemSpace::Host, RealType>(Si_host);
        memFreeArray<MemSpace::Host, RealType>(Ri_host);
        memFreeArray<MemSpace::Host, RealType>(Li_tilde_tail_host);
        memFreeArray<MemSpace::Host, RealType>(Ui_tilde_head_host);
    } 
}



/*!
 * Solving reduced system on distributed memory with performance
 * \param [in, out]    x_curr             right-hand side (b_hat) as input and solution (x_tilde) as output (see Fig.4 and Eq.(17) in Song et al. JCP (2022) 111443)
 * \param [in]         x_prev_buf         receive buffer for x_tilde from the previous rank
 * \param [in]         x_curr_buf         send buffer for x_tilde in the current rank
 * \param [in]         x_next_buf         receive buffer for x_tilde from the next rank
 * \param [in]         block_fact_prev    distributed factorization coefficients for the lower-diagonal block
 * \param [in]         block_fact_curr    distributed factorization coefficients for the diagonal block
 * \param [in]         block_fact_next    distributed factorization coefficients for the upper-diagonal block
 * \param [in]         N_batch            number of systems within the batch being solved concurrently, i.e., number of entries in the non-solve direction
 * \param [in]         Np                 total number of partitions in the solve direction
 * \param [in]         comm_sub           MPI sub-communicator of partitions in the solve direction for each reduced system
 * \param [in]         mpi_reqs           array of MPI requests
 * \param [in]         mpi_stats          array of MPI status
 * \param [in]         x_prev_buf_host    receive buffer on host for x_tilde from the previous rank
 * \param [in]         x_next_buf_host    receive buffer on host for x_tilde from the next rank
 * \param [in]         x_curr_buf_host    send buffer on host for x_tilde in the current rank
 * \note If host buffers are all nullptr then communication will be device-aware
 */
template<typename RealType, typename RealTypeComm = RealType, typename MemSpaceType = MemSpace::Device>
void distSolve (RealType* x_curr, RealTypeComm* x_prev_buf, RealTypeComm* x_curr_buf, RealTypeComm* x_next_buf,
        RealType* block_fact_prev, RealType* block_fact_curr, RealType* block_fact_next,
        const unsigned int N_batch, const unsigned int Np, MPI_Comm comm_sub, MPI_Request* mpi_reqs, MPI_Status* mpi_stats,
        RealTypeComm* x_prev_buf_host, RealTypeComm* x_curr_buf_host, RealTypeComm* x_next_buf_host)
{
#ifdef COMPACK3D_DEBUG_MODE_ENABLED
    {
        int mpi_size;
        MPI_Comm_size(comm_sub, &mpi_size);
        if (Np > mpi_size) {
            throw std::invalid_argument("Np is greater than maximum number of partitions involved in the reduced system.");
        }
    }
    assert(((x_prev_buf_host == nullptr) && (x_curr_buf_host == nullptr) && (x_next_buf_host == nullptr)) || (x_prev_buf_host && x_curr_buf_host && x_next_buf_host)); 
#endif

    #ifdef COMPACK3D_DEVICE_COMM_ENABLED
    constexpr bool USE_DEVICE_AWARE_COMM = true;
    #else
    constexpr bool USE_DEVICE_AWARE_COMM = std::is_same<MemSpaceType, MemSpace::Host>::value;
    #endif

    if constexpr (USE_DEVICE_AWARE_COMM) {
        x_prev_buf_host = x_prev_buf;
        x_curr_buf_host = x_curr_buf;
        x_next_buf_host = x_next_buf;
    }

    int rank_curr;
    MPI_Comm_rank(comm_sub, &rank_curr);
    const unsigned int NUM_COMM_ENTRIES = 2 * N_batch;

    int stride = 1; // NOTE: stride is also equal to the number of sub-systems
    int N_sub  = Np;
    int fact_idx_orig = 0;

    while (N_sub) {
        // Detach process
        if ((N_sub > 1) && (N_sub & 0b1)) {
            N_sub -= 1;
            const int detach_row_lo = N_sub * stride;
            const int detach_row_hi = detach_row_lo + stride;
            const int rank_prev = (rank_curr - stride + detach_row_hi) % detach_row_hi;
            const int rank_next = (rank_curr + stride                ) % detach_row_hi;
            RealType* fact_prev_step = &block_fact_prev[fact_idx_orig];
            RealType* fact_curr_step = &block_fact_curr[fact_idx_orig];
            RealType* fact_next_step = &block_fact_next[fact_idx_orig];

            if(detach_row_lo <= rank_prev && rank_prev < detach_row_hi) {
                if constexpr (USE_DEVICE_AWARE_COMM) deviceFence();
                MPI_Recv(x_prev_buf_host, NUM_COMM_ENTRIES, MPIDataType<RealTypeComm>::value, rank_prev, 451, comm_sub, &mpi_stats[2]);
                if constexpr (!USE_DEVICE_AWARE_COMM) deepCopy<MemSpaceType, MemSpace::Host, RealTypeComm>(x_prev_buf, x_prev_buf_host, NUM_COMM_ENTRIES);
                reduceCurrBlockOneSide<RealType, RealTypeComm>(x_curr, x_prev_buf, fact_curr_step, fact_prev_step, N_batch);
            }

            if(detach_row_lo <= rank_next && rank_next < detach_row_hi) {
                if constexpr (USE_DEVICE_AWARE_COMM) deviceFence();
                MPI_Recv(x_next_buf_host, NUM_COMM_ENTRIES, MPIDataType<RealTypeComm>::value, rank_next, 461, comm_sub, &mpi_stats[3]);
                if constexpr (!USE_DEVICE_AWARE_COMM) deepCopy<MemSpaceType, MemSpace::Host, RealTypeComm>(x_next_buf, x_next_buf_host, NUM_COMM_ENTRIES);
                reduceCurrBlockOneSide<RealType, RealTypeComm>(x_curr, x_next_buf, fact_curr_step, fact_next_step, N_batch);
            }

            if(detach_row_lo <= rank_curr && rank_curr < detach_row_hi) {
                copyAndCast2D<RealTypeComm, RealType>(x_curr_buf, x_curr, 2, N_batch, N_batch);
                if constexpr (!USE_DEVICE_AWARE_COMM) {
                    deepCopy<MemSpace::Host, MemSpaceType, RealTypeComm>(x_curr_buf_host, x_curr_buf, NUM_COMM_ENTRIES);
                } else {
                    deviceFence();
                }
                MPI_Isend(x_curr_buf_host, NUM_COMM_ENTRIES, MPIDataType<RealTypeComm>::value, rank_prev, 461, comm_sub, &mpi_reqs[0]);
                MPI_Isend(x_curr_buf_host, NUM_COMM_ENTRIES, MPIDataType<RealTypeComm>::value, rank_next, 451, comm_sub, &mpi_reqs[1]);
                MPI_Waitall(2, &mpi_reqs[0], &mpi_stats[0]);
            }

            fact_idx_orig += 4;
        } // if (Np & stride)

        // Regular PCR
        const int N_attached = N_sub * stride;
        const int rank_prev = (rank_curr - stride + N_attached) % N_attached;
        const int rank_next = (rank_curr + stride             ) % N_attached;
        RealType* fact_prev_step = &block_fact_prev[fact_idx_orig];
        RealType* fact_curr_step = &block_fact_curr[fact_idx_orig];
        RealType* fact_next_step = &block_fact_next[fact_idx_orig];

        if (rank_curr < N_attached) {
            if constexpr (USE_DEVICE_AWARE_COMM) {
                deviceFence();
            }
            MPI_Irecv(x_prev_buf_host, NUM_COMM_ENTRIES, MPIDataType<RealTypeComm>::value, rank_prev, 451, comm_sub, &mpi_reqs[0]);
            MPI_Irecv(x_next_buf_host, NUM_COMM_ENTRIES, MPIDataType<RealTypeComm>::value, rank_next, 461, comm_sub, &mpi_reqs[1]);
            copyAndCast2D<RealTypeComm, RealType>(x_curr_buf, x_curr, 2, N_batch, N_batch);
            if constexpr (!USE_DEVICE_AWARE_COMM) {
                deepCopy<MemSpace::Host, MemSpaceType, RealTypeComm>(x_curr_buf_host, x_curr_buf, NUM_COMM_ENTRIES);
            } else {
                deviceFence();
            }
            MPI_Isend(x_curr_buf_host, 2, MPIDataType<RealTypeComm>::value, rank_next, 451, comm_sub, &mpi_reqs[2]);
            MPI_Isend(x_curr_buf_host, 2, MPIDataType<RealTypeComm>::value, rank_prev, 461, comm_sub, &mpi_reqs[3]);
            MPI_Waitall(4, mpi_reqs, mpi_stats);
            if constexpr (!USE_DEVICE_AWARE_COMM) {
                deepCopy<MemSpaceType, MemSpace::Host, RealTypeComm>(x_next_buf, x_next_buf_host, NUM_COMM_ENTRIES);
                deepCopy<MemSpaceType, MemSpace::Host, RealTypeComm>(x_prev_buf, x_prev_buf_host, NUM_COMM_ENTRIES);
            }
            reduceCurrBlockSymm<RealType, RealTypeComm>(x_curr, x_prev_buf, x_next_buf, fact_curr_step, fact_prev_step, fact_next_step, N_batch);
        } // if (rank_curr < N_attached)

        stride <<= 1;
        N_sub  >>= 1;
        fact_idx_orig += 4;
    } // while ((stride << 1) <= Np)

    stride >>= 2;
    while (stride) {
        if ((Np > stride) && (Np & stride)) {
            // Reattach row floor(Np / (stride<<1)) * (stride<<1) + j for j in [0, stride)
            const int attach_row_lo = (Np / (stride << 1)) * (stride << 1);
            const int N_attached = attach_row_lo + stride;
            const int rank_prev = (rank_curr - stride + N_attached) % N_attached;
            const int rank_next = (rank_curr + stride             ) % N_attached;
            RealType* fact_prev_step = &block_fact_prev[fact_idx_orig];
            RealType* fact_curr_step = &block_fact_curr[fact_idx_orig];
            RealType* fact_next_step = &block_fact_next[fact_idx_orig];

            if(attach_row_lo <= rank_curr && rank_curr < N_attached) {
                if constexpr (USE_DEVICE_AWARE_COMM) deviceFence();
                MPI_Irecv(x_prev_buf_host, 2, MPIDataType<RealTypeComm>::value, rank_prev, 451, comm_sub, &mpi_reqs[0]);
                MPI_Irecv(x_next_buf_host, 2, MPIDataType<RealTypeComm>::value, rank_next, 461, comm_sub, &mpi_reqs[1]);
                MPI_Waitall(2, &mpi_reqs[0], &mpi_stats[0]);
                if constexpr (!USE_DEVICE_AWARE_COMM) {
                    deepCopy<MemSpaceType, MemSpace::Host>(x_next_buf, x_next_buf_host, NUM_COMM_ENTRIES);
                    deepCopy<MemSpaceType, MemSpace::Host>(x_prev_buf, x_prev_buf_host, NUM_COMM_ENTRIES);
                }
                reduceCurrBlockSymm<RealType, RealTypeComm>(x_curr, x_prev_buf, x_next_buf, fact_curr_step, fact_prev_step, fact_next_step, N_batch);
            }

            if(attach_row_lo <= rank_prev && rank_prev < N_attached) {
                copyAndCast2D<RealTypeComm, RealType>(x_curr_buf, x_curr, 2, N_batch, N_batch);
                if constexpr (!USE_DEVICE_AWARE_COMM) {
                    deepCopy<MemSpace::Host, MemSpaceType>(x_curr_buf_host, x_curr_buf, NUM_COMM_ENTRIES);
                } else {
                    deviceFence();
                }
                MPI_Send(x_curr_buf_host, 2, MPIDataType<RealTypeComm>::value, rank_prev, 461, comm_sub);
            }

            if(attach_row_lo <= rank_next && rank_next < N_attached) {
                copyAndCast2D<RealTypeComm, RealType>(x_curr_buf, x_curr, 2, N_batch, N_batch);
                if constexpr (!USE_DEVICE_AWARE_COMM) {
                    deepCopy<MemSpace::Host, MemSpaceType>(x_curr_buf_host, x_curr_buf, NUM_COMM_ENTRIES);
                } else {
                    deviceFence();
                }
                MPI_Send(x_curr_buf_host, 2, MPIDataType<RealTypeComm>::value, rank_next, 451, comm_sub);
            }

            fact_idx_orig += 4;
        } // if (Np & stride)
        stride >>= 1;
    } // while (stride)
}





/*!
 * Solving reduced system on distributed memory with performance
 * \param [in, out]    x_curr             right-hand side (b_hat) as input and solution (x_tilde) as output (see Fig.4 and Eq.(17) in Song et al. JCP (2022) 111443)
 * \param [in]         x_prev_buf         receive buffer for x_tilde from the previous rank
 * \param [in]         x_curr_buf         send buffer for x_tilde in the current rank
 * \param [in]         x_next_buf         receive buffer for x_tilde from the next rank
 * \param [in]         block_fact_prev    distributed factorization coefficients for the lower-diagonal block
 * \param [in]         block_fact_curr    distributed factorization coefficients for the diagonal block
 * \param [in]         block_fact_next    distributed factorization coefficients for the upper-diagonal block
 * \param [in]         N_batch            number of systems within the batch being solved concurrently, i.e., number of entries in the non-solve direction
 * \param [in]         Np                 total number of partitions in the solve direction
 * \param [in]         comm_sub           MPI sub-communicator of partitions in the solve direction for each reduced system
 * \param [in]         x_prev_buf_host    receive buffer on host for x_tilde from the previous rank
 * \param [in]         x_next_buf_host    receive buffer on host for x_tilde from the next rank
 * \param [in]         x_curr_buf_host    send buffer on host for x_tilde in the current rank
 * \note If host buffers are all nullptr then communication will be device-aware
 */
template<typename RealType, typename RealTypeComm = RealType, typename MemSpaceType>
void distSolve (RealType* x_curr, RealTypeComm* x_prev_buf, RealTypeComm* x_curr_buf, RealTypeComm* x_next_buf,
        RealType* block_fact_prev, RealType* block_fact_curr, RealType* block_fact_next,
        const unsigned int N_batch, const unsigned int Np, MPI_Comm comm_sub,
        RealTypeComm* x_prev_buf_host, RealTypeComm* x_curr_buf_host, RealTypeComm* x_next_buf_host)
{
#ifdef COMPACK3D_DEBUG_MODE_ENABLED
    {
        int mpi_size;
        MPI_Comm_size(comm_sub, &mpi_size);
        if (Np > mpi_size) {
            throw std::invalid_argument("Np is greater than maximum number of partitions involved in the reduced system.");
        }
    }
    assert(((x_prev_buf_host == nullptr) && (x_curr_buf_host == nullptr) && (x_next_buf_host == nullptr)) || (x_prev_buf_host && x_curr_buf_host && x_next_buf_host)); 
#endif
    MPI_Request mpi_reqs [4];
    MPI_Status  mpi_stats[4];
    distSolve<RealType, RealTypeComm, MemSpaceType>(
            x_curr,
            x_prev_buf, x_curr_buf, x_next_buf,
            block_fact_prev, block_fact_curr, block_fact_next,
            N_batch, Np, comm_sub, mpi_reqs, mpi_stats,
            x_prev_buf_host, x_curr_buf_host, x_next_buf_host
    );
}




////////////////////////////////////////
//////// EXPLICIT INSTANTIATION //////// 
////////////////////////////////////////
template void solFact<double>(double&, double&, double&, double&, double&, FactSysPenta<double>&, const int);
template void solFact< float>( float&,  float&,  float&,  float&,  float&, FactSysPenta< float>&, const int);

template void localFactPenta<double>(double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, const int, const int);
template void localFactPenta< float>( float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*, const int, const int);


template void vanillaLocalSolPentaPCR<double>(double*, const double*, const double*, const double*, const double*, const double*, const int, const int);
template void vanillaLocalSolPentaPCR< float>( float*, const  float*, const  float*, const  float*, const  float*, const  float*, const int, const int);

template void vanillaLocalSolPentaPCRBatch<double>(double*, const double*, const double*, const double*, const double*, const double*, const int, const int, const int);
template void vanillaLocalSolPentaPCRBatch< float>( float*, const  float*, const  float*, const  float*, const  float*, const  float*, const int, const int, const int);

template void mat2x2AXPY(double*, const double*, const double*, const double, const double);
template void mat2x2AXPY( float*, const  float*, const  float*, const  float, const  float);
    
template void mat2x2Inv(double*, const double*);
template void mat2x2Inv( float*, const  float*);

template void mat2x2AMultInvB(double*, const double*, const double*);
template void mat2x2AMultInvB( float*, const  float*, const  float*);

template void distFactPenta(double*, double*, double*, double*, const int, MPI_Comm);
template void distFactPenta( float*,  float*,  float*,  float*, const int, MPI_Comm);

template void factPartitionedPentaHost<double>(double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, const int, const int, const int, MPI_Comm);
template void factPartitionedPentaHost< float>( float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*, const int, const int, const int, MPI_Comm);

template void factPartitionedPenta<MemSpace::Host  , double>(double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, const unsigned int, const unsigned int, const unsigned int, MPI_Comm);
template void factPartitionedPenta<MemSpace::Host  ,  float>( float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*, const unsigned int, const unsigned int, const unsigned int, MPI_Comm);
template void factPartitionedPenta<MemSpace::Device, double>(double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, const unsigned int, const unsigned int, const unsigned int, MPI_Comm);
template void factPartitionedPenta<MemSpace::Device,  float>( float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*, const unsigned int, const unsigned int, const unsigned int, MPI_Comm);

template void allocFactBuffers<MemSpace::Host  , double>(double**, double**, double**, double**, double**, double**, double**, double**, double**, double**, double**, double**, const unsigned int, const unsigned int);
template void allocFactBuffers<MemSpace::Device, double>(double**, double**, double**, double**, double**, double**, double**, double**, double**, double**, double**, double**, const unsigned int, const unsigned int);
template void allocFactBuffers<MemSpace::Host  ,  float>( float**,  float**,  float**,  float**,  float**,  float**,  float**,  float**,  float**,  float**,  float**,  float**, const unsigned int, const unsigned int);
template void allocFactBuffers<MemSpace::Device,  float>( float**,  float**,  float**,  float**,  float**,  float**,  float**,  float**,  float**,  float**,  float**,  float**, const unsigned int, const unsigned int);

template void distSolve<double, double, MemSpace::Device>(double*, double*, double*, double*, double*, double*, double*, const unsigned int, const unsigned int, MPI_Comm, double*, double*, double*);
template void distSolve<double,  float, MemSpace::Device>(double*,  float*,  float*,  float*, double*, double*, double*, const unsigned int, const unsigned int, MPI_Comm,  float*,  float*,  float*);

template void distSolve<double, double, MemSpace::Device>(double*, double*, double*, double*, double*, double*, double*, const unsigned int, const unsigned int, MPI_Comm, MPI_Request*, MPI_Status*, double*, double*, double*);
template void distSolve<double,  float, MemSpace::Device>(double*,  float*,  float*,  float*, double*, double*, double*, const unsigned int, const unsigned int, MPI_Comm, MPI_Request*, MPI_Status*,  float*,  float*,  float*);

template void freeFactBuffers<MemSpace::Host  , double>(double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*);
template void freeFactBuffers<MemSpace::Device, double>(double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*);
template void freeFactBuffers<MemSpace::Host  ,  float>( float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*);
template void freeFactBuffers<MemSpace::Device,  float>( float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*,  float*);

template void vanillaDistSolve<double, double>(double*, double*, double*, double*, double*, double*, double*, const int, MPI_Comm);
} // namespace penta
} // namspace cmpk


#undef COMPACK3D_DEVICE_FENCE
