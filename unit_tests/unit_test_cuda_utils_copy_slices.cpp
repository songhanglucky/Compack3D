#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cstdio>
#include <cmath>
#include "Compack3D_utils_kernels.cuh"


#define cudaCheckError() {                                                                 \
    cudaDeviceSynchronize();                                                               \
    cudaError_t e=cudaPeekAtLastError();                                                   \
    if(e!=cudaSuccess) {                                                                   \
        printf("Cuda failure %s:%d: \"%s\"\n", __FILE__, __LINE__,cudaGetErrorString(e));  \
        exit(0);                                                                           \
    }                                                                                      \
}


typedef double Real;
using namespace cmpk;

int main(int argc, char* argv[]) {
    (void) argc; (void) argv;
    constexpr unsigned int Ni       = 65;
    constexpr unsigned int Nj       = 48;
    constexpr unsigned int Nk       = 16;
    constexpr unsigned int N_slices = 2;

    printf("Array size: %u x %u x %u.\nNumber of slices: %u\n", Ni, Nj, Nk, N_slices);
    cudaDeviceSynchronize();
    
    const unsigned int arr_stride_i = Nj * Nk;
    const unsigned int arr_stride_j =      Nk;
    Real*  X_dev; cudaMalloc((void**) & X_dev, Ni * Nj * Nk * sizeof(Real));
    Real* Xi_dev; cudaMalloc((void**) &Xi_dev, N_slices * Nj * Nk * sizeof(Real));
    Real* Xj_dev; cudaMalloc((void**) &Xj_dev, N_slices * Ni * Nk * sizeof(Real));
    Real* Xk_dev; cudaMalloc((void**) &Xk_dev, N_slices * Ni * Nj * sizeof(Real));

    Real*  X_host = new Real[Ni * Nj * Nk];
    Real* Xi_host = new Real[N_slices * Nj * Nk];
    Real* Xj_host = new Real[N_slices * Nk * Ni];
    Real* Xk_host = new Real[N_slices * Ni * Nj];


    /************************/
    /*** Test dimension i ***/
    /************************/
    for (unsigned int i = 0; i < Ni; i++) {
        for (unsigned int j = 0; j < Nj; j++) {
            for (unsigned int k = 0; k < Nk; k++) {
                X_host[i * arr_stride_i + j * arr_stride_j + k] = (Real) i;
            }
        }
    }
    cudaMemcpy(X_dev, X_host, Ni * Nj * Nk * sizeof(Real), cudaMemcpyHostToDevice);
    cudaCheckError();
    copySlicesFrom3DArrayDimI(Xi_dev, X_dev, N_slices, Nj, Nk, arr_stride_i, arr_stride_j);
    cudaCheckError();
    cudaMemcpy(Xi_host, Xi_dev, N_slices * Nj * Nk * sizeof(Real), cudaMemcpyDeviceToHost);
    Real err_i = 0.0;
    for (unsigned int i = 0; i < N_slices; i++) {
        for (unsigned int j = 0; j < Nj; j++) {
            for (unsigned int k = 0; k < Nk; k++) {
                const Real err = fabs(Xi_host[i * (Nj * Nk) + j * Nk + k] - (Real) i);
                err_i += err;
                if (!(err < 1e-16)) printf("ERROR: (%2u, %2u, %2u) Ref = %u, Num = %12.5e\n", i, j, k, i, Xi_host[i * arr_stride_i + j * arr_stride_j + k]);
            }
        }
    }
    printf("Error in i: %12.5e\n", err_i);
    

    /************************/
    /*** Test dimension j ***/
    /************************/
    for (unsigned int i = 0; i < Ni; i++) {
        for (unsigned int j = 0; j < Nj; j++) {
            for (unsigned int k = 0; k < Nk; k++) {
                X_host[i * arr_stride_i + j * arr_stride_j + k] = (Real) j;
            }
        }
    }
    cudaMemcpy(X_dev, X_host, Ni * Nj * Nk * sizeof(Real), cudaMemcpyHostToDevice);
    cudaCheckError();
    copySlicesFrom3DArrayDimJ(Xj_dev, X_dev, Ni, N_slices, Nk, arr_stride_i, arr_stride_j);
    cudaCheckError();
    cudaMemcpy(Xj_host, Xj_dev, N_slices * Ni * Nk * sizeof(Real), cudaMemcpyDeviceToHost);
    Real err_j = 0.0;
    for (unsigned int j = 0; j < N_slices; j++) {
        for (unsigned int i = 0; i < Ni; i++) {
            for (unsigned int k = 0; k < Nk; k++) {
                const Real err = fabs(Xj_host[j * (Ni * Nk) + i * Nk + k] - (Real) j);
                err_j += err;
                if (!(err < 1e-16)) printf("ERROR: (%2u, %2u, %2u) Ref = %u, Num = %12.5e\n", i, j, k, j, Xj_host[j * (Ni * Nk) + i * Nk + k]);
            }
        }
    }
    printf("Error in j: %12.5e\n", err_j);
    

    /************************/
    /*** Test dimension k ***/
    /************************/
    for (unsigned int i = 0; i < Ni; i++) {
        for (unsigned int j = 0; j < Nj; j++) {
            for (unsigned int k = 0; k < Nk; k++) {
                X_host[i * arr_stride_i + j * arr_stride_j + k] = (Real) k;
            }
        }
    }
    cudaMemcpy(X_dev, X_host, Ni * Nj * Nk * sizeof(Real), cudaMemcpyHostToDevice);
    cudaCheckError();
    copySlicesFrom3DArrayDimK(Xk_dev, X_dev, Ni, Nj, N_slices, arr_stride_i, arr_stride_j);
    cudaCheckError();
    cudaMemcpy(Xk_host, Xk_dev, N_slices * Ni * Nj * sizeof(Real), cudaMemcpyDeviceToHost);
    Real err_k = 0.0;
    for (unsigned int k = 0; k < N_slices; k++) {
        for (unsigned int i = 0; i < Ni; i++) {
            for (unsigned int j = 0; j < Nj; j++) {
                const Real err = fabs(Xk_host[k * (Ni * Nj) + i * Nj + j] - (Real) k);
                err_k += err;
                if (!(err < 1e-16)) printf("ERROR: (%2u, %2u, %2u) Ref = %u, Num = %12.5e\n", i, j, k, k, Xk_host[k * (Ni * Nj) + i * Nj + j]);
            }
        }
    }
    printf("Error in k: %12.5e\n", err_k);
    



    delete []  X_host;  X_host = nullptr;
    delete [] Xi_host; Xi_host = nullptr;
    delete [] Xj_host; Xj_host = nullptr;
    delete [] Xk_host; Xk_host = nullptr;
    cudaFree( X_dev);
    cudaFree(Xi_dev);
    cudaFree(Xj_dev);
    cudaFree(Xk_dev);

    return 0;
}
