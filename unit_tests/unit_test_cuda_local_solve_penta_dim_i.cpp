
#include "Compack3D_penta.h"
#include <stdexcept>
#include <cmath>
#include <cstdio>

using namespace cmpk;
using namespace penta;

typedef double Real;
typedef MemSpace::Device mem_space;

void resetSystem(Real* l2, Real* l1, Real* d, Real* u1, Real* u2, const int Ni) {
    for(int i = 0; i < Ni; i++) {
        l2  [i] = (i > 1       ) * 0.05;
        l1  [i] = (i > 0       ) * 0.50;
        u1  [i] = (i < (Ni - 1)) * 0.50;
        u2  [i] = (i < (Ni - 2)) * 0.05;
        d   [i] = 1.0;
    }

}



void resetRHS(Real* x, const int Ni, const int Nj, const int Nk) {
    for (int i = 0; i < Ni; i++) {
        const int offset_i = i * Nj * Nk;
        for (int j = 0; j < Nj; j++) {
            const int offset_j = j * Nk;
            for (int k = 0; k < Nk; k++) {
                x[offset_i + offset_j + k] = 1.0;
            }
        }
    }
}


void thomasSol(Real* x, Real* l2, Real* l1, Real* d, Real* u1, Real* u2, const int N) {
    for (int i = 0; i < (N - 1); i++) {
        double fact = l1[i+1] / d[i];
        d [i+1] -= fact * u1[i];
        x [i+1] -= fact * x [i];
        if ((i + 2) < N) {
            u1[i+1] -= fact * u2[i];
            fact = l2[i + 2] / d[i];
            l1[i+2] -= fact * u1[i];
            d [i+2] -= fact * u2[i];
            x [i+2] -= fact * x [i];
        }
    }
    for (int i = N - 1; i > -1; i--) {
        if ((i + 2) < N) x[i] -= u2[i] * x[i + 2];
        if ((i + 1) < N) x[i] -= u1[i] * x[i + 1];
        x[i] /= d[i];
    }
}


Real compareSolutions(Real* x1d, Real* x3d, const int Ni, const int Nj, const int Nk) {
    Real err  = 0.0;
    Real norm = 0.0;
    for (int i = 0; i < Ni; i++) {
        const int offset_i = i * Nj * Nk;
        const Real x_ref = x1d[i];
        for (int j = 0; j < Nj; j++) {
            const int offset_j = j * Nk;
            for (int k = 0; k < Nk; k++) {
                const Real err_loc = x3d[offset_i + offset_j + k] - x_ref;
                err  += err_loc * err_loc;
                norm += x_ref * x_ref;
            }
        }
    }
    return sqrt(err / norm);
}


int main(int argc, char* argv[]) {
    (void) argc; (void) argv;

    constexpr int Ni = 257;
    constexpr int Nj = 126;
    constexpr int Nk = 192;

    const int N_loc            = Ni;
    const int max_sub_sys_size = 16;

    Real* l2 = new Real [N_loc];
    Real* l1 = new Real [N_loc];
    Real*  d = new Real [N_loc];
    Real* u1 = new Real [N_loc];
    Real* u2 = new Real [N_loc];

    const int n_steps = cmpk::log2Ceil(N_loc);

    Real* fact_prev_2; cudaMalloc((void**) &fact_prev_2, N_loc * n_steps * sizeof(Real));
    Real* fact_prev_1; cudaMalloc((void**) &fact_prev_1, N_loc * n_steps * sizeof(Real));
    Real* fact_curr  ; cudaMalloc((void**) &fact_curr  , N_loc * n_steps * sizeof(Real));
    Real* fact_next_1; cudaMalloc((void**) &fact_next_1, N_loc * n_steps * sizeof(Real));
    Real* fact_next_2; cudaMalloc((void**) &fact_next_2, N_loc * n_steps * sizeof(Real));

    Real* fact_prev_2_host = new Real [N_loc * n_steps];
    Real* fact_prev_1_host = new Real [N_loc * n_steps];
    Real* fact_curr_host   = new Real [N_loc * n_steps];
    Real* fact_next_1_host = new Real [N_loc * n_steps];
    Real* fact_next_2_host = new Real [N_loc * n_steps];

    resetSystem(l2, l1, d, u1, u2, N_loc);
    localFactPenta(fact_prev_2_host, fact_prev_1_host, fact_curr_host, fact_next_1_host, fact_next_2_host, l2, l1, d, u1, u2, N_loc, max_sub_sys_size);
    cudaMemcpy(fact_prev_2, fact_prev_2_host, N_loc * n_steps * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(fact_prev_1, fact_prev_1_host, N_loc * n_steps * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(fact_curr  , fact_curr_host  , N_loc * n_steps * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(fact_next_1, fact_next_1_host, N_loc * n_steps * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(fact_next_2, fact_next_2_host, N_loc * n_steps * sizeof(Real), cudaMemcpyHostToDevice);
    delete [] fact_prev_2_host;
    delete [] fact_prev_1_host;
    delete [] fact_curr_host;
    delete [] fact_next_1_host;
    delete [] fact_next_2_host;

    Real* x_thm  = new Real [N_loc];
    resetSystem(l2, l1, d, u1, u2, N_loc);
    resetRHS(x_thm, 1, 1, N_loc);
    thomasSol(x_thm, l2, l1, d, u1, u2, N_loc);

    delete [] l2;
    delete [] l1;
    delete []  d;
    delete [] u1;
    delete [] u2;

    Real* x_pcr  = new Real [Ni * Nj * Nk];
    Real* x_dev; cudaMalloc((void**) &x_dev, Ni * Nj * Nk * sizeof(Real));
    resetRHS(x_pcr, Ni, Nj, Nk);
    cudaMemcpy(x_dev, x_pcr, Ni * Nj * Nk * sizeof(Real), cudaMemcpyHostToDevice);
    localSolPentaPCRDimI<Real>(x_dev, fact_prev_2, fact_prev_1, fact_curr, fact_next_1, fact_next_2, max_sub_sys_size, Ni, Nj, Nk, Nj*Nk, Nk);
    cudaMemcpy(x_pcr, x_dev, Ni * Nj * Nk * sizeof(Real), cudaMemcpyDeviceToHost);

    const Real err = compareSolutions(x_thm, x_pcr, Ni, Nj, Nk);
    printf("Error of a batch size (%d x %d x %d) is %.5e\n", Ni, Nj, Nk, err);

    delete [] x_pcr;
    delete [] x_thm;
    cudaFree(fact_prev_2);
    cudaFree(fact_prev_1);
    cudaFree(fact_curr  );
    cudaFree(fact_next_1);
    cudaFree(fact_next_2);
    cudaFree(x_dev);
    return 0;
}
