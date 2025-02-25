
#include "Compack3D_tri.h"
#include <stdexcept>
#include <cmath>
#include <cstdio>

using namespace cmpk;
using namespace tri;

typedef double Real;
typedef MemSpace::Device mem_space;

void resetSystem(Real* l, Real* d, Real* u, const int Ni) {
    for(int i = 0; i < Ni; i++) {
        l[i] = (i > 0       ) * (1.0/3.0);
        u[i] = (i < (Ni - 1)) * (1.0/3.0);
        d[i] = 1.0;
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



void thomasSol(Real* x, Real* l, Real* d, Real* u, const int N) {
    for (int i = 0; i < (N - 1); i++) {
        double fact = l[i+1] / d[i];
        d [i+1] -= fact * u[i];
        x [i+1] -= fact * x[i];
    }
    for (int i = N - 1; i > -1; i--) {
        if ((i+1) < N) x[i] -= u[i] * x[i+1];
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

    Real* l = new Real [N_loc];
    Real* d = new Real [N_loc];
    Real* u = new Real [N_loc];

    const int n_steps = cmpk::log2Ceil(N_loc);

    Real* fact_prev; cudaMalloc((void**) &fact_prev, N_loc * n_steps * sizeof(Real));
    Real* fact_curr; cudaMalloc((void**) &fact_curr, N_loc * n_steps * sizeof(Real));
    Real* fact_next; cudaMalloc((void**) &fact_next, N_loc * n_steps * sizeof(Real));

    Real* fact_prev_host = new Real [N_loc * n_steps];
    Real* fact_curr_host = new Real [N_loc * n_steps];
    Real* fact_next_host = new Real [N_loc * n_steps];

    resetSystem(l, d, u, N_loc);
    localFactTri(fact_prev_host, fact_curr_host, fact_next_host, l, d, u, N_loc, max_sub_sys_size);
    cudaMemcpy(fact_prev, fact_prev_host, N_loc * n_steps * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(fact_curr, fact_curr_host, N_loc * n_steps * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(fact_next, fact_next_host, N_loc * n_steps * sizeof(Real), cudaMemcpyHostToDevice);
    delete [] fact_prev_host;
    delete [] fact_curr_host;
    delete [] fact_next_host;

    Real* x_thm  = new Real [N_loc];
    resetSystem(l, d, u, N_loc);
    resetRHS(x_thm, 1, 1, N_loc);
    thomasSol(x_thm, l, d, u, N_loc);

    delete [] l;
    delete [] d;
    delete [] u;

    Real* x_pcr  = new Real [Ni * Nj * Nk];
    Real* x_dev; cudaMalloc((void**) &x_dev, Ni * Nj * Nk * sizeof(Real));
    resetRHS(x_pcr, Ni, Nj, Nk);
    cudaMemcpy(x_dev, x_pcr, Ni * Nj * Nk * sizeof(Real), cudaMemcpyHostToDevice);
    localSolTriPCRDimI<Real>(x_dev, fact_prev, fact_curr, fact_next, max_sub_sys_size, Ni, Nj, Nk, Nj*Nk, Nk);
    cudaMemcpy(x_pcr, x_dev, Ni * Nj * Nk * sizeof(Real), cudaMemcpyDeviceToHost);

    const Real err = compareSolutions(x_thm, x_pcr, Ni, Nj, Nk);
    printf("Error of a batch size (%d x %d x %d) is %.5e\n", Ni, Nj, Nk, err);

    delete [] x_pcr;
    delete [] x_thm;
    cudaFree(fact_prev);
    cudaFree(fact_curr);
    cudaFree(fact_next);
    cudaFree(x_dev);
    return 0;
}
