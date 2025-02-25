
#include "Compack3D_tri.h"
#include <stdexcept>

using namespace cmpk;
using namespace tri;

typedef double Real;

void resetSystem(Real* l, Real* d, Real* u, Real* x, const int N) {
    for(int i = 0; i < N; i++) {
        l[i] = (i > 0      ) * (1.0/3.0);
        u[i] = (i < (N - 1)) * (1.0/3.0);
        d [i] = 1.0;
        x [i] = 1.0;
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

int main(int argc, char* argv[]) {
    (void) argc; (void) argv;

    const int N_loc = 15;
    const int max_sub_sys_size = 16;

    Real* l = new Real [N_loc];
    Real* d = new Real [N_loc];
    Real* u = new Real [N_loc];

    Real* x_pcr  = new Real [N_loc];
    Real* x_thm  = new Real [N_loc];

    const int n_steps = cmpk::log2Ceil(N_loc);
    Real* fact_prev = new Real [N_loc * n_steps];
    Real* fact_curr = new Real [N_loc * n_steps];
    Real* fact_next = new Real [N_loc * n_steps];

    resetSystem(l, d, u, x_thm, N_loc);
    thomasSol(x_thm, l, d, u, N_loc);

    resetSystem(l, d, u, x_pcr, N_loc);
    localFactTri(fact_prev, fact_curr, fact_next, l, d, u, N_loc, max_sub_sys_size);
    vanillaLocalSolTriPCR(x_pcr, fact_prev, fact_curr, fact_next, N_loc, max_sub_sys_size);

    printf("System size: %d\n", N_loc);
    for (int level = 0; level < n_steps; level++) {
        printf("Elimination stage %d:\n", level);
        printf("%4s, %12s, %12s, %12s\n", "row", "fact_prev", "fact_curr", "fact_next");
        for (int i = 0; i < N_loc; i++) {
            const int idx_fact = level * N_loc + locFactIdx(i, N_loc, 1<<level, max_sub_sys_size);
            printf("%4d, %12.5e, %12.5e, %12.5e\n", i, fact_prev[idx_fact], fact_curr[idx_fact], fact_next[idx_fact]);
        }
    }

    printf("%4s, %15s, %15s, %15s\n", "idx", "ref", "num", "dif");
    for (int i = 0; i < N_loc; i++) printf("%4d, %15.5E, %15.5E, %15.5E\n", i, x_thm[i], x_pcr[i], x_thm[i]-x_pcr[i]);


    delete [] l;
    delete [] d;
    delete [] u;
    delete [] x_pcr;
    delete [] x_thm;
    delete [] fact_prev;
    delete [] fact_curr;
    delete [] fact_next;

    return 0;
}
