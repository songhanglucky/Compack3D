
#include "Compack3D_penta.h"
#include <stdexcept>
#include <cmath>

using namespace cmpk;
using namespace penta;

typedef double Real;

void resetSystem(Real* l2, Real* l1, Real* d, Real* u1, Real* u2, Real* x, const int N, const int num_batches) {
    for(int i = 0; i < N; i++) {
        l2[i] = (i > 1      ) * 0.05;
        l1[i] = (i > 0      ) * 0.50;
        u1[i] = (i < (N - 1)) * 0.50;
        u2[i] = (i < (N - 2)) * 0.05;
        d [i] = 1.0;
        for (int batch = 0; batch < num_batches; batch ++) {
            x [i + N * batch] = 1.0;
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

int main(int argc, char* argv[]) {
    (void) argc; (void) argv;

    const int N_loc            = 65;
    const int batch_size       = 2;
    const int max_sub_sys_size = 128;

    Real* l2 = new Real [N_loc];
    Real* l1 = new Real [N_loc];
    Real*  d = new Real [N_loc];
    Real* u1 = new Real [N_loc];
    Real* u2 = new Real [N_loc];

    Real* x_pcr  = new Real [N_loc * batch_size];
    Real* x_thm  = new Real [N_loc];

    const int n_steps = cmpk::log2Ceil(N_loc);
    Real* fact_prev_2 = new Real [N_loc * n_steps];
    Real* fact_prev_1 = new Real [N_loc * n_steps];
    Real* fact_curr   = new Real [N_loc * n_steps];
    Real* fact_next_1 = new Real [N_loc * n_steps];
    Real* fact_next_2 = new Real [N_loc * n_steps];

    resetSystem(l2, l1, d, u1, u2, x_thm, N_loc, 1);
    thomasSol(x_thm, l2, l1, d, u1, u2, N_loc);

    resetSystem(l2, l1, d, u1, u2, x_pcr, N_loc, batch_size);
    localFactPenta(fact_prev_2, fact_prev_1, fact_curr, fact_next_1, fact_next_2, l2, l1, d, u1, u2, N_loc, max_sub_sys_size);
    vanillaLocalSolPentaPCRBatch(x_pcr, fact_prev_2, fact_prev_1, fact_curr, fact_next_1, fact_next_2, N_loc, batch_size, max_sub_sys_size);

    printf("System size: %d\n", N_loc);
    for (int level = 0; level < n_steps; level++) {
        printf("Elimination stage %d:\n", level);
        printf("%4s, %12s, %12s, %12s, %12s, %12s\n", "row", "fact_prev_2", "fact_prev_1", "fact_curr", "fact_next_1", "fact_next_2");
        for (int i = 0; i < N_loc; i++) {
            const int idx_fact = level * N_loc + locFactIdx(i, N_loc, 1<<level, max_sub_sys_size);
            printf("%4d, %12.5e, %12.5e, %12.5e, %12.5e, %12.5e\n", i, fact_prev_2[idx_fact], fact_prev_1[idx_fact], fact_curr[idx_fact], fact_next_1[idx_fact], fact_next_2[idx_fact]);
        }
    }

    printf("%5s, %5s, %15s, %15s, %15s\n", "batch", "idx", "ref", "num", "dif");
    double err_tot = 0.0;
    for (int batch = 0; batch < batch_size; batch ++) {
        for (int i = 0; i < N_loc; i++) {
            const double dif = x_thm[i]-x_pcr[i + batch * N_loc];
            printf("%5d, %5d, %15.5E, %15.5E, %15.5E\n", batch, i, x_thm[i], x_pcr[i + batch * N_loc], dif);
            err_tot += dif * dif;
        }
    }
    printf("Total RMS error: %.5e\n", sqrt(err_tot / (N_loc * batch_size)));


    delete [] l2;
    delete [] l1;
    delete []  d;
    delete [] u1;
    delete [] u2;
    delete [] x_pcr;
    delete [] x_thm;
    delete [] fact_prev_2;
    delete [] fact_prev_1;
    delete [] fact_curr;
    delete [] fact_next_1;
    delete [] fact_next_2;

    return 0;
}
