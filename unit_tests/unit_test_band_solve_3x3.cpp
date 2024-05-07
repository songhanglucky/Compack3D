
#include <cstdio>
#include <cassert>
#include <stdexcept>
#include <cmath>
#include "Compack3D_penta.h"

using namespace cmpk;
using namespace penta;

void resetTriSys(FactSysTri<double>& A) {
    constexpr double d = 1.0;
    constexpr double o = 1.0 / 3.0;
    A.prev[0] = d;
    A.prev[1] = o;
    A.curr[0] = o;
    A.curr[1] = d;
    A.curr[2] = o;
    A.next[0] = o;
    A.next[1] = d;
}


void resetRefSoln(double& x0, double& x1, double& x2, const int row_label) {
    assert( (row_label & 0b010) > 0 );
    if        (row_label == 0b111) {
        x0 = -4.2857142857142860e-01;
        x1 =  1.2857142857142858e+00;
        x2 = -4.2857142857142855e-01;
    } else if (row_label == 0b110) {
        x0 = -0.375;
        x1 =  1.125;
        x2 =  0.000;
    } else if (row_label == 0b011) {
        x0 =  0.000;
        x1 =  1.125;
        x2 = -0.375;
    } else {
        x0 = 0.0;
        x1 = 1.0;
        x2 = 0.0;
    }
}


void checkAnswer(FactSysTri<double>& A, const int row_label) {
    constexpr double tol = 1e-15;
    double x0, x1, x2;
    double a0, a1, a2;
    resetTriSys(A);
    solFact(x0, x1, x2, A, row_label);
    resetRefSoln(a0, a1, a2, row_label);
    printf("Ref Soln: %23.16e, %23.16e, %23.16e\n", a0, a1, a2);
    printf("Num Soln: %23.16e, %23.16e, %23.16e\n", x0, x1, x2);
    a0 -= x0;
    a1 -= x1;
    a2 -= x2;
    printf(" Error  : %23.16e, %23.16e, %23.16e\n\n", a0, a1, a2);
    if (sqrt((a0 * a0 + a1 * a1 + a2 * a2) / 3.0) > tol) {
        throw std::runtime_error("Check failed. The solution is incorrect.");
    }
}


int main(int argc, char* argv[]) {
    (void) argc; (void) argv;
    FactSysTri<double> A;
    checkAnswer(A, 0b111);
    checkAnswer(A, 0b011);
    checkAnswer(A, 0b110);

    return 0;
}
