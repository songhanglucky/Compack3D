
#include <cstdio>
#include <cassert>
#include <stdexcept>
#include <cmath>
#include "Compack3D_penta.h"

using namespace cmpk;
using namespace penta;

void resetPentaSys(FactSysPenta<double>& A) {
    constexpr double d = 1.00;
    constexpr double a = 0.50;
    constexpr double b = 0.05;
    A.prev2[0] = a;
    A.prev2[1] = a;
    A.prev2[2] = b;
    A.prev1[0] = b;
    A.prev1[1] = d;
    A.prev1[2] = a;
    A.prev1[3] = b;
    A.curr [0] = a;
    A.curr [1] = d;
    A.curr [2] = a;
    A.next1[0] = b;
    A.next1[1] = a;
    A.next1[2] = d;
    A.next1[3] = b;
    A.next2[0] = b;
    A.next2[1] = a;
    A.next2[2] = a;
}


void resetRefSoln(double& x0, double& x1, double& x2, double& x3, double& x4, const int row_label) {
    assert( (row_label & 0b00100) > 0 );
    if        (row_label == 0b11111) {
        x0 =  9.90099009900990146e-02;
        x1 = -9.90099009900990090e-01;
        x2 =  1.98019801980198018e+00;
        x3 = -9.90099009900990090e-01;
        x4 =  9.90099009900990285e-02;

    } else if (row_label == 0b01111) {
        x0 =  0.00000000000000000e+00;
        x1 = -9.23313670171838741e-01;
        x2 =  1.94408822775070500e+00;
        x3 = -9.74608874070274189e-01;
        x4 =  9.74608874070274078e-02;

    } else if (row_label == 0b11110) {
        x4 =  0.00000000000000000e+00;
        x3 = -9.23313670171838741e-01;
        x2 =  1.94408822775070500e+00;
        x1 = -9.74608874070274189e-01;
        x0 =  9.74608874070274078e-02;

    } else if (row_label == 0b00111) {
        x0 =  0.00000000000000000e+00;
        x1 =  0.00000000000000000e+00;
        x2 =  1.35231316725978656e+00;
        x3 = -7.11743772241992922e-01;
        x4 =  7.11743772241992950e-02;

    } else if (row_label == 0b11100) {
        x4 =  0.00000000000000000e+00;
        x3 =  0.00000000000000000e+00;
        x2 =  1.35231316725978656e+00;
        x1 = -7.11743772241992922e-01;
        x0 =  7.11743772241992950e-02;

    } else if (row_label == 0b01110) {
        x0 =  0.00000000000000000e+00;
        x1 = -9.09090909090909061e-01;
        x2 =  1.90909090909090895e+00;
        x3 = -9.09090909090909061e-01;
        x4 =  0.00000000000000000e+00;

    } else if (row_label == 0b01100) {
        x4 =  0.00000000000000000e+00;
        x3 =  0.00000000000000000e+00;
        x2 =  1.33333333333333326e+00;
        x1 = -6.66666666666666630e-01;
        x0 =  0.00000000000000000e+00;

    } else if (row_label == 0b00110) {
        x0 =  0.00000000000000000e+00;
        x1 =  0.00000000000000000e+00;
        x2 =  1.33333333333333326e+00;
        x3 = -6.66666666666666630e-01;
        x4 =  0.00000000000000000e+00;

    } else if (row_label == 0b00100) {
        x0 = 0.0;
        x1 = 0.0;
        x2 = 1.0;
        x3 = 0.0;
        x4 = 0.0;

    } else {
        throw std::invalid_argument("row_label is not supported");
    }
}


void checkAnswer(FactSysPenta<double>& A, const int row_label) {
    constexpr double tol = 1e-15;
    double x0, x1, x2, x3, x4;
    double a0, a1, a2, a3, a4;
    resetPentaSys(A);
    solFact     (x0, x1, x2, x3, x4, A, row_label);
    resetRefSoln(a0, a1, a2, a3, a4,    row_label);
    printf("Ref Soln: %23.16e, %23.16e, %23.16e, %23.16e, %23.16e\n", a0, a1, a2, a3, a4);
    printf("Num Soln: %23.16e, %23.16e, %23.16e, %23.16e, %23.16e\n", x0, x1, x2, x3, x4);
    a0 -= x0;
    a1 -= x1;
    a2 -= x2;
    a3 -= x3;
    a4 -= x4;
    printf(" Error  : %23.16e, %23.16e, %23.16e, %23.16e, %23.16e\n\n", a0, a1, a2, a3, a4);
    if (sqrt((a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3 + a4 * a4) / 5.0) > tol) {
        throw std::runtime_error("Check failed. The solution is incorrect.");
    }
}


int main(int argc, char* argv[]) {
    (void) argc; (void) argv;
    FactSysPenta<double> A;
    checkAnswer(A, 0b11111);
    checkAnswer(A, 0b01111);
    checkAnswer(A, 0b11110);
    checkAnswer(A, 0b00111);
    checkAnswer(A, 0b01110);
    checkAnswer(A, 0b11100);
    checkAnswer(A, 0b00110);
    checkAnswer(A, 0b01100);
    checkAnswer(A, 0b00100);

    return 0;
}
