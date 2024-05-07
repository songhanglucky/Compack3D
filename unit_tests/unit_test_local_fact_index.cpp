
#include <stdexcept>
#include "Compack3D_penta.h"

using namespace cmpk;
using namespace penta;

bool printResult(const int N, const int* ref, int* sol) {
    bool err = false;
    printf("Results:\n");
    printf("%3s: ", "idx");
    for (int i = 0; i < N; i++) printf("%4d", i);
    printf("\n%3s: ", "ref");
    for (int i = 0; i < N; i++) printf("%4d", ref[i]);
    printf("\n%3s: ", "sol");
    for (int i = 0; i < N; i++) printf("%4d", sol[i]);
    printf("\nerr: ");
    for (int i = 0; i < N; i++) {
        if (ref[i] == sol[i]) printf("%4s", "o");
        else {
            printf("%4s", "X");
            err = true;
        }
    }
    printf("\n");
    return err;
}

int main(int argc, char* argv[]) {
    (void) argc; (void) argv;

    const int idx_ref_0 [21] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    const int idx_ref_1 [21] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
    const int idx_ref_2 [21] = {0, 4, 8, 12, 16, 20, 2, 6, 10, 14, 18, 1, 5, 9, 13, 17, 3, 7, 11, 15, 19};
    const int idx_ref_3 [21] = {0, 8, 16, 4, 12, 20, 2, 10, 18, 6, 14, 1, 9, 17, 5, 13, 3, 11, 19, 7, 15};
    int idx_sol[21];

    for (int i = 0; i < 21; i++) idx_sol[ locFactIdx(i, 21,  1,  4) ] = i;
    if (printResult(21, idx_ref_0, idx_sol)) throw std::runtime_error("Incorrect result!");
    for (int i = 0; i < 21; i++) idx_sol[ locFactIdx(i, 21,  2, 21) ] = i;
    if (printResult(21, idx_ref_0, idx_sol)) throw std::runtime_error("Incorrect result!");
    for (int i = 0; i < 21; i++) idx_sol[ locFactIdx(i, 21,  2,  4) ] = i;
    if (printResult(21, idx_ref_1, idx_sol)) throw std::runtime_error("Incorrect result!");
    for (int i = 0; i < 21; i++) idx_sol[ locFactIdx(i, 21,  4,  6) ] = i;
    if (printResult(21, idx_ref_2, idx_sol)) throw std::runtime_error("Incorrect result!");
    for (int i = 0; i < 21; i++) idx_sol[ locFactIdx(i, 21,  8,  4) ] = i;
    if (printResult(21, idx_ref_3, idx_sol)) throw std::runtime_error("Incorrect result!");

    return 0;
}
