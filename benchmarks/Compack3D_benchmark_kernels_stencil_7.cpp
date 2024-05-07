#include "Compack3D_api.h"
#include "Compack3D_stencil_api.h"
#include <cmath>

typedef double Real;

using namespace cmpk;
using namespace stencil;


int main(int argc, char* argv[]) {
    (void) argc; (void) argv;

    //////////////////////////
    // INPUT CONFIGURATIONS //
    //////////////////////////
    constexpr unsigned int NUM_GRIDS = 256;

    constexpr Real DOMAIN_LENGTH = 2.0 * 3.1415926535897932384626433832795;
    constexpr Real dx = DOMAIN_LENGTH / NUM_GRIDS;

    constexpr Real COEF_A =  (3.0 /  4.0) / dx;
    constexpr Real COEF_B = -(3.0 / 20.0) / dx;
    constexpr Real COEF_C =  (1.0 / 60.0) / dx;

    constexpr Real K_WAVE = 2.0;

    //////////////////////////////////
    // INITIALIZATIONS OF UTILITIES //
    //////////////////////////////////

    Real* f_local      = nullptr;
    Real* df_local     = nullptr;
    Real* f_halo_prev  = nullptr;
    Real* f_halo_next  = nullptr;
    Real* f_local_host = nullptr;
    Real* f_prev_host  = nullptr;
    Real* f_next_host  = nullptr;

    memAllocArray<MemSpace::Device, Real>(&f_local     , NUM_GRIDS * NUM_GRIDS * NUM_GRIDS);
    memAllocArray<MemSpace::Device, Real>(&df_local    , NUM_GRIDS * NUM_GRIDS * NUM_GRIDS);
    memAllocArray<MemSpace::Device, Real>(&f_halo_prev ,         3 * NUM_GRIDS * NUM_GRIDS);
    memAllocArray<MemSpace::Device, Real>(&f_halo_next ,         3 * NUM_GRIDS * NUM_GRIDS);
    memAllocArray<MemSpace::Host  , Real>(&f_local_host, NUM_GRIDS * NUM_GRIDS * NUM_GRIDS);
    memAllocArray<MemSpace::Host  , Real>(&f_prev_host ,         3 * NUM_GRIDS * NUM_GRIDS);
    memAllocArray<MemSpace::Host  , Real>(&f_next_host ,         3 * NUM_GRIDS * NUM_GRIDS);

    Real* coef_a = nullptr;
    Real* coef_b = nullptr;
    Real* coef_c = nullptr;

    memAllocArray<MemSpace::Device, Real>(&coef_a, NUM_GRIDS);
    memAllocArray<MemSpace::Device, Real>(&coef_b, NUM_GRIDS);
    memAllocArray<MemSpace::Device, Real>(&coef_c, NUM_GRIDS);


    ////////////////////////////
    // CONFIGURE COEFFICIENTS //
    ////////////////////////////
    {
        Real* coef_a_host = nullptr;
        Real* coef_b_host = nullptr;
        Real* coef_c_host = nullptr;
        memAllocArray<MemSpace::Host, Real>(&coef_a_host, NUM_GRIDS);
        memAllocArray<MemSpace::Host, Real>(&coef_b_host, NUM_GRIDS);
        memAllocArray<MemSpace::Host, Real>(&coef_c_host, NUM_GRIDS);
        for (unsigned int i = 0; i < NUM_GRIDS; i++) {
            coef_a_host[i] = COEF_A;
            coef_b_host[i] = COEF_B;
            coef_c_host[i] = COEF_C;
        }
        deepCopy<MemSpace::Device, MemSpace::Host, Real>(coef_a, coef_a_host, NUM_GRIDS);
        deepCopy<MemSpace::Device, MemSpace::Host, Real>(coef_b, coef_b_host, NUM_GRIDS);
        deepCopy<MemSpace::Device, MemSpace::Host, Real>(coef_c, coef_c_host, NUM_GRIDS);
        memFreeArray<MemSpace::Host, Real>(coef_a_host);
        memFreeArray<MemSpace::Host, Real>(coef_b_host);
        memFreeArray<MemSpace::Host, Real>(coef_c_host);
        coef_a_host = nullptr;
        coef_b_host = nullptr;
        coef_c_host = nullptr;
    }


    ////////////////////////////
    // CONFIGURE INPUT FIELDS //
    ////////////////////////////
    for (unsigned int i = 0; i < NUM_GRIDS; i++) {
        const Real x = i * dx;
        for (unsigned int j = 0; j < NUM_GRIDS; j++) {
            const Real y = j * dx;
            for (unsigned int k = 0; k < NUM_GRIDS; k++) {
                const Real z = k * dx;
                f_local_host[i * NUM_GRIDS * NUM_GRIDS + j * NUM_GRIDS + k] = cos(K_WAVE * x) * cos(K_WAVE * y) * cos(K_WAVE * z);
            }
        }
    }

    for (int i = 0; i < 3; i++) {
        const Real x_prev = (i -         3) * dx;
        const Real x_next = (i + NUM_GRIDS) * dx;
        for (unsigned int j = 0; j < NUM_GRIDS; j++) {
            const Real y = j * dx;
            for (unsigned int k = 0; k < NUM_GRIDS; k++) {
                const Real z = k * dx;
                f_prev_host[i * NUM_GRIDS * NUM_GRIDS + j * NUM_GRIDS + k] = cos(K_WAVE * x_prev) * cos(K_WAVE * y) * cos(K_WAVE * z);
                f_next_host[i * NUM_GRIDS * NUM_GRIDS + j * NUM_GRIDS + k] = cos(K_WAVE * x_next) * cos(K_WAVE * y) * cos(K_WAVE * z);
            }
        }
    }

    deepCopy<MemSpace::Device, MemSpace::Host, Real>(f_local    , f_local_host, NUM_GRIDS * NUM_GRIDS * NUM_GRIDS);
    deepCopy<MemSpace::Device, MemSpace::Host, Real>(f_halo_prev, f_prev_host ,         3 * NUM_GRIDS * NUM_GRIDS);
    deepCopy<MemSpace::Device, MemSpace::Host, Real>(f_halo_next, f_next_host ,         3 * NUM_GRIDS * NUM_GRIDS);


    ///////////////////////
    // CONDUCT UNIT TEST //
    ///////////////////////
    centralDiffCollStencil7DimI<Real>(df_local, f_local, f_halo_prev, f_halo_next, coef_a, coef_b, coef_c,
            NUM_GRIDS, NUM_GRIDS, NUM_GRIDS, NUM_GRIDS * NUM_GRIDS, NUM_GRIDS);

    centralDiffCollStencil7DimJ<Real>(df_local, f_local, f_halo_prev, f_halo_next, coef_a, coef_b, coef_c,
            NUM_GRIDS, NUM_GRIDS, NUM_GRIDS, NUM_GRIDS * NUM_GRIDS, NUM_GRIDS);

    centralDiffCollStencil7DimK<Real>(df_local, f_local, f_halo_prev, f_halo_next, coef_a, coef_b, coef_c,
            NUM_GRIDS, NUM_GRIDS, NUM_GRIDS, NUM_GRIDS * NUM_GRIDS, NUM_GRIDS);

    centralDiffCollStencil7DimI<Real>(df_local, f_local, f_halo_prev, f_halo_next, coef_a, coef_b, coef_c,
            NUM_GRIDS, NUM_GRIDS, NUM_GRIDS, NUM_GRIDS * NUM_GRIDS, NUM_GRIDS);

    centralDiffCollStencil7DimJ<Real>(df_local, f_local, f_halo_prev, f_halo_next, coef_a, coef_b, coef_c,
            NUM_GRIDS, NUM_GRIDS, NUM_GRIDS, NUM_GRIDS * NUM_GRIDS, NUM_GRIDS);

    centralDiffCollStencil7DimK<Real>(df_local, f_local, f_halo_prev, f_halo_next, coef_a, coef_b, coef_c,
            NUM_GRIDS, NUM_GRIDS, NUM_GRIDS, NUM_GRIDS * NUM_GRIDS, NUM_GRIDS);



    //////////////////
    // FINALIZATION //
    //////////////////
    memFreeArray<MemSpace::Device, Real>(f_local     );
    memFreeArray<MemSpace::Device, Real>(df_local    );
    memFreeArray<MemSpace::Device, Real>(f_halo_prev );
    memFreeArray<MemSpace::Device, Real>(f_halo_next );
    memFreeArray<MemSpace::Host  , Real>(f_local_host);
    memFreeArray<MemSpace::Host  , Real>(f_prev_host );
    memFreeArray<MemSpace::Host  , Real>(f_next_host );
    f_local      = nullptr;
    df_local     = nullptr;
    f_halo_prev  = nullptr;
    f_halo_next  = nullptr;
    f_local_host = nullptr;
    f_prev_host  = nullptr;
    f_next_host  = nullptr;
    return 0;
}
