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
    constexpr unsigned int NUM_GRIDS_I = 249;
    constexpr unsigned int NUM_GRIDS_J = 253;
    constexpr unsigned int NUM_GRIDS_K = 253;

    constexpr Real DOMAIN_LENGTH = 2.0 * 3.1415926535897932384626433832795;
    constexpr Real dx = DOMAIN_LENGTH / NUM_GRIDS_I;
    constexpr Real dy = DOMAIN_LENGTH / NUM_GRIDS_J;
    constexpr Real dz = DOMAIN_LENGTH / NUM_GRIDS_K;

    constexpr Real COEF_A =  (3.0 /  4.0) / dz;
    constexpr Real COEF_B = -(3.0 / 20.0) / dz;
    constexpr Real COEF_C =  (1.0 / 60.0) / dz;

    constexpr Real K_WAVE = 2.0;
    constexpr Real K_MOD  = 2.0 * COEF_A * sin(K_WAVE * dz) + 2.0 * COEF_B * sin(2.0 * K_WAVE * dz) + 2.0 * COEF_C * sin(3.0 * K_WAVE * dz);

    //////////////////////////////////
    // INITIALIZATIONS OF UTILITIES //
    //////////////////////////////////
    //MPI_Init(NULL, NULL);
    //MPI_Comm comm_sub = MPI_COMM_WORLD;
    //int rank, Np;
    //MPI_Comm_rank(comm_sub, &rank);
    //MPI_Comm_size(comm_sub,   &Np);

    Real* f_local      = nullptr;
    Real* df_local     = nullptr;
    Real* f_halo_prev  = nullptr;
    Real* f_halo_next  = nullptr;
    Real* f_local_host = nullptr;
    Real* f_prev_host  = nullptr;
    Real* f_next_host  = nullptr;

    memAllocArray<MemSpace::Device, Real>(&f_local     , NUM_GRIDS_I * NUM_GRIDS_J * NUM_GRIDS_K);
    memAllocArray<MemSpace::Device, Real>(&df_local    , NUM_GRIDS_I * NUM_GRIDS_J * NUM_GRIDS_K);
    memAllocArray<MemSpace::Device, Real>(&f_halo_prev , NUM_GRIDS_I * NUM_GRIDS_J *           3);
    memAllocArray<MemSpace::Device, Real>(&f_halo_next , NUM_GRIDS_I * NUM_GRIDS_J *           3);
    memAllocArray<MemSpace::Host  , Real>(&f_local_host, NUM_GRIDS_I * NUM_GRIDS_J * NUM_GRIDS_K);
    memAllocArray<MemSpace::Host  , Real>(&f_prev_host , NUM_GRIDS_I * NUM_GRIDS_J *           3);
    memAllocArray<MemSpace::Host  , Real>(&f_next_host , NUM_GRIDS_I * NUM_GRIDS_J *           3);

    Real* coef_a = nullptr;
    Real* coef_b = nullptr;
    Real* coef_c = nullptr;

    memAllocArray<MemSpace::Device, Real>(&coef_a, NUM_GRIDS_K);
    memAllocArray<MemSpace::Device, Real>(&coef_b, NUM_GRIDS_K);
    memAllocArray<MemSpace::Device, Real>(&coef_c, NUM_GRIDS_K);


    ////////////////////////////
    // CONFIGURE COEFFICIENTS //
    ////////////////////////////
    {
        Real* coef_a_host = nullptr;
        Real* coef_b_host = nullptr;
        Real* coef_c_host = nullptr;
        memAllocArray<MemSpace::Host, Real>(&coef_a_host, NUM_GRIDS_K);
        memAllocArray<MemSpace::Host, Real>(&coef_b_host, NUM_GRIDS_K);
        memAllocArray<MemSpace::Host, Real>(&coef_c_host, NUM_GRIDS_K);
        for (unsigned int k = 0; k < NUM_GRIDS_K; k++) {
            coef_a_host[k] = COEF_A;
            coef_b_host[k] = COEF_B;
            coef_c_host[k] = COEF_C;
        }
        deepCopy<MemSpace::Device, MemSpace::Host, Real>(coef_a, coef_a_host, NUM_GRIDS_K);
        deepCopy<MemSpace::Device, MemSpace::Host, Real>(coef_b, coef_b_host, NUM_GRIDS_K);
        deepCopy<MemSpace::Device, MemSpace::Host, Real>(coef_c, coef_c_host, NUM_GRIDS_K);
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
    for (unsigned int i = 0; i < NUM_GRIDS_I; i++) {
        const Real x = i * dx;
        for (unsigned int j = 0; j < NUM_GRIDS_J; j++) {
            const Real y = j * dy;
            for (unsigned int k = 0; k < NUM_GRIDS_K; k++) {
                const Real z = k * dz;
                f_local_host[i * NUM_GRIDS_J * NUM_GRIDS_K + j * NUM_GRIDS_K + k] = cos(K_WAVE * x) * cos(K_WAVE * y) * cos(K_WAVE * z);
            }
        }
    }

    for (unsigned int i = 0; i < NUM_GRIDS_I; i++) {
        const Real x = i * dx;
        for (unsigned int j = 0; j < NUM_GRIDS_J; j++) {
            const Real y = j * dy;
            for (int k = 0; k < 3; k++) {
                const Real z_prev = (k -           3) * dz;
                const Real z_next = (k + NUM_GRIDS_K) * dz;
                f_prev_host[i * NUM_GRIDS_J * 3 + j * 3 + k] = cos(K_WAVE * x) * cos(K_WAVE * y) * cos(K_WAVE * z_prev);
                f_next_host[i * NUM_GRIDS_J * 3 + j * 3 + k] = cos(K_WAVE * x) * cos(K_WAVE * y) * cos(K_WAVE * z_next);
            }
        }
    }

    deepCopy<MemSpace::Device, MemSpace::Host, Real>(f_local    , f_local_host, NUM_GRIDS_I * NUM_GRIDS_J * NUM_GRIDS_K);
    deepCopy<MemSpace::Device, MemSpace::Host, Real>(f_halo_prev, f_prev_host , NUM_GRIDS_I * NUM_GRIDS_J *           3);
    deepCopy<MemSpace::Device, MemSpace::Host, Real>(f_halo_next, f_next_host , NUM_GRIDS_I * NUM_GRIDS_J *           3);


    ///////////////////////
    // CONDUCT UNIT TEST //
    ///////////////////////
    centralDiffCollStencil7DimK<Real>(df_local, f_local, f_halo_prev, f_halo_next, coef_a, coef_b, coef_c,
            NUM_GRIDS_I, NUM_GRIDS_J, NUM_GRIDS_K, NUM_GRIDS_J * NUM_GRIDS_K, NUM_GRIDS_K);


    //////////////////
    // CHECK RESULT //
    //////////////////
    deepCopy<MemSpace::Host, MemSpace::Device, Real>(f_local_host, df_local, NUM_GRIDS_I * NUM_GRIDS_J * NUM_GRIDS_K);
    double err_rank = 0.0;
    for (unsigned int i = 0; i < NUM_GRIDS_I; i++) {
        const Real x = i * dx;
        for (unsigned int j = 0; j < NUM_GRIDS_J; j++) {
            const Real y = j * dy;
            for (unsigned int k = 0; k < NUM_GRIDS_K; k++) {
                const Real z = k * dz;
                const Real df_ref = -K_MOD * cos(K_WAVE * x) * cos(K_WAVE * y) * sin(K_WAVE * z);
                const Real df_num = f_local_host[i * NUM_GRIDS_J * NUM_GRIDS_K + j * NUM_GRIDS_K + k];
                const Real err_ijk = df_num - df_ref;
                err_rank += err_ijk * err_ijk;
                //if(j==0 && k == 0 && fabs(err_ijk) > 1e-13) printf("[%3u]\t ref = %12.5e\t num = %12.5e\t err = %12.5e\n", i, df_ref, df_num, err_ijk);
            }
        }
    }
    double err_rms = sqrt(err_rank / (NUM_GRIDS_I * NUM_GRIDS_J * NUM_GRIDS_K));
    printf("Stencil central-diff in dim_k RMS error: %12.5e\n.", err_rms);
    //double err_rms = 0.0;
    //MPI_Allreduce(&err_rank, &err_rms, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    //err_rms = sqrt(err_rms / (NUM_GRIDS_I * NUM_GRIDS_J * NUM_GRIDS_K));
    //if (rank == 0) printf("RMS error: %12.5e\n.", err_rms);

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
    //MPI_Finalize();
    return 0;
}
