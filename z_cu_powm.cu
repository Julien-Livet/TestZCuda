#include <assert.h>

#include "z_cu_powm.cuh"

#define Z_CU_POWM(suffix, type)                                                                         \
__global__ void z_cu_powm_##suffix(z_cu_t const* base, type exp, type mod, z_cu_t* p, bool* synchro)    \
{                                                                                                       \
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;                                           \
                                                                                                        \
    __shared__ bool* synchroTmp;                                                                        \
    cudaMalloc(&synchroTmp, sizeof(bool));                                                              \
                                                                                                        \
    z_cu_t* e;                                                                                          \
    cudaMalloc(&e, sizeof(z_cu_t));                                                                     \
    SYNCHRO((z_cu_from_##suffix<<<1, 1>>>(e, exp, synchroTmp)), synchroTmp, 1);                         \
    z_cu_t* m;                                                                                          \
    cudaMalloc(&m, sizeof(z_cu_t));                                                                     \
    SYNCHRO((z_cu_from_##suffix<<<1, 1>>>(m, mod, synchroTmp)), synchroTmp, 1);                         \
                                                                                                        \
    SYNCHRO((z_cu_powm_z<<<1, 1>>>(base, e, m, p, synchroTmp)), synchroTmp, 1);                         \
                                                                                                        \
    SYNCHRO((z_cu_free<<<1, 1>>>(e, synchroTmp)), synchroTmp, 1);                                       \
    SYNCHRO((z_cu_free<<<1, 1>>>(m, synchroTmp)), synchroTmp, 1);                                       \
    cudaFree(e);                                                                                        \
    cudaFree(m);                                                                                        \
                                                                                                        \
    cudaFree(synchroTmp);                                                                               \
                                                                                                        \
    synchro[idx] = true;                                                                                \
}

Z_CU_POWM(c, char)
Z_CU_POWM(i, int)
Z_CU_POWM(l, long)
Z_CU_POWM(ll, long long)
Z_CU_POWM(s, short)
Z_CU_POWM(uc, unsigned char)
Z_CU_POWM(ui, unsigned int)
Z_CU_POWM(ul, unsigned long)
Z_CU_POWM(ull, unsigned long long)
Z_CU_POWM(us, unsigned short)

__global__ void z_cu_powm_z(z_cu_t const* base, z_cu_t const* exp, z_cu_t const* mod, z_cu_t* p, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ bool* synchroTmp;
    cudaMalloc(&synchroTmp, sizeof(bool));

    int* cmp;
    cudaMalloc(&cmp, sizeof(int));
    SYNCHRO((z_cu_cmp_c<<<1, 1>>>(exp, 0, cmp, synchroTmp)), synchroTmp, 1);

    assert(*cmp >= 0);

    SYNCHRO((z_cu_from_c<<<1, 1>>>(p, 1, synchroTmp)), synchroTmp, 1);
    z_cu_t* base_mod;
    cudaMalloc(&base_mod, sizeof(z_cu_t));
    SYNCHRO((z_cu_div_r_z<<<1, 1>>>(base, mod, base_mod, synchroTmp)), synchroTmp, 1);
    z_cu_t* e;
    cudaMalloc(&e, sizeof(z_cu_t));
    SYNCHRO((z_cu_copy<<<1, 1>>>(exp, e, synchroTmp)), synchroTmp, 1);

    SYNCHRO((z_cu_cmp_c<<<1, 1>>>(e, 0, cmp, synchroTmp)), synchroTmp, 1);

    while (*cmp > 0)
    {
        z_cu_t* a;
        cudaMalloc(&a, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(e, a, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_and_c<<<1, 1>>>(a, 1, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_cmp_c<<<1, 1>>>(a, 1, cmp, synchroTmp)), synchroTmp, 1);

        if (!*cmp)
        {
            SYNCHRO((z_cu_mul_z<<<1, 1>>>(p, base_mod, synchroTmp)), synchroTmp, 1);

            z_cu_t* r;
            cudaMalloc(&r, sizeof(z_cu_t));
            SYNCHRO((z_cu_div_r_z<<<1, 1>>>(p, mod, r, synchroTmp)), synchroTmp, 1);

            SYNCHRO((z_cu_free<<<1, 1>>>(p, synchroTmp)), synchroTmp, 1);
            p = r;
        }

        SYNCHRO((z_cu_free<<<1, 1>>>(a, synchroTmp)), synchroTmp, 1);
        cudaFree(a);

        z_cu_t* base_modTmp;
        cudaMalloc(&base_modTmp, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(base_mod, base_modTmp, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_mul_z<<<1, 1>>>(base_mod, base_modTmp, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_free<<<1, 1>>>(base_modTmp, synchroTmp)), synchroTmp, 1);
        cudaFree(base_modTmp);

        z_cu_t* r;
        cudaMalloc(&r, sizeof(z_cu_t));
        SYNCHRO((z_cu_div_r_z<<<1, 1>>>(base_mod, mod, r, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_free<<<1, 1>>>(base_mod, synchroTmp)), synchroTmp, 1);
        *base_mod = *r;

        SYNCHRO((z_cu_rshift_c<<<1, 1>>>(e, 1, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_cmp_c<<<1, 1>>>(e, 0, cmp, synchroTmp)), synchroTmp, 1);
    }

    SYNCHRO((z_cu_free<<<1, 1>>>(base_mod, synchroTmp)), synchroTmp, 1);
    SYNCHRO((z_cu_free<<<1, 1>>>(e, synchroTmp)), synchroTmp, 1);
    cudaFree(base_mod);
    cudaFree(e);

    cudaFree(synchroTmp);

    synchro[idx] = true;
}
