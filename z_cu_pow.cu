#include <assert.h>

#include "z_cu_pow.cuh"

#define Z_CU_POW(suffix, type)                                                              \
__global__ void z_cu_pow_##suffix(z_cu_t const* base, type exp, z_cu_t* p, bool* synchro)   \
{                                                                                           \
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;                               \
                                                                                            \
    __shared__ bool* synchroTmp;                                                            \
    cudaMalloc(&synchroTmp, sizeof(bool));                                                  \
                                                                                            \
    z_cu_t* e;                                                                              \
    cudaMalloc(&e, sizeof(z_cu_t));                                                         \
    SYNCHRO((z_cu_from_##suffix<<<1, 1>>>(e, exp, synchroTmp)), synchroTmp, 1);             \
                                                                                            \
    SYNCHRO((z_cu_pow_z<<<1, 1>>>(base, e, p, synchroTmp)), synchroTmp, 1);                 \
                                                                                            \
    SYNCHRO((z_cu_free<<<1, 1>>>(e, synchroTmp)), synchroTmp, 1);                           \
    cudaFree(e);                                                                            \
                                                                                            \
    cudaFree(synchroTmp);                                                                   \
                                                                                            \
    synchro[idx] = true;                                                                             \
}

Z_CU_POW(c, char)
Z_CU_POW(i, int)
Z_CU_POW(l, long)
Z_CU_POW(ll, long long)
Z_CU_POW(s, short)
Z_CU_POW(uc, unsigned char)
Z_CU_POW(ui, unsigned int)
Z_CU_POW(ul, unsigned long)
Z_CU_POW(ull, unsigned long long)
Z_CU_POW(us, unsigned short)

__global__ void z_cu_pow_z(z_cu_t const* base, z_cu_t const* exp, z_cu_t* p, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ bool* synchroTmp;
    cudaMalloc(&synchroTmp, sizeof(bool));

    int* cmp1;
    cudaMalloc(&cmp1, sizeof(int));
    SYNCHRO((z_cu_cmp_c<<<1, 1>>>(exp, 0, cmp1, synchroTmp)), synchroTmp, 1);

    assert(*cmp1 >= 0);

    SYNCHRO((z_cu_from_c<<<1, 1>>>(p, 0, synchroTmp)), synchroTmp, 1);

    SYNCHRO((z_cu_cmp_c<<<1, 1>>>(base, 0, cmp1, synchroTmp)), synchroTmp, 1);

    int* cmp2;
    cudaMalloc(&cmp2, sizeof(int));
    SYNCHRO((z_cu_cmp_c<<<1, 1>>>(base, 2, cmp2, synchroTmp)), synchroTmp, 1);

    if (base->is_infinity || base->is_nan)
    {
        SYNCHRO((z_cu_free<<<1, 1>>>(p, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_copy<<<1, 1>>>(base, p, synchroTmp)), synchroTmp, 1);
    }
    else if (exp->is_nan || exp->is_infinity)
    {
        SYNCHRO((z_cu_free<<<1, 1>>>(p, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_copy<<<1, 1>>>(exp, p, synchroTmp)), synchroTmp, 1);
    }
    else if (*cmp1 < 0)
    {
        SYNCHRO((z_cu_free<<<1, 1>>>(p, synchroTmp)), synchroTmp, 1);

        z_cu_t* base_abs;
        cudaMalloc(&base_abs, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(base, base_abs, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_abs<<<1, 1>>>(base_abs, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_pow_z<<<1, 1>>>(base_abs, exp, p, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_free<<<1, 1>>>(base_abs, synchroTmp)), synchroTmp, 1);
        cudaFree(base_abs);

        z_cu_t* a;
        cudaMalloc(&a, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(exp, a, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_and_c<<<1, 1>>>(a, 1, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_cmp_c<<<1, 1>>>(a, 0, cmp1, synchroTmp)), synchroTmp, 1);

        if (*cmp1)
            p->is_positive = !p->is_positive;

        SYNCHRO((z_cu_free<<<1, 1>>>(a, synchroTmp)), synchroTmp, 1);
        cudaFree(a);
    }
    else if (*cmp2)
    {
        SYNCHRO((z_cu_free<<<1, 1>>>(p, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_from_c<<<1, 1>>>(p, 1, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_rshift_z<<<1, 1>>>(p, exp, synchroTmp)), synchroTmp, 1);
    }
    else
    {
        SYNCHRO((z_cu_free<<<1, 1>>>(p, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_from_c<<<1, 1>>>(p, 1, synchroTmp)), synchroTmp, 1);
        z_cu_t* e;
        cudaMalloc(&e, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(exp, e, synchroTmp)), synchroTmp, 1);
        z_cu_t* b;
        cudaMalloc(&b, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(base, b, synchroTmp)), synchroTmp, 1);

        for (;;)
        {
            z_cu_t* a;
            cudaMalloc(&a, sizeof(z_cu_t));
            SYNCHRO((z_cu_copy<<<1, 1>>>(e, a, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_and_c<<<1, 1>>>(a, 1, synchroTmp)), synchroTmp, 1);

            SYNCHRO((z_cu_cmp_c<<<1, 1>>>(a, 0, cmp1, synchroTmp)), synchroTmp, 1);

            if (*cmp1)
            {
                SYNCHRO((z_cu_mul_z<<<1, 1>>>(p, b, synchroTmp)), synchroTmp, 1);
            }

            SYNCHRO((z_cu_free<<<1, 1>>>(a, synchroTmp)), synchroTmp, 1);
            cudaFree(a);

            SYNCHRO((z_cu_rshift_c<<<1, 1>>>(e, 1, synchroTmp)), synchroTmp, 1);

            SYNCHRO((z_cu_cmp_c<<<1, 1>>>(e, 0, cmp1, synchroTmp)), synchroTmp, 1);

            if (!*cmp1)
                break;

            z_cu_t* bTmp;
            cudaMalloc(&bTmp, sizeof(z_cu_t));
            SYNCHRO((z_cu_copy<<<1, 1>>>(b, bTmp, synchroTmp)), synchroTmp, 1);

            SYNCHRO((z_cu_mul_z<<<1, 1>>>(b, bTmp, synchroTmp)), synchroTmp, 1);

            SYNCHRO((z_cu_free<<<1, 1>>>(bTmp, synchroTmp)), synchroTmp, 1);
            cudaFree(bTmp);
        }

        SYNCHRO((z_cu_free<<<1, 1>>>(e, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(b, synchroTmp)), synchroTmp, 1);
        cudaFree(e);
        cudaFree(b);
    }

    cudaFree(cmp1);
    cudaFree(cmp2);

    cudaFree(synchroTmp);

    synchro[idx] = true;
}
