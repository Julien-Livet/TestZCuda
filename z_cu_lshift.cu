#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "z_cu_lshift.cuh"

#define Z_CU_LSHIFT(suffix, type)                                                   \
__global__ void z_cu_lshift_##suffix(z_cu_t* lhs, type rhs, bool* synchro)          \
{                                                                                   \
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;                       \
                                                                                    \
    assert(lhs);                                                                    \
                                                                                    \
    z_cu_t* other = 0;                                                              \
    cudaMalloc(&other, sizeof(z_cu_t));                                             \
                                                                                    \
    __shared__ bool* synchroTmp;                                                    \
    cudaMalloc(&synchroTmp, sizeof(bool));                                          \
                                                                                    \
    SYNCHRO((z_cu_from_##suffix<<<1, 1>>>(other, rhs, synchroTmp)), synchroTmp, 1); \
                                                                                    \
    SYNCHRO((z_cu_lshift_z<<<1, 1>>>(lhs, other, synchroTmp)), synchroTmp, 1);      \
                                                                                    \
    SYNCHRO((z_cu_free<<<1, 1>>>(other, synchroTmp)), synchroTmp, 1);               \
                                                                                    \
    cudaFree(synchroTmp);                                                           \
    cudaFree(other);                                                                \
                                                                                    \
    synchro[idx] = true;                                                            \
}

Z_CU_LSHIFT(c, char)
Z_CU_LSHIFT(i, int)
Z_CU_LSHIFT(l, long)
Z_CU_LSHIFT(ll, long long)
Z_CU_LSHIFT(s, short)
Z_CU_LSHIFT(uc, unsigned char)
Z_CU_LSHIFT(ui, unsigned int)
Z_CU_LSHIFT(ul, unsigned long)
Z_CU_LSHIFT(ull, unsigned long long)
Z_CU_LSHIFT(us, unsigned short)

__global__ void z_cu_lshift_z(z_cu_t* lhs, z_cu_t const* rhs, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    assert(lhs);
    assert(rhs);

    __shared__ bool* synchroTmp;
    cudaMalloc(&synchroTmp, sizeof(bool));

    int* cmp1;
    cudaMalloc(&cmp1, sizeof(int));

    SYNCHRO((z_cu_cmp_c<<<1, 1>>>(rhs, 0, cmp1, synchroTmp)), synchroTmp, 1);

    assert(*cmp1 >= 0);

    int* cmp2;
    cudaMalloc(&cmp2, sizeof(int));

    SYNCHRO((z_cu_cmp_c<<<1, 1>>>(lhs, 0, cmp1, synchroTmp)), synchroTmp, 1);
    SYNCHRO((z_cu_cmp_c<<<1, 1>>>(rhs, 0, cmp2, synchroTmp)), synchroTmp, 1);

    if (!*cmp1 || !*cmp2)
    {
        cudaFree(cmp1);
        cudaFree(cmp2);
    }
    else if (lhs->is_nan || rhs->is_nan || lhs->is_infinity || rhs->is_infinity)
    {
        cudaFree(cmp1);
        cudaFree(cmp2);

        SYNCHRO((z_cu_set_nan<<<1, 1>>>(lhs, synchroTmp)), synchroTmp, 1);
    }
    else
    {
        cudaFree(cmp1);
        cudaFree(cmp2);

        unsigned short const us = sizeof(z_cu_type) * 8;
        z_cu_t* n;
        cudaMalloc(&n, sizeof(z_cu_t));
        SYNCHRO((z_cu_div_q_us<<<1, 1>>>(rhs, us, n, synchroTmp)), synchroTmp, 1);

        {
            unsigned long long* ull;
            cudaMalloc(&ull, sizeof(unsigned long long));
            SYNCHRO((z_cu_to_ull<<<1, 1>>>(n, ull, synchroTmp)), synchroTmp, 1);

            z_cu_type* bits = 0;
            cudaMalloc(&bits, (*ull + lhs->size) * sizeof(z_cu_type));
            assert(bits);
            memset((char*)(bits) + lhs->size * sizeof(z_cu_type), 0, *ull * sizeof(z_cu_type));
            memcpy(bits, lhs->bits, lhs->size * sizeof(z_cu_type));
            cudaFree(lhs->bits);
            lhs->bits = bits;
            lhs->size += *ull;

            cudaFree(ull);
        }

        z_cu_t* other;
        cudaMalloc(&other, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(rhs, other, synchroTmp)), synchroTmp, 1);
        z_cu_t* nTmp;
        cudaMalloc(&nTmp, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(n, nTmp, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_mul_us<<<1, 1>>>(nTmp, us, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_sub_z<<<1, 1>>>(other, nTmp, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(nTmp, synchroTmp)), synchroTmp, 1);
        cudaFree(nTmp);

        z_cu_type* bits = 0;
        cudaMalloc(&bits, (lhs->size + 1) * sizeof(z_cu_type));
        assert(bits);
        memset(bits, 0, sizeof(z_cu_type));
        memcpy((char*)(bits) + sizeof(z_cu_type), lhs->bits, lhs->size * sizeof(z_cu_type));
        cudaFree(lhs->bits);
        lhs->bits = bits;
        ++lhs->size;

        unsigned long long* shift;
        cudaMalloc(&shift, sizeof(unsigned long long));
        SYNCHRO((z_cu_to_ull<<<1, 1>>>(other, shift, synchroTmp)), synchroTmp, 1);

        if (*shift)
        {
            for (size_t i = 1; i < lhs->size; ++i)
            {
                longest_type const s = sizeof(z_cu_type) * 8;

                if ((lhs->bits[i] >> (s - *shift)))
                    lhs->bits[i - 1] |= (lhs->bits[i] >> (s - *shift));

                lhs->bits[i] <<= *shift;
            }
        }

        cudaFree(shift);

        SYNCHRO((z_cu_free<<<1, 1>>>(n, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(other, synchroTmp)), synchroTmp, 1);
        cudaFree(n);
        cudaFree(other);

        if (lhs->is_auto_adjust)
        {
            SYNCHRO((z_cu_adjust<<<1, 1>>>(lhs, synchroTmp)), synchroTmp, 1);
        }
    }

    cudaFree(synchroTmp);

    synchro[idx] = true;
}
