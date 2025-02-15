#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "z_cu_add.cuh"

#define Z_CU_ADD(suffix, type)                                                      \
__global__ void z_cu_add_##suffix(z_cu_t* lhs, type rhs, bool* synchro)             \
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
    SYNCHRO((z_cu_add_z<<<1, 1>>>(lhs, other, synchroTmp)), synchroTmp, 1);         \
                                                                                    \
    SYNCHRO((z_cu_free<<<1, 1>>>(other, synchroTmp)), synchroTmp, 1);               \
                                                                                    \
    cudaFree(synchroTmp);                                                           \
    cudaFree(other);                                                                \
                                                                                    \
    synchro[idx] = true;                                                            \
}

Z_CU_ADD(c, char)
Z_CU_ADD(i, int)
Z_CU_ADD(l, long)
Z_CU_ADD(ll, long long)
Z_CU_ADD(s, short)
Z_CU_ADD(uc, unsigned char)
Z_CU_ADD(ui, unsigned int)
Z_CU_ADD(ul, unsigned long)
Z_CU_ADD(ull, unsigned long long)
Z_CU_ADD(us, unsigned short)

__global__ void z_cu_add_z(z_cu_t* lhs, z_cu_t const* rhs, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    assert(lhs);
    assert(rhs);

    __shared__ bool* synchroTmp;
    cudaMalloc(&synchroTmp, sizeof(bool));

    if (lhs->is_nan || rhs->is_nan)
    {
        SYNCHRO((z_cu_set_nan<<<1, 1>>>(lhs, synchroTmp)), synchroTmp, 1);
    }
    else if (lhs->is_infinity || rhs->is_infinity)
    {
        if ((lhs->is_positive && !rhs->is_positive)
            || (!lhs->is_positive && rhs->is_positive))
        {
            SYNCHRO((z_cu_set_nan<<<1, 1>>>(lhs, synchroTmp)), synchroTmp, 1);
        }
        else
        {
            if (rhs->is_infinity)
                lhs->is_positive = rhs->is_positive;

            SYNCHRO((z_cu_set_infinity<<<1, 1>>>(lhs, synchroTmp)), synchroTmp, 1);
        }
    }
    else
    {
        if ((lhs->is_positive && rhs->is_positive)
            || (!lhs->is_positive && !rhs->is_positive))
        {
            z_cu_type carry = 0;

            z_cu_type const* a = lhs->bits;
            z_cu_type const* b = rhs->bits;

            assert(a && b);

            size_t n = MAX(lhs->size, rhs->size);

            z_cu_type* result = 0;
            cudaMalloc(&result, n * sizeof(z_cu_type));

            assert(result);

            for (size_t i = 0; i < n; ++i)
            {
                z_cu_type const bit_a = (i < lhs->size) ? a[lhs->size - 1 - i] : 0;
                z_cu_type const bit_b = (i < rhs->size) ? b[rhs->size - 1 - i] : 0;
                z_cu_type const sum = bit_a + bit_b + carry;

                carry = (sum < bit_a || sum < bit_b);

                result[i] = sum;
            }

            if (carry)
            {
                z_cu_type* r = 0;
                cudaMalloc(&r, (n + 1) * sizeof(z_cu_type));
                assert(r);
                memcpy(r, result, n * sizeof(z_cu_type));
                r[n] = 1;
                cudaFree(result);
                result = r;
                ++n;
            }

            for (size_t i = 0; i < n / 2; ++i)
            {
                z_cu_type tmp = result[i];
                result[i] = result[n - 1 - i];
                result[n - 1 - i] = tmp;
            }

            cudaFree(lhs->bits);
            lhs->bits = result;
            lhs->size = n;
        }
        else
        {
            z_cu_type* other_bits = 0;
            cudaMalloc(&other_bits, rhs->size * sizeof(z_cu_type));
            assert(other_bits);
            memcpy(other_bits, rhs->bits, rhs->size * sizeof(z_cu_type));
            z_cu_type other_size = rhs->size;

            if (lhs->is_positive)
            {
                z_cu_t* neg = 0;
                cudaMalloc(&neg, sizeof(z_cu_t));
                SYNCHRO((z_cu_copy<<<1, 1>>>(rhs, neg, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_neg<<<1, 1>>>(neg, synchroTmp)), synchroTmp, 1);
                int* cmp;
                cudaMalloc(&cmp, sizeof(int));
                SYNCHRO((z_cu_cmp_z<<<1, 1>>>(lhs, neg, cmp, synchroTmp)), synchroTmp, 1);

                if (*cmp < 0)
                {
                    lhs->is_positive = false;
                    cudaFree(other_bits);
                    other_bits = 0;
                    cudaMalloc(&other_bits, lhs->size * sizeof(z_cu_type));
                    assert(other_bits);
                    memcpy(other_bits, lhs->bits, lhs->size * sizeof(z_cu_type));
                    other_size = lhs->size;
                    cudaFree(lhs->bits);
                    lhs->bits = 0;
                    cudaMalloc(&lhs->bits, rhs->size * sizeof(z_cu_type));
                    assert(lhs->bits);
                    memcpy(lhs->bits, rhs->bits, rhs->size * sizeof(z_cu_type));
                    lhs->size = rhs->size;
                }

                SYNCHRO((z_cu_free<<<1, 1>>>(neg, synchroTmp)), synchroTmp, 1);
                cudaFree(neg);
                cudaFree(cmp);
            }
            else
            {
                z_cu_t* neg = 0;
                cudaMalloc(&neg, sizeof(z_cu_t));
                SYNCHRO((z_cu_copy<<<1, 1>>>(rhs, neg, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_neg<<<1, 1>>>(neg, synchroTmp)), synchroTmp, 1);
                int* cmp;
                cudaMalloc(&cmp, sizeof(int));
                SYNCHRO((z_cu_cmp_z<<<1, 1>>>(lhs, neg, cmp, synchroTmp)), synchroTmp, 1);

                if (*cmp > 0)
                {
                    lhs->is_positive = true;
                    cudaFree(other_bits);
                    other_bits = 0;
                    cudaMalloc(&other_bits, lhs->size * sizeof(z_cu_type));
                    assert(other_bits);
                    memcpy(other_bits, lhs->bits, lhs->size * sizeof(z_cu_type));
                    other_size = lhs->size;
                    cudaFree(lhs->bits);
                    lhs->bits = 0;
                    cudaMalloc(&lhs->bits, rhs->size * sizeof(z_cu_type));
                    assert(lhs->bits);
                    memcpy(lhs->bits, rhs->bits, rhs->size * sizeof(z_cu_type));
                    lhs->size = rhs->size;
                }

                SYNCHRO((z_cu_free<<<1, 1>>>(neg, synchroTmp)), synchroTmp, 1);
                cudaFree(neg);
                cudaFree(cmp);
            }

            z_cu_type const* a = lhs->bits;
            z_cu_type const* b = other_bits;

            assert(a && b);

            size_t n = MAX(lhs->size, other_size);

            z_cu_type* result = 0;
            cudaMalloc(&result, n * sizeof(z_cu_type));

            assert(result);

            for (size_t i = n - 1; i <= n - 1; --i)
            {
                size_t const ia = lhs->size - 1 - i;
                size_t const ib = rhs->size - 1 - i;

                z_cu_type const bit_a = ia < lhs->size ? a[ia] : 0;
                z_cu_type const bit_b = ib < rhs->size ? b[ib] : 0;

                z_cu_type bit_result = bit_a - bit_b;

                if (bit_a < bit_b)
                {
                    for (size_t j = i - 1; j <= i - 1; --j)
                    {
                        bool const stop = (result[j] > 0);

                        result[j] -= 1;

                        if (stop)
                            break;
                    }

                    bit_result = -1;
                    bit_result -= bit_b;
                    bit_result += bit_a + 1;
                }

                result[i] = bit_result;
            }

            cudaFree(lhs->bits);
            lhs->bits = result;
            lhs->size = n;
            cudaFree(other_bits);
        }
    }

    if (lhs->is_auto_adjust)
    {
        SYNCHRO((z_cu_adjust<<<1, 1>>>(lhs, synchroTmp)), synchroTmp, 1);
    }

    cudaFree(synchroTmp);

    synchro[idx] = true;
}
