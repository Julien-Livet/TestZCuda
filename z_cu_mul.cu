#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "z_cu_mul.cuh"

__global__ void number(longest_type n, size_t* number, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    *number = 0;

    while (n)
    {
        ++*number;
        n >>= 1;
    }

    synchro[idx] = true;
}

#define Z_CU_MUL(suffix, type)                                                      \
__global__ void z_cu_mul_##suffix(z_cu_t* lhs, type rhs, bool* synchro)             \
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
    SYNCHRO((z_cu_mul_z<<<1, 1>>>(lhs, other, synchroTmp)), synchroTmp, 1);         \
                                                                                    \
    SYNCHRO((z_cu_free<<<1, 1>>>(other, synchroTmp)), synchroTmp, 1);               \
                                                                                    \
    cudaFree(synchroTmp);                                                           \
    cudaFree(other);                                                                \
                                                                                    \
    synchro[idx] = true;                                                            \
}

Z_CU_MUL(c, char)
Z_CU_MUL(i, int)
Z_CU_MUL(l, long)
Z_CU_MUL(ll, long long)
Z_CU_MUL(s, short)
Z_CU_MUL(uc, unsigned char)
Z_CU_MUL(ui, unsigned int)
Z_CU_MUL(ul, unsigned long)
Z_CU_MUL(ull, unsigned long long)
Z_CU_MUL(us, unsigned short)

__global__ void z_cu_mul_z(z_cu_t* lhs, z_cu_t const* rhs, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    assert(lhs);
    assert(rhs);

    __shared__ bool* synchroTmp;
    cudaMalloc(&synchroTmp, sizeof(bool));

    if (!rhs->is_positive)
    {
        SYNCHRO((z_cu_neg<<<1, 1>>>(lhs, synchroTmp)), synchroTmp, 1);

        z_cu_t* other;
        cudaMalloc(&other, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(rhs, other, synchroTmp)), synchroTmp, 1);

        other->is_positive = !other->is_positive;

        SYNCHRO((z_cu_mul_z<<<1, 1>>>(lhs, rhs, synchroTmp)), synchroTmp, 1);
        
        SYNCHRO((z_cu_free<<<1, 1>>>(other, synchroTmp)), synchroTmp, 1);
        cudaFree(other);
    }
    else if (lhs->is_nan || rhs->is_nan)
    {
        SYNCHRO((z_cu_set_nan<<<1, 1>>>(lhs, synchroTmp)), synchroTmp, 1);
    }
    else
    {
        int* cmp1;
        cudaMalloc(&cmp1, sizeof(int));
        int* cmp2;
        cudaMalloc(&cmp2, sizeof(int));
        
        SYNCHRO((z_cu_cmp_c<<<1, 1>>>(lhs, 0, cmp1, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_cmp_c<<<1, 1>>>(rhs, 0, cmp2, synchroTmp)), synchroTmp, 1);

        if (!*cmp1 || !*cmp2)
        {
            cudaFree(cmp1);
            cudaFree(cmp2);

            SYNCHRO((z_cu_set_from_c<<<1, 1>>>(lhs, 0, synchroTmp)), synchroTmp, 1);
        }
        else if (lhs->is_infinity || rhs->is_infinity)
        {
            cudaFree(cmp1);
            cudaFree(cmp2);

            SYNCHRO((z_cu_set_infinity<<<1, 1>>>(lhs, synchroTmp)), synchroTmp, 1);

            if (!rhs->is_positive)
                lhs->is_positive = !lhs->is_positive;
        }
        else
        {
            cudaFree(cmp1);
            cudaFree(cmp2);

            if (lhs->is_positive && rhs->is_positive)
            {
                z_cu_t* rhsTmp;
                cudaMalloc(&rhsTmp, sizeof(z_cu_t));
                SYNCHRO((z_cu_copy<<<1, 1>>>(rhs, rhsTmp, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_and_c<<<1, 1>>>(rhsTmp, 1, synchroTmp)), synchroTmp, 1);

                bool* fits1;
                cudaMalloc(&fits1, sizeof(bool));
                bool* fits2;
                cudaMalloc(&fits2, sizeof(bool));

                SYNCHRO((z_cu_fits_ull<<<1, 1>>>(lhs, fits1, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_fits_ull<<<1, 1>>>(rhs, fits2, synchroTmp)), synchroTmp, 1);

                if (*fits1 && *fits2)
                {
                    unsigned long long* a;
                    cudaMalloc(&a, sizeof(unsigned long long));
                    SYNCHRO((z_cu_to_ull<<<1, 1>>>(lhs, a, synchroTmp)), synchroTmp, 1);
                    unsigned long long* b;
                    cudaMalloc(&b, sizeof(unsigned long long));
                    SYNCHRO((z_cu_to_ull<<<1, 1>>>(rhs, b, synchroTmp)), synchroTmp, 1);
                    unsigned long long const ab = *a * *b;

                    if (ab / *b == *a)
                    {
                        SYNCHRO((z_cu_set_from_ull<<<1, 1>>>(lhs, ab, synchroTmp)), synchroTmp, 1);
                    }
                    else
                    {
                        //Karatsuba algorithm
                        size_t* na;
                        cudaMalloc(&na, sizeof(size_t));
                        size_t* nb;
                        cudaMalloc(&nb, sizeof(size_t));

                        SYNCHRO((number<<<1, 1>>>(*a, na, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((number<<<1, 1>>>(*b, nb, synchroTmp)), synchroTmp, 1);

                        size_t n = MAX(*na, *nb);
                        cudaFree(na);
                        cudaFree(nb);
                        if (n % 2)
                            ++n;
                        size_t const m = n / 2;
                        longest_type zero = 0;
                        z_cu_t* x0;
                        cudaMalloc(&x0, sizeof(z_cu_t));
                        SYNCHRO((z_cu_from_ull<<<1, 1>>>(x0, (~zero >> (sizeof(longest_type) * 8 - m)) & *a, synchroTmp)), synchroTmp, 1);
                        z_cu_t* x1;
                        cudaMalloc(&x1, sizeof(z_cu_t));
                        SYNCHRO((z_cu_from_ull<<<1, 1>>>(x1, (((~zero >> (sizeof(longest_type) * 8 - m)) << m) & *a) >> m, synchroTmp)), synchroTmp, 1);
                        z_cu_t* y0;
                        cudaMalloc(&y0, sizeof(z_cu_t));
                        SYNCHRO((z_cu_from_ull<<<1, 1>>>(y0, (~zero >> (sizeof(longest_type) * 8 - m)) & *b, synchroTmp)), synchroTmp, 1);
                        z_cu_t* y1;
                        cudaMalloc(&y1, sizeof(z_cu_t));
                        SYNCHRO((z_cu_from_ull<<<1, 1>>>(y1, (((~zero >> (sizeof(longest_type) * 8 - m)) << m) & *b) >> m, synchroTmp)), synchroTmp, 1);

                        z_cu_t* z0;
                        cudaMalloc(&z0, sizeof(z_cu_t));
                        SYNCHRO((z_cu_copy<<<1, 1>>>(x0, z0, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_mul_z<<<1, 1>>>(z0, y0, synchroTmp)), synchroTmp, 1);
                        z_cu_t* z1a;
                        cudaMalloc(&z1a, sizeof(z_cu_t));
                        SYNCHRO((z_cu_copy<<<1, 1>>>(x1, z1a, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_mul_z<<<1, 1>>>(z1a, y0, synchroTmp)), synchroTmp, 1);
                        z_cu_t* z1b;
                        cudaMalloc(&z1b, sizeof(z_cu_t));
                        SYNCHRO((z_cu_copy<<<1, 1>>>(x0, z1b, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_mul_z<<<1, 1>>>(z1b, y1, synchroTmp)), synchroTmp, 1);
                        z_cu_t* z1;
                        cudaMalloc(&z1, sizeof(z_cu_t));
                        SYNCHRO((z_cu_copy<<<1, 1>>>(z1a, z1, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_add_z<<<1, 1>>>(z1, z1b, synchroTmp)), synchroTmp, 1);
                        z_cu_t* z2;
                        cudaMalloc(&z2, sizeof(z_cu_t));
                        SYNCHRO((z_cu_copy<<<1, 1>>>(x1, z2, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_mul_z<<<1, 1>>>(z2, y1, synchroTmp)), synchroTmp, 1);

                        //xy = z2 * 2^(2 * m) + z1 * 2^m + z0
                        SYNCHRO((z_cu_lshift_ull<<<1, 1>>>(z1, m, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_lshift_ul<<<1, 1>>>(z2, 2 * m, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_set_from_z<<<1, 1>>>(lhs, z0, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_add_z<<<1, 1>>>(lhs, z1, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_add_z<<<1, 1>>>(lhs, z2, synchroTmp)), synchroTmp, 1);

                        SYNCHRO((z_cu_free<<<1, 1>>>(x0, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_free<<<1, 1>>>(x1, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_free<<<1, 1>>>(y0, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_free<<<1, 1>>>(y1, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_free<<<1, 1>>>(z0, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_free<<<1, 1>>>(z1a, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_free<<<1, 1>>>(z1b, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_free<<<1, 1>>>(z1, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_free<<<1, 1>>>(z2, synchroTmp)), synchroTmp, 1);
                        cudaFree(x0);
                        cudaFree(x1);
                        cudaFree(y0);
                        cudaFree(y1);
                        cudaFree(z0);
                        cudaFree(z1a);
                        cudaFree(z1b);
                        cudaFree(z1);
                        cudaFree(z2);
                    }
                }
                else
                {
                    int* cmp;
                    cudaMalloc(&cmp, sizeof(int));

                    SYNCHRO((z_cu_cmp_c<<<1, 1>>>(rhsTmp, 0, cmp, synchroTmp)), synchroTmp, 1);

                    if (!*cmp)
                    {
                        z_cu_t* r;
                        cudaMalloc(&r, sizeof(z_cu_t));
                        SYNCHRO((z_cu_copy<<<1, 1>>>(rhs, r, synchroTmp)), synchroTmp, 1);
                        z_cu_t* shift;
                        cudaMalloc(&shift, sizeof(z_cu_t));
                        SYNCHRO((z_cu_from_c<<<1, 1>>>(shift, 0, synchroTmp)), synchroTmp, 1);
                        z_cu_t* rTmp;
                        cudaMalloc(&rTmp, sizeof(z_cu_t));
                        SYNCHRO((z_cu_copy<<<1, 1>>>(r, rTmp, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_and_c<<<1, 1>>>(rTmp, 1, synchroTmp)), synchroTmp, 1);

                        SYNCHRO((z_cu_cmp_c<<<1, 1>>>(rTmp, 0, cmp, synchroTmp)), synchroTmp, 1);

                        while (!*cmp)
                        {
                            SYNCHRO((z_cu_rshift_c<<<1, 1>>>(r, 1, synchroTmp)), synchroTmp, 1);
                            SYNCHRO((z_cu_add_c<<<1, 1>>>(shift, 1, synchroTmp)), synchroTmp, 1);
                            SYNCHRO((z_cu_free<<<1, 1>>>(rTmp, synchroTmp)), synchroTmp, 1);
                            SYNCHRO((z_cu_copy<<<1, 1>>>(r, rTmp, synchroTmp)), synchroTmp, 1);
                            SYNCHRO((z_cu_and_c<<<1, 1>>>(rTmp, 1, synchroTmp)), synchroTmp, 1);

                            SYNCHRO((z_cu_cmp_c<<<1, 1>>>(rTmp, 0, cmp, synchroTmp)), synchroTmp, 1);
                        }

                        SYNCHRO((z_cu_lshift_z<<<1, 1>>>(lhs, shift, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_mul_z<<<1, 1>>>(lhs, r, synchroTmp)), synchroTmp, 1);

                        SYNCHRO((z_cu_free<<<1, 1>>>(r, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_free<<<1, 1>>>(rTmp, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_free<<<1, 1>>>(shift, synchroTmp)), synchroTmp, 1);
                        cudaFree(r);
                        cudaFree(rTmp);
                        cudaFree(shift);
                    }
                    else
                    {
                        //Karatsuba algorithm
                        //x = x1 * 2^m + x0
                        //y = y1 * 2^m + y0

                        z_cu_t* tmp;
                        cudaMalloc(&tmp, sizeof(z_cu_t));
                        SYNCHRO((z_cu_number<<<1, 1>>>(lhs, tmp, synchroTmp)), synchroTmp, 1);
                        z_cu_t* n1;
                        cudaMalloc(&n1, sizeof(z_cu_t));
                        SYNCHRO((z_cu_from_c<<<1, 1>>>(n1, 0, synchroTmp)), synchroTmp, 1);
                        z_cu_t* r;
                        cudaMalloc(&r, sizeof(z_cu_t));
                        SYNCHRO((z_cu_from_c<<<1, 1>>>(r, 0, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_div_qr_ull<<<1, 1>>>(tmp, sizeof(z_cu_type) * 8, n1, r, synchroTmp)), synchroTmp, 1);

                        SYNCHRO((z_cu_cmp_c<<<1, 1>>>(r, 0, cmp, synchroTmp)), synchroTmp, 1);

                        if (*cmp)
                        {
                            SYNCHRO((z_cu_add_c<<<1, 1>>>(n1, 1, synchroTmp)), synchroTmp, 1);
                        }

                        SYNCHRO((z_cu_free<<<1, 1>>>(tmp, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_free<<<1, 1>>>(r, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_div_r_c<<<1, 1>>>(n1, 2, r, synchroTmp)), synchroTmp, 1);

                        SYNCHRO((z_cu_cmp_c<<<1, 1>>>(r, 0, cmp, synchroTmp)), synchroTmp, 1);

                        if (*cmp)
                        {
                            SYNCHRO((z_cu_add_c<<<1, 1>>>(n1, 1, synchroTmp)), synchroTmp, 1);
                        }

                        SYNCHRO((z_cu_free<<<1, 1>>>(r, synchroTmp)), synchroTmp, 1);

                        SYNCHRO((z_cu_number<<<1, 1>>>(rhs, tmp, synchroTmp)), synchroTmp, 1);
                        z_cu_t* n2;
                        cudaMalloc(&n2, sizeof(z_cu_t));
                        SYNCHRO((z_cu_from_c<<<1, 1>>>(n2, 0, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_from_c<<<1, 1>>>(r, 0, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_div_qr_ull<<<1, 1>>>(tmp, sizeof(z_cu_type) * 8, n2, r, synchroTmp)), synchroTmp, 1);
                        
                        SYNCHRO((z_cu_cmp_c<<<1, 1>>>(r, 0, cmp, synchroTmp)), synchroTmp, 1);

                        if (*cmp)
                        {
                            SYNCHRO((z_cu_add_c<<<1, 1>>>(n2, 1, synchroTmp)), synchroTmp, 1);
                        }

                        SYNCHRO((z_cu_free<<<1, 1>>>(tmp, synchroTmp)), synchroTmp, 1);
                        cudaFree(tmp);
                        SYNCHRO((z_cu_free<<<1, 1>>>(r, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_div_r_c<<<1, 1>>>(n2, 2, r, synchroTmp)), synchroTmp, 1);
                        
                        SYNCHRO((z_cu_cmp_c<<<1, 1>>>(r, 0, cmp, synchroTmp)), synchroTmp, 1);
                        
                        if (*cmp)
                        {
                            SYNCHRO((z_cu_add_c<<<1, 1>>>(n2, 1, synchroTmp)), synchroTmp, 1);
                        }

                        SYNCHRO((z_cu_free<<<1, 1>>>(r, synchroTmp)), synchroTmp, 1);
                        cudaFree(r);
                        cudaFree(cmp);

                        unsigned long long* n1a;
                        cudaMalloc(&n1a, sizeof(unsigned long long));
                        unsigned long long* n2a;
                        cudaMalloc(&n2a, sizeof(unsigned long long));
                        
                        SYNCHRO((z_cu_to_ull<<<1, 1>>>(n1, n1a, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_to_ull<<<1, 1>>>(n2, n2a, synchroTmp)), synchroTmp, 1);

                        size_t const n = MAX(*n1a, *n2a);
                        size_t const m = n / 2;

                        cudaFree(n1a);
                        cudaFree(n2a);

                        z_cu_t* x0;
                        cudaMalloc(&x0, sizeof(z_cu_t));
                        SYNCHRO((z_cu_from_c<<<1, 1>>>(x0, 0, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_set_precision<<<1, 1>>>(x0, m, synchroTmp)), synchroTmp, 1);
                        memcpy((char*)(x0->bits) + (x0->size - MIN(lhs->size, m)) * sizeof(z_cu_type),
                               (char*)(lhs->bits) + (lhs->size - MIN(lhs->size, m)) * sizeof(z_cu_type),
                               MIN(lhs->size, m) * sizeof(z_cu_type));

                        z_cu_t* x1;
                        cudaMalloc(&x1, sizeof(z_cu_t));
                        SYNCHRO((z_cu_from_c<<<1, 1>>>(x1, 0, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_set_precision<<<1, 1>>>(x1, m, synchroTmp)), synchroTmp, 1);
                        if (MIN(lhs->size, 2 * m) >= m)
                            memcpy((char*)(x1->bits) + (x1->size - (MIN(lhs->size, 2 * m) - m)) * sizeof(z_cu_type),
                                   (char*)(lhs->bits) + (lhs->size - MIN(lhs->size, 2 * m)) * sizeof(z_cu_type),
                                   (MIN(lhs->size, 2 * m) - m) * sizeof(z_cu_type));

                        z_cu_t* y0;
                        cudaMalloc(&y0, sizeof(z_cu_t));
                        SYNCHRO((z_cu_from_c<<<1, 1>>>(y0, 0, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_set_precision<<<1, 1>>>(y0, m, synchroTmp)), synchroTmp, 1);
                        memcpy((char*)(y0->bits) + (y0->size - MIN(rhs->size, m)) * sizeof(z_cu_type),
                               (char*)(rhs->bits) + (rhs->size - MIN(rhs->size, m)) * sizeof(z_cu_type),
                               MIN(rhs->size, m) * sizeof(z_cu_type));

                        z_cu_t* y1;
                        cudaMalloc(&y1, sizeof(z_cu_t));
                        SYNCHRO((z_cu_from_c<<<1, 1>>>(y1, 0, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_set_precision<<<1, 1>>>(y1, m, synchroTmp)), synchroTmp, 1);
                        if (MIN(rhs->size, 2 * m) >= m)
                            memcpy((char*)(y1->bits) + (y1->size - (MIN(rhs->size, 2 * m) - m)) * sizeof(z_cu_type),
                                   (char*)(rhs->bits) + (rhs->size - MIN(rhs->size, 2 * m)) * sizeof(z_cu_type),
                                   (MIN(rhs->size, 2 * m) - m) * sizeof(z_cu_type));

                        z_cu_t* z0;
                        cudaMalloc(&z0, sizeof(z_cu_t));
                        SYNCHRO((z_cu_copy<<<1, 1>>>(x0, z0, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_mul_z<<<1, 1>>>(z0, y0, synchroTmp)), synchroTmp, 1);
                        z_cu_t* z1a;
                        cudaMalloc(&z1a, sizeof(z_cu_t));
                        SYNCHRO((z_cu_copy<<<1, 1>>>(x1, z1a, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_mul_z<<<1, 1>>>(z1a, y0, synchroTmp)), synchroTmp, 1);
                        z_cu_t* z1b;
                        cudaMalloc(&z1b, sizeof(z_cu_t));
                        SYNCHRO((z_cu_copy<<<1, 1>>>(x0, z1b, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_mul_z<<<1, 1>>>(z1b, y1, synchroTmp)), synchroTmp, 1);
                        z_cu_t* z1;
                        cudaMalloc(&z1, sizeof(z_cu_t));
                        SYNCHRO((z_cu_copy<<<1, 1>>>(z1a, z1, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_add_z<<<1, 1>>>(z1, z1b, synchroTmp)), synchroTmp, 1);
                        z_cu_t* z2;
                        cudaMalloc(&z2, sizeof(z_cu_t));
                        SYNCHRO((z_cu_copy<<<1, 1>>>(x1, z2, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_mul_z<<<1, 1>>>(z2, y1, synchroTmp)), synchroTmp, 1);

                        //o = m * 8 * sizeof(z_cu_type)
                        //xy = z2 * 2^(2 * o) + z1 * 2^o + z0

                        SYNCHRO((z_cu_set_from_z<<<1, 1>>>(lhs, z0, synchroTmp)), synchroTmp, 1);

                        z_cu_t* w1;
                        cudaMalloc(&w1, sizeof(z_cu_t));
                        SYNCHRO((z_cu_from_c<<<1, 1>>>(w1, 0, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_set_precision<<<1, 1>>>(w1, z1->size + m, synchroTmp)), synchroTmp, 1);
                        memcpy(w1->bits, z1->bits, z1->size * sizeof(z_cu_type));

                        SYNCHRO((z_cu_add_z<<<1, 1>>>(lhs, w1, synchroTmp)), synchroTmp, 1);

                        z_cu_t* w2;
                        cudaMalloc(&w2, sizeof(z_cu_t));
                        SYNCHRO((z_cu_from_c<<<1, 1>>>(w2, 0, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_set_precision<<<1, 1>>>(w2, z1->size + 2 * m, synchroTmp)), synchroTmp, 1);
                        memcpy(w2->bits, z2->bits, z2->size * sizeof(z_cu_type));

                        SYNCHRO((z_cu_add_z<<<1, 1>>>(lhs, w2, synchroTmp)), synchroTmp, 1);

                        SYNCHRO((z_cu_free<<<1, 1>>>(x0, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_free<<<1, 1>>>(x1, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_free<<<1, 1>>>(y0, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_free<<<1, 1>>>(y1, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_free<<<1, 1>>>(z0, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_free<<<1, 1>>>(z1a, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_free<<<1, 1>>>(z1b, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_free<<<1, 1>>>(z1, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_free<<<1, 1>>>(z2, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_free<<<1, 1>>>(w1, synchroTmp)), synchroTmp, 1);
                        SYNCHRO((z_cu_free<<<1, 1>>>(w2, synchroTmp)), synchroTmp, 1);
                        
                        cudaFree(x0);
                        cudaFree(x1);
                        cudaFree(y0);
                        cudaFree(y1);
                        cudaFree(z0);
                        cudaFree(z1a);
                        cudaFree(z1b);
                        cudaFree(z1);
                        cudaFree(z2);
                        cudaFree(w1);
                        cudaFree(w2);
                    }

                    cudaFree(cmp);
                }

                SYNCHRO((z_cu_free<<<1, 1>>>(rhsTmp, synchroTmp)), synchroTmp, 1);
                cudaFree(rhsTmp);

                cudaFree(fits1);
                cudaFree(fits2);
            }
            else
            {
                z_cu_t* other;
                cudaMalloc(&other, sizeof(z_cu_t));
                SYNCHRO((z_cu_copy<<<1, 1>>>(rhs, other, synchroTmp)), synchroTmp, 1);

                other->is_positive = !other->is_positive;

                SYNCHRO((z_cu_mul_z<<<1, 1>>>(lhs, rhs, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_neg<<<1, 1>>>(lhs, synchroTmp)), synchroTmp, 1);

                SYNCHRO((z_cu_free<<<1, 1>>>(other, synchroTmp)), synchroTmp, 1);
                cudaFree(other);
            }
        }
    }

    if (lhs->is_auto_adjust)
    {
        SYNCHRO((z_cu_adjust<<<1, 1>>>(lhs, synchroTmp)), synchroTmp, 1);
    }

    cudaFree(synchroTmp);

    synchro[idx] = true;
}
