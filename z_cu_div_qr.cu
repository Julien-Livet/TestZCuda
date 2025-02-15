#include <assert.h>
#include <stdlib.h>

#include "z_cu_div_qr.cuh"

#define Z_CU_DIV_QR(suffix, type)                                                                       \
__global__ void z_cu_div_qr_##suffix(z_cu_t const* lhs, type rhs, z_cu_t* q, z_cu_t* r, bool* synchro)  \
{                                                                                                       \
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;                                           \
                                                                                                        \
    z_cu_t* other = 0;                                                                                  \
    cudaMalloc(&other, sizeof(z_cu_t));                                                                 \
                                                                                                        \
    __shared__ bool* synchroTmp;                                                                        \
    cudaMalloc(&synchroTmp, sizeof(bool));                                                              \
                                                                                                        \
    SYNCHRO((z_cu_from_##suffix<<<1, 1>>>(other, rhs, synchroTmp)), synchroTmp, 1);                     \
                                                                                                        \
    SYNCHRO((z_cu_div_qr_z<<<1, 1>>>(lhs, other, q, r, synchroTmp)), synchroTmp, 1);                    \
                                                                                                        \
    SYNCHRO((z_cu_free<<<1, 1>>>(other, synchroTmp)), synchroTmp, 1);                                   \
                                                                                                        \
    cudaFree(synchroTmp);                                                                               \
    cudaFree(other);                                                                                    \
                                                                                                        \
    synchro[idx] = true;                                                                                \
}

Z_CU_DIV_QR(c, char)
Z_CU_DIV_QR(i, int)
Z_CU_DIV_QR(l, long)
Z_CU_DIV_QR(ll, long long)
Z_CU_DIV_QR(s, short)
Z_CU_DIV_QR(uc, unsigned char)
Z_CU_DIV_QR(ui, unsigned int)
Z_CU_DIV_QR(ul, unsigned long)
Z_CU_DIV_QR(ull, unsigned long long)
Z_CU_DIV_QR(us, unsigned short)

__global__ void inner1(z_cu_t* a_digits, z_cu_t const* x, z_cu_t const* L,
                       z_cu_t const* R, z_cu_t const* n, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ bool* synchroTmp;
    cudaMalloc(&synchroTmp, sizeof(bool));

    z_cu_t* l;
    cudaMalloc(&l, sizeof(z_cu_t));
    SYNCHRO((z_cu_copy<<<1, 1>>>(L, l, synchroTmp)), synchroTmp, 1);
    SYNCHRO((z_cu_add_c<<<1, 1>>>(l, 1, synchroTmp)), synchroTmp, 1);

    int* cmp;
    cudaMalloc(&cmp, sizeof(int));
    SYNCHRO((z_cu_cmp_z<<<1, 1>>>(l, R, cmp, synchroTmp)), synchroTmp, 1);

    if (!*cmp)
    {
        unsigned long long* n;
        cudaMalloc(&n, sizeof(unsigned long long));
        SYNCHRO((z_cu_to_ull<<<1, 1>>>(L, n, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_free<<<1, 1>>>(&a_digits[*n], synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_copy<<<1, 1>>>(x, &a_digits[*n], synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_free<<<1, 1>>>(l, synchroTmp)), synchroTmp, 1);
        cudaFree(l);
        cudaFree(cmp);
        cudaFree(n);
    }
    else
    {
        SYNCHRO((z_cu_free<<<1, 1>>>(l, synchroTmp)), synchroTmp, 1);
        cudaFree(l);
        cudaFree(cmp);

        z_cu_t* mid;
        cudaMalloc(&mid, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(L, mid, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_add_z<<<1, 1>>>(mid, R, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_rshift_c<<<1, 1>>>(mid, 1, synchroTmp)), synchroTmp, 1);

        z_cu_t* shift;
        cudaMalloc(&shift, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(mid, shift, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_sub_z<<<1, 1>>>(shift, L, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_mul_z<<<1, 1>>>(shift, n, synchroTmp)), synchroTmp, 1);

        z_cu_t* upper;
        cudaMalloc(&upper, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(x, upper, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_rshift_z<<<1, 1>>>(upper, shift, synchroTmp)), synchroTmp, 1);

        z_cu_t* lower;
        cudaMalloc(&lower, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(upper, lower, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_lshift_z<<<1, 1>>>(lower, shift, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_xor_z<<<1, 1>>>(lower, x, synchroTmp)), synchroTmp, 1);

        SYNCHRO((inner1<<<1, 1>>>(a_digits, lower, L, mid, n, synchroTmp)), synchroTmp, 1);
        SYNCHRO((inner1<<<1, 1>>>(a_digits, upper, mid, R, n, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_free<<<1, 1>>>(mid, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(shift, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(upper, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(lower, synchroTmp)), synchroTmp, 1);
        cudaFree(mid);
        cudaFree(shift);
        cudaFree(upper);
        cudaFree(lower);
    }

    cudaFree(synchroTmp);

    synchro[idx] = true;
}

__global__ void _int2digits(z_cu_t const* a, z_cu_t const* n, z_cu_t** a_digits, size_t* a_size, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    assert(a_digits && a_size);

    __shared__ bool* synchroTmp;
    cudaMalloc(&synchroTmp, sizeof(bool));

    int* cmp;
    cudaMalloc(&cmp, sizeof(int));
    SYNCHRO((z_cu_cmp_c<<<1, 1>>>(a, 0, cmp, synchroTmp)), synchroTmp, 1);

    assert(*cmp >= 0);

    z_cu_t* number;
    cudaMalloc(&number, sizeof(z_cu_t));
    SYNCHRO((z_cu_number<<<1, 1>>>(a, number, synchroTmp)), synchroTmp, 1);
    SYNCHRO((z_cu_add_z<<<1, 1>>>(number, n, synchroTmp)), synchroTmp, 1);
    SYNCHRO((z_cu_sub_c<<<1, 1>>>(number, 1, synchroTmp)), synchroTmp, 1);

    unsigned long long* n1;
    cudaMalloc(&n1, sizeof(unsigned long long));
    unsigned long long* n2;
    cudaMalloc(&n2, sizeof(unsigned long long));

    SYNCHRO((z_cu_to_ull<<<1, 1>>>(number, n1, synchroTmp)), synchroTmp, 1);
    SYNCHRO((z_cu_to_ull<<<1, 1>>>(n, n2, synchroTmp)), synchroTmp, 1);

    *a_size = *n1 / *n2;

    cudaFree(n1);
    cudaFree(n2);

    *a_digits = 0;
    cudaMalloc(a_digits, *a_size * sizeof(z_cu_t));
    assert(*a_digits);

    for (size_t i = 0; i < *a_size; ++i)
    {
        SYNCHRO((z_cu_from_c<<<1, 1>>>(&(*a_digits)[i], 0, synchroTmp)), synchroTmp, 1);
    }

    SYNCHRO((z_cu_free<<<1, 1>>>(number, synchroTmp)), synchroTmp, 1);

    cudaFree(number);

    SYNCHRO((z_cu_cmp_c<<<1, 1>>>(a, 0, cmp, synchroTmp)), synchroTmp, 1);

    if (*cmp)
    {
        z_cu_t* zero;
        cudaMalloc(&zero, sizeof(z_cu_t));
        SYNCHRO((z_cu_from_c<<<1, 1>>>(zero, 0, synchroTmp)), synchroTmp, 1);
        z_cu_t* s;
        cudaMalloc(&s, sizeof(z_cu_t));
        SYNCHRO((z_cu_from_ull<<<1, 1>>>(s, *a_size, synchroTmp)), synchroTmp, 1);

        SYNCHRO((inner1<<<1, 1>>>(*a_digits, a, zero, s, n, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_free<<<1, 1>>>(zero, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(s, synchroTmp)), synchroTmp, 1);

        cudaFree(zero);
        cudaFree(s);
    }

    cudaFree(cmp);

    cudaFree(synchroTmp);

    synchro[idx] = true;
}

__global__ void inner2(z_cu_t const* digits, z_cu_t const* L,
                       z_cu_t const* R, z_cu_t const* n, z_cu_t* result, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ bool* synchroTmp;
    cudaMalloc(&synchroTmp, sizeof(bool));

    z_cu_t *l;
    cudaMalloc(&l, sizeof(z_cu_t));
    SYNCHRO((z_cu_copy<<<1, 1>>>(L, l, synchroTmp)), synchroTmp, 1);
    SYNCHRO((z_cu_add_c<<<1, 1>>>(l, 1, synchroTmp)), synchroTmp, 1);

    int* cmp;
    cudaMalloc(&cmp, sizeof(int));
    
    SYNCHRO((z_cu_cmp_z<<<1, 1>>>(l, R, cmp, synchroTmp)), synchroTmp, 1);

    if (!*cmp)
    {
        cudaFree(cmp);

        SYNCHRO((z_cu_free<<<1, 1>>>(l, synchroTmp)), synchroTmp, 1);
        cudaFree(l);

        unsigned long long* i;
        cudaMalloc(&i, sizeof(unsigned long long));
        SYNCHRO((z_cu_to_ull<<<1, 1>>>(L, i, synchroTmp)), synchroTmp, 1);

        *result = digits[*i];

        cudaFree(i);
    }
    else
    {
        cudaFree(cmp);

        SYNCHRO((z_cu_free<<<1, 1>>>(l, synchroTmp)), synchroTmp, 1);
        cudaFree(l);

        z_cu_t* mid;
        cudaMalloc(&mid, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(L, mid, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_add_z<<<1, 1>>>(mid, R, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_rshift_c<<<1, 1>>>(mid, 1, synchroTmp)), synchroTmp, 1);

        z_cu_t* shift;
        cudaMalloc(&shift, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(mid, shift, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_sub_z<<<1, 1>>>(shift, L, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_mul_z<<<1, 1>>>(shift, n, synchroTmp)), synchroTmp, 1);

        z_cu_t* i1;
        cudaMalloc(&i1, sizeof(z_cu_t));
        SYNCHRO((inner2<<<1, 1>>>(digits, mid, R, n, i1, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_lshift_z<<<1, 1>>>(i1, shift, synchroTmp)), synchroTmp, 1);
        z_cu_t* i2;
        cudaMalloc(&i2, sizeof(z_cu_t));
        SYNCHRO((inner2<<<1, 1>>>(digits, L, mid, n, i1, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_add_z<<<1, 1>>>(i1, i2, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_free<<<1, 1>>>(mid, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(shift, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(i2, synchroTmp)), synchroTmp, 1);

        cudaFree(mid);
        cudaFree(shift);
        cudaFree(i2);

        *result = *i1;
    }

    cudaFree(synchroTmp);

    synchro[idx] = true;
};

__global__ void _digits2int(z_cu_t const* digits, z_cu_t const* n,
                            size_t digits_size, z_cu_t* result, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ bool* synchroTmp;
    cudaMalloc(&synchroTmp, sizeof(bool));

    if (!digits_size)
    {
        SYNCHRO((z_cu_from_c<<<1, 1>>>(result, 0, synchroTmp)), synchroTmp, 1);
    }
    else
    {
        z_cu_t* zero;
        cudaMalloc(&zero, sizeof(z_cu_t));
        SYNCHRO((z_cu_from_c<<<1, 1>>>(zero, 0, synchroTmp)), synchroTmp, 1);
        z_cu_t* s;
        cudaMalloc(&s, sizeof(z_cu_t));
        SYNCHRO((z_cu_from_ull<<<1, 1>>>(s, digits_size, synchroTmp)), synchroTmp, 1);

        z_cu_t* i;
        cudaMalloc(&i, sizeof(z_cu_t));
        SYNCHRO((inner2<<<1, 1>>>(digits, zero, s, n, i, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_free<<<1, 1>>>(zero, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(s, synchroTmp)), synchroTmp, 1);

        cudaFree(zero);
        cudaFree(s);

        *result = *i;
    }

    cudaFree(synchroTmp);

    synchro[idx] = true;
}

__global__ void _div3n2n(z_cu_t const* a12, z_cu_t const* a3, z_cu_t const* b,
                         z_cu_t const* b1, z_cu_t const* b2, z_cu_t const* n,
                         z_cu_t* q, z_cu_t* r, bool* synchro);

__global__ void _div2n1n(z_cu_t const* a, z_cu_t const* b, z_cu_t const* n,
                         z_cu_t* q, z_cu_t* r, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ bool* synchroTmp;
    cudaMalloc(&synchroTmp, sizeof(bool));

    bool* fits1;
    cudaMalloc(&fits1, sizeof(bool));
    bool* fits2;
    cudaMalloc(&fits2, sizeof(bool));

    SYNCHRO((z_cu_fits_ull<<<1, 1>>>(a, fits1, synchroTmp)), synchroTmp, 1);
    SYNCHRO((z_cu_fits_ull<<<1, 1>>>(b, fits2, synchroTmp)), synchroTmp, 1);

    if (*fits1 && *fits2)
    {
        cudaFree(fits1);
        cudaFree(fits2);

        unsigned long long* a_;
        cudaMalloc(&a_, sizeof(unsigned long long));
        unsigned long long* b_;
        cudaMalloc(&b_, sizeof(unsigned long long));

        SYNCHRO((z_cu_to_ull<<<1, 1>>>(a, a_, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_to_ull<<<1, 1>>>(b, b_, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_free<<<1, 1>>>(q, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_set_from_ull<<<1, 1>>>(q, *a_ / *b_, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(r, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_set_from_ull<<<1, 1>>>(r, *a_ % *b_, synchroTmp)), synchroTmp, 1);

        cudaFree(a_);
        cudaFree(b_);
    }
    else
    {
        cudaFree(fits1);
        cudaFree(fits2);

        z_cu_t* pad;
        cudaMalloc(&pad, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(n, pad, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_and_c<<<1, 1>>>(pad, 1, synchroTmp)), synchroTmp, 1);

        z_cu_t *aTmp;
        cudaMalloc(&aTmp, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(a, aTmp, synchroTmp)), synchroTmp, 1);
        z_cu_t *bTmp;
        cudaMalloc(&bTmp, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(b, bTmp, synchroTmp)), synchroTmp, 1);
        z_cu_t *nTmp;
        cudaMalloc(&nTmp, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(n, nTmp, synchroTmp)), synchroTmp, 1);

        int* cmp;
        cudaMalloc(&cmp, sizeof(int));

        SYNCHRO((z_cu_cmp_c<<<1, 1>>>(pad, 0, cmp, synchroTmp)), synchroTmp, 1);

        if (*cmp)
        {
            SYNCHRO((z_cu_lshift_c<<<1, 1>>>(aTmp, 1, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_lshift_c<<<1, 1>>>(bTmp, 1, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_add_c<<<1, 1>>>(nTmp, 1, synchroTmp)), synchroTmp, 1);
        }

        z_cu_t* half_n;
        cudaMalloc(&half_n, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(nTmp, half_n, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_rshift_c<<<1, 1>>>(half_n, 1, synchroTmp)), synchroTmp, 1);

        z_cu_t* mask;
        cudaMalloc(&mask, sizeof(z_cu_t));
        SYNCHRO((z_cu_from_c<<<1, 1>>>(mask, 1, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_lshift_z<<<1, 1>>>(mask, half_n, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_sub_c<<<1, 1>>>(mask, 1, synchroTmp)), synchroTmp, 1);

        z_cu_t* b1;
        cudaMalloc(&b1, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(bTmp, b1, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_rshift_z<<<1, 1>>>(b1, half_n, synchroTmp)), synchroTmp, 1);

        z_cu_t* b2;
        cudaMalloc(&b2, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(bTmp, b2, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_and_z<<<1, 1>>>(b2, mask, synchroTmp)), synchroTmp, 1);

        z_cu_t* q1;
        cudaMalloc(&q1, sizeof(z_cu_t));
        SYNCHRO((z_cu_from_c<<<1, 1>>>(q1, 0, synchroTmp)), synchroTmp, 1);
        z_cu_t* q2;
        cudaMalloc(&q2, sizeof(z_cu_t));
        SYNCHRO((z_cu_from_c<<<1, 1>>>(q2, 0, synchroTmp)), synchroTmp, 1);

        z_cu_t* a1;
        cudaMalloc(&a1, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(aTmp, a1, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_rshift_z<<<1, 1>>>(a1, nTmp, synchroTmp)), synchroTmp, 1);

        z_cu_t* a2;
        cudaMalloc(&a2, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(aTmp, a2, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_rshift_z<<<1, 1>>>(a2, half_n, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_and_z<<<1, 1>>>(a2, mask, synchroTmp)), synchroTmp, 1);

        z_cu_t* a3;
        cudaMalloc(&a3, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(aTmp, a3, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_and_z<<<1, 1>>>(a3, mask, synchroTmp)), synchroTmp, 1);

        SYNCHRO((_div3n2n<<<1, 1>>>(a1, a2, bTmp, b1, b2, half_n, q1, r, synchroTmp)), synchroTmp, 1);
        z_cu_t* rTmp;
        cudaMalloc(&rTmp, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(r, rTmp, synchroTmp)), synchroTmp, 1);
        SYNCHRO((_div3n2n<<<1, 1>>>(rTmp, a3, bTmp, b1, b2, half_n, q2, r, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(rTmp, synchroTmp)), synchroTmp, 1);
        cudaFree(rTmp);

        SYNCHRO((z_cu_cmp_c<<<1, 1>>>(pad, 0, cmp, synchroTmp)), synchroTmp, 1);

        if (*cmp)
        {
            SYNCHRO((z_cu_rshift_c<<<1, 1>>>(r, 1, synchroTmp)), synchroTmp, 1);
        }

        cudaFree(cmp);

        SYNCHRO((z_cu_free<<<1, 1>>>(pad, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(aTmp, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(bTmp, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(nTmp, synchroTmp)), synchroTmp, 1);

        cudaFree(pad);
        cudaFree(aTmp);
        cudaFree(bTmp);
        cudaFree(nTmp);

        SYNCHRO((z_cu_set_from_z<<<1, 1>>>(q, q1, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_lshift_z<<<1, 1>>>(q, half_n, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_or_z<<<1, 1>>>(q, q2, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_free<<<1, 1>>>(half_n, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(mask, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(b1, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(b2, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(q1, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(q2, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(a1, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(a2, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(a3, synchroTmp)), synchroTmp, 1);

        cudaFree(half_n);
        cudaFree(mask);
        cudaFree(b1);
        cudaFree(b2);
        cudaFree(q1);
        cudaFree(q2);
        cudaFree(a1);
        cudaFree(a2);
        cudaFree(a3);
    }

    cudaFree(synchroTmp);

    synchro[idx] = true;
}

__global__ void _div3n2n(z_cu_t const* a12, z_cu_t const* a3, z_cu_t const* b,
                         z_cu_t const* b1, z_cu_t const* b2, z_cu_t const* n,
                         z_cu_t* q, z_cu_t* r, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ bool* synchroTmp;
    cudaMalloc(&synchroTmp, sizeof(bool));

    z_cu_t* a12Tmp;
    cudaMalloc(&a12Tmp, sizeof(z_cu_t));
    SYNCHRO((z_cu_copy<<<1, 1>>>(a12, a12Tmp, synchroTmp)), synchroTmp, 1);
    SYNCHRO((z_cu_rshift_z<<<1, 1>>>(a12Tmp, n, synchroTmp)), synchroTmp, 1);

    int* cmp;
    cudaMalloc(&cmp, sizeof(int));

    SYNCHRO((z_cu_cmp_z<<<1, 1>>>(a12Tmp, b1, cmp, synchroTmp)), synchroTmp, 1);

    if (!*cmp)
    {
        SYNCHRO((z_cu_free<<<1, 1>>>(q, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_set_from_c<<<1, 1>>>(q, 1, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_lshift_z<<<1, 1>>>(q, n, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_sub_c<<<1, 1>>>(q, 1, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_free<<<1, 1>>>(r, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_set_from_z<<<1, 1>>>(r, b1, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_lshift_z<<<1, 1>>>(r, n, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_neg<<<1, 1>>>(r, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_add_z<<<1, 1>>>(r, a12, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_add_z<<<1, 1>>>(r, b1, synchroTmp)), synchroTmp, 1);
    }
    else
    {
        SYNCHRO((_div2n1n<<<1, 1>>>(a12, b1, n, q, r, synchroTmp)), synchroTmp, 1);
    }

    SYNCHRO((z_cu_free<<<1, 1>>>(a12Tmp, synchroTmp)), synchroTmp, 1);
    cudaFree(a12Tmp);

    SYNCHRO((z_cu_lshift_z<<<1, 1>>>(r, n, synchroTmp)), synchroTmp, 1);
    SYNCHRO((z_cu_or_z<<<1, 1>>>(r, a3, synchroTmp)), synchroTmp, 1);

    z_cu_t* tmp;
    cudaMalloc(&tmp, sizeof(z_cu_t));
    SYNCHRO((z_cu_copy<<<1, 1>>>(q, tmp, synchroTmp)), synchroTmp, 1);
    SYNCHRO((z_cu_mul_z<<<1, 1>>>(tmp, b2, synchroTmp)), synchroTmp, 1);
    SYNCHRO((z_cu_sub_z<<<1, 1>>>(r, tmp, synchroTmp)), synchroTmp, 1);

    SYNCHRO((z_cu_free<<<1, 1>>>(tmp, synchroTmp)), synchroTmp, 1);
    cudaFree(tmp);

    SYNCHRO((z_cu_cmp_c<<<1, 1>>>(r, 0, cmp, synchroTmp)), synchroTmp, 1);

    while (*cmp < 0)
    {
        SYNCHRO((z_cu_sub_c<<<1, 1>>>(q, 1, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_add_z<<<1, 1>>>(r, b, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_cmp_c<<<1, 1>>>(r, 0, cmp, synchroTmp)), synchroTmp, 1);
    }
    
    cudaFree(cmp);

    cudaFree(synchroTmp);

    synchro[idx] = true;
}

__global__ void z_cu_div_qr_z(z_cu_t const* lhs, z_cu_t const* rhs, z_cu_t* q, z_cu_t* r, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    assert(q && r);

    __shared__ bool* synchroTmp;
    cudaMalloc(&synchroTmp, sizeof(bool));

    int* cmp1;
    cudaMalloc(&cmp1, sizeof(int));
    SYNCHRO((z_cu_cmp_c<<<1, 1>>>(lhs, 0, cmp1, synchroTmp)), synchroTmp, 1);

    int* cmp2;
    cudaMalloc(&cmp2, sizeof(int));
    SYNCHRO((z_cu_cmp_c<<<1, 1>>>(rhs, 0, cmp2, synchroTmp)), synchroTmp, 1);

    if (!*cmp1)
    {
        SYNCHRO((z_cu_free<<<1, 1>>>(q, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_set_nan<<<1, 1>>>(q, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(r, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_set_nan<<<1, 1>>>(r, synchroTmp)), synchroTmp, 1);
    }
    else if (!*cmp2)
    {
        SYNCHRO((z_cu_free<<<1, 1>>>(q, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_set_from_c<<<1, 1>>>(q, 0, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(r, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_set_from_c<<<1, 1>>>(r, 0, synchroTmp)), synchroTmp, 1);
    }
    else if (lhs->is_nan)
    {
        SYNCHRO((z_cu_free<<<1, 1>>>(q, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_copy<<<1, 1>>>(lhs, q, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(r, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_copy<<<1, 1>>>(lhs, r, synchroTmp)), synchroTmp, 1);
    }
    else if (rhs->is_nan)
    {
        SYNCHRO((z_cu_free<<<1, 1>>>(q, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_copy<<<1, 1>>>(rhs, q, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(r, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_copy<<<1, 1>>>(rhs, r, synchroTmp)), synchroTmp, 1);
    }
    else if (lhs->is_infinity || rhs->is_infinity)
    {
        SYNCHRO((z_cu_free<<<1, 1>>>(q, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_set_nan<<<1, 1>>>(q, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(r, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_set_nan<<<1, 1>>>(r, synchroTmp)), synchroTmp, 1);
    }
    else
    {
        z_cu_t* lhs_abs;
        cudaMalloc(&lhs_abs, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(lhs, lhs_abs, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_abs<<<1, 1>>>(lhs_abs, synchroTmp)), synchroTmp, 1);
        z_cu_t* rhs_abs;
        cudaMalloc(&rhs_abs, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(rhs, rhs_abs, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_abs<<<1, 1>>>(rhs_abs, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_cmp_z<<<1, 1>>>(rhs_abs, lhs_abs, cmp1, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_cmp_c<<<1, 1>>>(rhs_abs, 1, cmp2, synchroTmp)), synchroTmp, 1);

        if (*cmp1 > 0)
        {
            SYNCHRO((z_cu_free<<<1, 1>>>(q, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_set_from_c<<<1, 1>>>(q, 0, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_free<<<1, 1>>>(r, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_set_from_z<<<1, 1>>>(r, lhs, synchroTmp)), synchroTmp, 1);
        }
        else if (!*cmp2)
        {
            SYNCHRO((z_cu_free<<<1, 1>>>(q, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_set_from_z<<<1, 1>>>(q, lhs, synchroTmp)), synchroTmp, 1);

            int* sign;
            cudaMalloc(&sign, sizeof(int));
            SYNCHRO((z_cu_sign<<<1, 1>>>(rhs, sign, synchroTmp)), synchroTmp, 1);

            SYNCHRO((z_cu_mul_i<<<1, 1>>>(q, *sign, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_free<<<1, 1>>>(r, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_set_from_c<<<1, 1>>>(r, 0, synchroTmp)), synchroTmp, 1);

            cudaFree(sign);
        }
        else
        {
            SYNCHRO((z_cu_cmp_c<<<1, 1>>>(lhs, 0, cmp1, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_cmp_c<<<1, 1>>>(rhs, 0, cmp2, synchroTmp)), synchroTmp, 1);

            bool* fits1;
            cudaMalloc(&fits1, sizeof(bool));
            bool* fits2;
            cudaMalloc(&fits2, sizeof(bool));
            
            SYNCHRO((z_cu_fits_ull<<<1, 1>>>(lhs, fits1, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_fits_ull<<<1, 1>>>(rhs, fits2, synchroTmp)), synchroTmp, 1);

            if (*cmp1 < 0 && *cmp2 < 0)
            {
                z_cu_t* dividend;
                cudaMalloc(&dividend, sizeof(z_cu_t));
                SYNCHRO((z_cu_copy<<<1, 1>>>(lhs, dividend, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_neg<<<1, 1>>>(dividend, synchroTmp)), synchroTmp, 1);
                z_cu_t* divisor;
                cudaMalloc(&divisor, sizeof(z_cu_t));
                SYNCHRO((z_cu_copy<<<1, 1>>>(rhs, divisor, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_neg<<<1, 1>>>(divisor, synchroTmp)), synchroTmp, 1);

                SYNCHRO((z_cu_div_qr_z<<<1, 1>>>(dividend, divisor, q, r, synchroTmp)), synchroTmp, 1);

                SYNCHRO((z_cu_free<<<1, 1>>>(dividend, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_free<<<1, 1>>>(divisor, synchroTmp)), synchroTmp, 1);

                cudaFree(dividend);
                cudaFree(divisor);

                SYNCHRO((z_cu_cmp_c<<<1, 1>>>(r, 0, cmp1, synchroTmp)), synchroTmp, 1);

                if (*cmp1)
                {
                    SYNCHRO((z_cu_add_c<<<1, 1>>>(q, 1, synchroTmp)), synchroTmp, 1);
                    SYNCHRO((z_cu_add_z<<<1, 1>>>(r, rhs, synchroTmp)), synchroTmp, 1);
                }
            }
            else if (*cmp1 > 0 && *cmp2 < 0)
            {
                z_cu_t* divisor;
                cudaMalloc(&divisor, sizeof(z_cu_t));
                SYNCHRO((z_cu_copy<<<1, 1>>>(rhs, divisor, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_neg<<<1, 1>>>(divisor, synchroTmp)), synchroTmp, 1);

                SYNCHRO((z_cu_div_qr_z<<<1, 1>>>(lhs, divisor, q, r, synchroTmp)), synchroTmp, 1);

                SYNCHRO((z_cu_free<<<1, 1>>>(divisor, synchroTmp)), synchroTmp, 1);
                cudaFree(divisor);

                SYNCHRO((z_cu_neg<<<1, 1>>>(q, synchroTmp)), synchroTmp, 1);

                SYNCHRO((z_cu_cmp_c<<<1, 1>>>(r, 0, cmp1, synchroTmp)), synchroTmp, 1);

                if (*cmp1)
                {
                    SYNCHRO((z_cu_sub_c<<<1, 1>>>(q, 1, synchroTmp)), synchroTmp, 1);
                    SYNCHRO((z_cu_add_z<<<1, 1>>>(r, rhs, synchroTmp)), synchroTmp, 1);
                }
            }
            else if (*cmp1 < 0 && *cmp2 > 0)
            {
                z_cu_t* dividend;
                cudaMalloc(&dividend, sizeof(z_cu_t));
                SYNCHRO((z_cu_copy<<<1, 1>>>(lhs, dividend, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_neg<<<1, 1>>>(dividend, synchroTmp)), synchroTmp, 1);

                SYNCHRO((z_cu_div_qr_z<<<1, 1>>>(dividend, rhs, q, r, synchroTmp)), synchroTmp, 1);

                SYNCHRO((z_cu_free<<<1, 1>>>(dividend, synchroTmp)), synchroTmp, 1);
                cudaFree(dividend);

                SYNCHRO((z_cu_neg<<<1, 1>>>(q, synchroTmp)), synchroTmp, 1);

                SYNCHRO((z_cu_cmp_c<<<1, 1>>>(r, 0, cmp1, synchroTmp)), synchroTmp, 1);

                if (*cmp1)
                {
                    SYNCHRO((z_cu_sub_c<<<1, 1>>>(q, 1, synchroTmp)), synchroTmp, 1);
                    SYNCHRO((z_cu_neg<<<1, 1>>>(r, synchroTmp)), synchroTmp, 1);
                    SYNCHRO((z_cu_add_z<<<1, 1>>>(r, rhs, synchroTmp)), synchroTmp, 1);
                }
            }
            else if (*fits1 && *fits2)
            {
                unsigned long long* lhs_;
                cudaMalloc(&lhs_, sizeof(unsigned long long));
                unsigned long long* rhs_;
                cudaMalloc(&rhs_, sizeof(unsigned long long));

                SYNCHRO((z_cu_to_ull<<<1, 1>>>(lhs, lhs_, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_to_ull<<<1, 1>>>(rhs, rhs_, synchroTmp)), synchroTmp, 1);

                SYNCHRO((z_cu_free<<<1, 1>>>(q, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_from_ull<<<1, 1>>>(q, *lhs_ / *rhs_, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_free<<<1, 1>>>(r, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_from_ull<<<1, 1>>>(r, *lhs_ % *rhs_, synchroTmp)), synchroTmp, 1);

                cudaFree(lhs_);
                cudaFree(rhs_);
            }
            else
            {
                z_cu_t* n;
                cudaMalloc(&n, sizeof(z_cu_t));
                SYNCHRO((z_cu_number<<<1, 1>>>(rhs, n, synchroTmp)), synchroTmp, 1);
                z_cu_t** a_digits;
                cudaMalloc(&a_digits, sizeof(z_cu_t*));
                size_t* a_digits_size;
                cudaMalloc(&a_digits_size, sizeof(size_t));
                SYNCHRO((_int2digits<<<1, 1>>>(lhs, n, a_digits, a_digits_size, synchroTmp)), synchroTmp, 1);

                z_cu_t* q_digits = 0;
                cudaMalloc(&q_digits, *a_digits_size * sizeof(z_cu_t));
                assert(q_digits);

                for (size_t i = 0; i < *a_digits_size; ++i)
                {
                    z_cu_t* q_digit;
                    cudaMalloc(&q_digit, sizeof(z_cu_t));
                    SYNCHRO((z_cu_from_c<<<1, 1>>>(q_digit, 0, synchroTmp)), synchroTmp, 1);

                    z_cu_t* rTmp;
                    cudaMalloc(&rTmp, sizeof(z_cu_t));
                    SYNCHRO((z_cu_copy<<<1, 1>>>(r, rTmp, synchroTmp)), synchroTmp, 1);
                    SYNCHRO((z_cu_lshift_z<<<1, 1>>>(rTmp, n, synchroTmp)), synchroTmp, 1);

                    SYNCHRO((z_cu_add_z<<<1, 1>>>(rTmp, &(*a_digits)[*a_digits_size - 1 - i], synchroTmp)), synchroTmp, 1);

                    SYNCHRO((_div2n1n<<<1, 1>>>(rTmp, rhs, n, q_digit, r, synchroTmp)), synchroTmp, 1);

                    q_digits[i] = *q_digit;

                    SYNCHRO((z_cu_free<<<1, 1>>>(rTmp, synchroTmp)), synchroTmp, 1);
                    cudaFree(rTmp);
                }

                for (size_t i = 0; i < *a_digits_size / 2; ++i)
                {
                    z_cu_t tmp = q_digits[i];
                    q_digits[i] = q_digits[*a_digits_size - 1 - i];
                    q_digits[*a_digits_size - 1 - i] = tmp;
                }

                SYNCHRO((z_cu_free<<<1, 1>>>(n, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_free<<<1, 1>>>(q, synchroTmp)), synchroTmp, 1);
                SYNCHRO((_digits2int<<<1, 1>>>(q_digits, n, *a_digits_size, q, synchroTmp)), synchroTmp, 1);

                cudaFree(a_digits_size);
                cudaFree(a_digits);
                cudaFree(q_digits);
            }

            cudaFree(fits1);
            cudaFree(fits2);
        }
        
        SYNCHRO((z_cu_free<<<1, 1>>>(lhs_abs, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(rhs_abs, synchroTmp)), synchroTmp, 1);
        
        cudaFree(lhs_abs);
        cudaFree(rhs_abs);
    }

    cudaFree(cmp1);
    cudaFree(cmp2);

    cudaFree(synchroTmp);

    synchro[idx] = true;
}
