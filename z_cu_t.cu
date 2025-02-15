#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <curand.h>
#include <curand_kernel.h>

#include "z_cu_t.cuh"

__global__ void z_cu_abs(z_cu_t* z, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    z->is_positive = true;

    synchro[idx] = true;
}

__global__ void z_cu_adjust(z_cu_t* z, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    assert(z);

    if (z->size)
    {
        assert(z->bits);

        size_t i = 0;

        while (!z->bits[i] && i < z->size)
            ++i;

        if (i == z->size)
            i = z->size - 1;

        if (i != 0)
        {
            z_cu_type* bits = 0;
            cudaMalloc(&bits, (z->size - i) * sizeof(z_cu_type));
            assert(bits);
            memcpy(bits, (char*)(z->bits) + i * sizeof(z_cu_type), (z->size - i) * sizeof(z_cu_type));
            cudaFree(z->bits);
            z->bits = bits;
            z->size = z->size - i;
        }
    }

    synchro[idx] = true;
}

__global__ void z_cu_copy(z_cu_t const* from, z_cu_t* to, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    *to = *from;

    to->bits = 0;
    cudaMalloc(&to->bits, to->size * sizeof(z_cu_type));
    assert(to->bits);
    memcpy(to->bits, from->bits, to->size * sizeof(z_cu_type));

    synchro[idx] = true;
}
/*
z_cu_t z_cu_factorial(z_cu_t n)
{
    z_cu_t res = z_cu_from_c(0);

    if (z_cu_is_nan(n) || z_cu_is_infinity(n))
    {
        z_cu_free(&res);
        res = z_cu_copy(n);

        return res;
    }

    assert(z_cu_cmp_c(n, 0) >= 0);

    if (!z_cu_cmp_c(n, 0))
    {
        z_cu_free(&res);
        res = z_cu_from_c(1);

        return res;
    }

    z_cu_free(&res);
    res = z_cu_copy(n);

    z_cu_t n_1 = z_cu_copy(n);
    z_cu_sub_c(&n_1, 1);

    z_cu_t f = z_cu_factorial(n_1);

    z_cu_mul_z(&res, f);

    z_cu_free(&n_1);
    z_cu_free(&f);

    return res;
}
*/
__global__ void z_cu_free(z_cu_t* z, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    assert(z);

    cudaFree(z->bits);

    z->bits = NULL;
    z->size = 0;
    z->is_nan = true;
    z->is_infinity = false;

    synchro[idx] = true;
}

void z_cu_free2(z_cu_t* z)
{
    assert(z);

    int* tmp;
    cudaMemcpy(&tmp, &z->bits, sizeof(int*), cudaMemcpyDeviceToHost);
    cudaFree(tmp);

    z_type const* null = NULL;

    cudaMemcpy(&z->bits, null, sizeof(z_type*), cudaMemcpyDeviceToHost);
    cudaMemset(&z->size, 0, sizeof(size_t));
    cudaMemset(&z->is_nan, true, sizeof(bool));
    cudaMemset(&z->is_infinity, false, sizeof(bool));
}

__global__ void z_cu_gcd(z_cu_t const* a, z_cu_t const* b, z_cu_t* gcd, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ bool* synchroTmp;
    cudaMalloc(&synchroTmp, sizeof(bool));

    int* cmp1;
    cudaMalloc(&cmp1, sizeof(int));
    SYNCHRO((z_cu_cmp_c<<<1, 1>>>(a, 0, cmp1, synchroTmp)), synchroTmp, 1);

    int* cmp2;
    cudaMalloc(&cmp2, sizeof(int));
    SYNCHRO((z_cu_cmp_c<<<1, 1>>>(b, 0, cmp2, synchroTmp)), synchroTmp, 1);

    int* cmp3;
    cudaMalloc(&cmp3, sizeof(int));
    SYNCHRO((z_cu_cmp_z<<<1, 1>>>(a, b, cmp3, synchroTmp)), synchroTmp, 1);

    bool* a_is_even;
    cudaMalloc(&a_is_even, sizeof(bool));
    SYNCHRO((z_cu_is_even<<<1, 1>>>(a, a_is_even, synchroTmp)), synchroTmp, 1);

    bool const a_is_odd = !*a_is_even;

    bool* b_is_even;
    cudaMalloc(&b_is_even, sizeof(bool));
    SYNCHRO((z_cu_is_even<<<1, 1>>>(b, b_is_even, synchroTmp)), synchroTmp, 1);

    bool const b_is_odd = !*b_is_even;

    if (a->is_nan || b->is_nan || a->is_infinity || b->is_infinity)
    {
        SYNCHRO((z_cu_set_nan<<<1, 1>>>(gcd, synchroTmp)), synchroTmp, 1);
    }
    else if (*cmp1 < 0)
    {
        z_cu_t* aTmp;
        cudaMalloc(&aTmp, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(a, aTmp, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_abs<<<1, 1>>>(aTmp, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_gcd<<<1, 1>>>(aTmp, b, gcd, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_free<<<1, 1>>>(aTmp, synchroTmp)), synchroTmp, 1);

        cudaFree(aTmp);
    }
    else if (*cmp2 < 0)
    {
        z_cu_t* bTmp;
        cudaMalloc(&bTmp, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(b, bTmp, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_abs<<<1, 1>>>(bTmp, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_gcd<<<1, 1>>>(a, bTmp, gcd, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_free<<<1, 1>>>(bTmp, synchroTmp)), synchroTmp, 1);
    }
    else if (*cmp3 < 0)
    {
        SYNCHRO((z_cu_gcd<<<1, 1>>>(b, a, gcd, synchroTmp)), synchroTmp, 1);
    }
    else if (!*cmp1)
    {
        SYNCHRO((z_cu_copy<<<1, 1>>>(b, gcd, synchroTmp)), synchroTmp, 1);
    }
    else if (!*cmp2)
    {
        SYNCHRO((z_cu_copy<<<1, 1>>>(a, gcd, synchroTmp)), synchroTmp, 1);
    }
    else if (a_is_even && b_is_even)
    {
        z_cu_t* aTmp;
        cudaMalloc(&aTmp, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(a, aTmp, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_rshift_c<<<1, 1>>>(aTmp, 1, synchroTmp)), synchroTmp, 1);
        z_cu_t* bTmp;
        cudaMalloc(&bTmp, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(b, bTmp, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_rshift_c<<<1, 1>>>(bTmp, 1, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_gcd<<<1, 1>>>(aTmp, bTmp, gcd, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_rshift_c<<<1, 1>>>(gcd, 1, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_free<<<1, 1>>>(aTmp, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(bTmp, synchroTmp)), synchroTmp, 1);
        cudaFree(aTmp);
        cudaFree(bTmp);
    }
    else if (a_is_odd && b_is_even)
    {
        z_cu_t* bTmp;
        cudaMalloc(&bTmp, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(b, bTmp, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_rshift_c<<<1, 1>>>(bTmp, 1, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_gcd<<<1, 1>>>(a, bTmp, gcd, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_free<<<1, 1>>>(bTmp, synchroTmp)), synchroTmp, 1);
        cudaFree(bTmp);
    }
    else if (a_is_even && b_is_odd)
    {
        z_cu_t* aTmp;
        cudaMalloc(&aTmp, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(a, aTmp, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_rshift_c<<<1, 1>>>(aTmp, 1, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_gcd<<<1, 1>>>(aTmp, b, gcd, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_free<<<1, 1>>>(aTmp, synchroTmp)), synchroTmp, 1);
        cudaFree(aTmp);
    }
    else //if (a_is_odd && b_is_odd)
    {
        z_cu_t* aTmp;
        cudaMalloc(&aTmp, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(a, aTmp, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_sub_z<<<1, 1>>>(aTmp, b, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_rshift_c<<<1, 1>>>(aTmp, 1, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_gcd<<<1, 1>>>(aTmp, b, gcd, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_free<<<1, 1>>>(aTmp, synchroTmp)), synchroTmp, 1);
        cudaFree(aTmp);
    }

    cudaFree(cmp1);
    cudaFree(cmp2);
    cudaFree(cmp3);
    cudaFree(a_is_even);
    cudaFree(b_is_even);
    cudaFree(synchroTmp);

    synchro[idx] = true;
}

__global__ void z_cu_gcd_extended(z_cu_t const* a, z_cu_t const* b, z_cu_t* u, z_cu_t* v, z_cu_t* gcd, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    assert(u && v);

    __shared__ bool* synchroTmp;
    cudaMalloc(&synchroTmp, sizeof(bool));

    int* cmp1;
    cudaMalloc(&cmp1, sizeof(int));
    SYNCHRO((z_cu_cmp_c<<<1, 1>>>(a, 0, cmp1, synchroTmp)), synchroTmp, 1);

    int* cmp2;
    cudaMalloc(&cmp2, sizeof(int));
    SYNCHRO((z_cu_cmp_c<<<1, 1>>>(b, 0, cmp2, synchroTmp)), synchroTmp, 1);

    if (a->is_nan || b->is_nan || a->is_infinity || b->is_infinity)
    {
        SYNCHRO((z_cu_set_nan<<<1, 1>>>(u, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_set_nan<<<1, 1>>>(v, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_set_nan<<<1, 1>>>(gcd, synchroTmp)), synchroTmp, 1);
    }
    else if (!*cmp1 && !*cmp2)
    {
        SYNCHRO((z_cu_from_c<<<1, 1>>>(gcd, 0, synchroTmp)), synchroTmp, 1);
    }
    else
    {
        z_cu_t* r1;
        cudaMalloc(&r1, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(a, r1, synchroTmp)), synchroTmp, 1);
        z_cu_t* u1;
        cudaMalloc(&u1, sizeof(z_cu_t));
        SYNCHRO((z_cu_from_c<<<1, 1>>>(u1, 1, synchroTmp)), synchroTmp, 1);
        z_cu_t* v1;
        cudaMalloc(&v1, sizeof(z_cu_t));
        SYNCHRO((z_cu_from_c<<<1, 1>>>(v1, 0, synchroTmp)), synchroTmp, 1);
        z_cu_t* r2;
        cudaMalloc(&r2, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(b, r2, synchroTmp)), synchroTmp, 1);
        z_cu_t* u2;
        cudaMalloc(&u2, sizeof(z_cu_t));
        SYNCHRO((z_cu_from_c<<<1, 1>>>(u2, 0, synchroTmp)), synchroTmp, 1);
        z_cu_t* v2;
        cudaMalloc(&v2, sizeof(z_cu_t));
        SYNCHRO((z_cu_from_c<<<1, 1>>>(v2, 1, synchroTmp)), synchroTmp, 1);
        z_cu_t* q;
        cudaMalloc(&q, sizeof(z_cu_t));
        SYNCHRO((z_cu_from_c<<<1, 1>>>(q, 0, synchroTmp)), synchroTmp, 1);
        z_cu_t* r_temp;
        cudaMalloc(&r_temp, sizeof(z_cu_t));
        SYNCHRO((z_cu_from_c<<<1, 1>>>(r_temp, 0, synchroTmp)), synchroTmp, 1);
        z_cu_t* u_temp;
        cudaMalloc(&u_temp, sizeof(z_cu_t));
        SYNCHRO((z_cu_from_c<<<1, 1>>>(u_temp, 0, synchroTmp)), synchroTmp, 1);
        z_cu_t* v_temp;
        cudaMalloc(&v_temp, sizeof(z_cu_t));
        SYNCHRO((z_cu_from_c<<<1, 1>>>(v_temp, 0, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_cmp_c<<<1, 1>>>(r2, 0, cmp1, synchroTmp)), synchroTmp, 1);

        while (*cmp1)
        {
            SYNCHRO((z_cu_free<<<1, 1>>>(q, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_div_q_z<<<1, 1>>>(r1, r2, q, synchroTmp)), synchroTmp, 1);

            SYNCHRO((z_cu_free<<<1, 1>>>(r_temp, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_copy<<<1, 1>>>(r2, r_temp, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_neg<<<1, 1>>>(r_temp, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_mul_z<<<1, 1>>>(r_temp, q, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_add_z<<<1, 1>>>(r_temp, r1, synchroTmp)), synchroTmp, 1);

            SYNCHRO((z_cu_free<<<1, 1>>>(u_temp, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_copy<<<1, 1>>>(u2, u_temp, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_neg<<<1, 1>>>(u_temp, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_mul_z<<<1, 1>>>(u_temp, q, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_add_z<<<1, 1>>>(u_temp, u1, synchroTmp)), synchroTmp, 1);

            SYNCHRO((z_cu_free<<<1, 1>>>(v_temp, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_copy<<<1, 1>>>(v2, v_temp, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_neg<<<1, 1>>>(v_temp, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_mul_z<<<1, 1>>>(v_temp, q, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_add_z<<<1, 1>>>(v_temp, v1, synchroTmp)), synchroTmp, 1);

            SYNCHRO((z_cu_free<<<1, 1>>>(r1, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_copy<<<1, 1>>>(r2, r1, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_free<<<1, 1>>>(u1, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_copy<<<1, 1>>>(u2, u1, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_free<<<1, 1>>>(v1, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_copy<<<1, 1>>>(v2, v1, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_free<<<1, 1>>>(r2, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_copy<<<1, 1>>>(r_temp, r2, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_free<<<1, 1>>>(u2, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_copy<<<1, 1>>>(u_temp, u2, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_free<<<1, 1>>>(v2, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_copy<<<1, 1>>>(v_temp, v2, synchroTmp)), synchroTmp, 1);

            SYNCHRO((z_cu_cmp_c<<<1, 1>>>(r2, 0, cmp1, synchroTmp)), synchroTmp, 1);
        }

        SYNCHRO((z_cu_free<<<1, 1>>>(u, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_copy<<<1, 1>>>(u1, u, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(v, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_copy<<<1, 1>>>(v1, v, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_free<<<1, 1>>>(u1, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(v1, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(r2, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(u2, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(v2, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(q, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(r_temp, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(u_temp, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(v_temp, synchroTmp)), synchroTmp, 1);
        cudaFree(u1);
        cudaFree(v1);
        cudaFree(r2);
        cudaFree(u2);
        cudaFree(v2);
        cudaFree(q);
        cudaFree(r_temp);
        cudaFree(u_temp);
        cudaFree(v_temp);

        SYNCHRO((z_cu_copy<<<1, 1>>>(r1, gcd, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_free<<<1, 1>>>(r1, synchroTmp)), synchroTmp, 1);
        cudaFree(r1);
    }

    cudaFree(cmp1);
    cudaFree(cmp2);
    cudaFree(synchroTmp);

    synchro[idx] = true;
}
/*
z_cu_t z_cu_infinity()
{
    z_cu_t z;

    z.is_positive = true;
    z.bits = NULL;
    z.size = 0;
    z.is_nan = false;
    z.is_infinity = true;
    z.is_auto_adjust = true;

    return z;
}
*/
__global__ void invert_uintmax_t(uintmax_t* a, size_t n, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n)
        a[idx] = ~a[idx];

    synchro[idx] = true;
}

__global__ void z_cu_invert(z_cu_t* z, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    assert(z);

    assert(z->bits);

    size_t const blockSize = BLOCK_SIZE;
    size_t const gridSize = (z->size + blockSize) / blockSize;
    
    __shared__ bool* synchroTmp;
    cudaMalloc(&synchroTmp, sizeof(bool) * gridSize * blockSize);

    SYNCHRO((invert_uintmax_t<<<gridSize, blockSize>>>(z->bits, z->size, synchroTmp)), synchroTmp, gridSize * blockSize);

    z->is_positive = !z->is_positive;

    cudaFree(synchroTmp);

    synchro[idx] = true;
}

__global__ void z_cu_is_even(z_cu_t const* z, bool* even, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (!z->size)
        *even = false;

    *even = !(z->bits[z->size - 1] & 1);

    synchro[idx] = true;
}
/*
bool z_cu_is_negative(z_cu_t z)
{
    return !z.is_positive;
}

bool z_cu_is_null(z_cu_t z)
{
    return !z_cu_cmp_c(z, 0);
}
*/
__global__ void z_cu_is_odd(z_cu_t const* z, bool* odd, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (!z->size)
        *odd = false;

    *odd = z->bits[z->size - 1] & 1;

    synchro[idx] = true;
}
/*
bool z_cu_is_positive(z_cu_t z)
{
    return z.is_positive;
}

z_cu_t z_cu_max(z_cu_t a, z_cu_t b)
{
    if (z_cu_cmp_z(a, b) > 0)
        return a;
    else
        return b;
}

z_cu_t z_cu_min(z_cu_t a, z_cu_t b)
{
    if (z_cu_cmp_z(a, b) < 0)
        return a;
    else
        return b;
}

z_cu_t z_cu_nan()
{
    z_cu_t z;

    z.is_positive = true;
    z.bits = NULL;
    z.size = 0;
    z.is_nan = true;
    z.is_infinity = false;
    z.is_auto_adjust = true;

    return z;
}
*/
__global__ void z_cu_neg(z_cu_t* z, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    assert(z);

    z->is_positive = !z->is_positive;

    synchro[idx] = true;
}

__global__ void z_cu_number(z_cu_t const* z, z_cu_t* number, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    assert(z);
    assert(number);

    __shared__ bool* synchroTmp;
    cudaMalloc(&synchroTmp, sizeof(bool));

    SYNCHRO((z_cu_from_c<<<1, 1>>>(number, 0, synchroTmp)), synchroTmp, 1);

    if (!(z->is_nan || z->is_infinity))
    {
        size_t i = 0;

        while (!z->bits[i] && i < z->size)
            ++i;

        if (i < z->size)
        {
            z_cu_type b = z->bits[i];

            while (b)
            {
                SYNCHRO((z_cu_add_c<<<1, 1>>>(number, 1, synchroTmp)), synchroTmp, 1);
                b >>= 1;
            }

            SYNCHRO((z_cu_add_ull<<<1, 1>>>(number, (z->size - i - 1) * sizeof(z_cu_type) * 8, synchroTmp)), synchroTmp, 1);
        }
    }

    cudaFree(synchroTmp);

    synchro[idx] = true;
}
/**
__global__ void z_cu_printf(z_cu_t z, size_t base)
{
    char* s = z_cu_to_str(z, base);

    printf(s);

    free(s);
}
**/
__global__ void z_cu_printf_bits(z_cu_t const* z, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = 0; i < z->size; ++i)
        printf("%llu ", z->bits[i]);

    synchro[idx] = true;
}

__global__ void z_cu_printf_bytes(z_cu_t const* z, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = 0; i < z->size * sizeof(z_cu_type); ++i)
        printf("%d ", *((char*)(z->bits) + i));

    synchro[idx] = true;
}

__global__ void z_cu_set_auto_adjust(z_cu_t* z, bool is_auto_adjust, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    assert(z);

    z->is_auto_adjust = is_auto_adjust;

    synchro[idx] = true;
}

__global__ void z_cu_set_infinity(z_cu_t* z, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    assert(z);

    cudaFree(z->bits);
    z->bits = NULL;
    z->size = 0;
    z->is_nan = false;
    z->is_infinity = true;

    synchro[idx] = true;
}

__global__ void z_cu_set_nan(z_cu_t* z, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    assert(z);

    cudaFree(z->bits);
    z->bits = NULL;
    z->size = 0;
    z->is_nan = true;
    z->is_infinity = false;

    synchro[idx] = true;
}

__global__ void z_cu_set_negative(z_cu_t* z)
{
    assert(z);

    z->is_positive = false;
}

__global__ void z_cu_set_positive(z_cu_t* z)
{
    assert(z);

    z->is_positive = true;
}

__global__ void z_cu_set_precision(z_cu_t* z, size_t precision, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    assert(z);
    assert(precision);

    if (precision != z->size)
    {
        z_cu_type* bits = z->bits;
        z->bits = 0;
        cudaMalloc(&z->bits, precision * sizeof(z_cu_type));
        assert(z->bits);

        if (precision > z->size)
        {
            memset(z->bits, 0, (precision - z->size) * sizeof(z_cu_type));
            memcpy((char*)(z->bits) + (precision - z->size) * sizeof(z_cu_type), bits, z->size * sizeof(z_cu_type));
        }
        else
            memcpy(z->bits, (char*)(bits) + (z->size - precision) * sizeof(z_cu_type), precision * sizeof(z_cu_type));

        z->size = precision;
        cudaFree(bits);
    }

    synchro[idx] = true;
}

__global__ void set_random_uintmax_t(uintmax_t* a, size_t n, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n)
    {
        curandState state;
        curand_init(0, idx, 0, &state);
        
        a[idx] = curand(&state);
        a[idx] <<= 32;
        a[idx] |= curand(&state);
    }

    synchro[idx] = true;
}

__global__ void z_cu_set_random(z_cu_t* z, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    assert(z);

    z->is_nan = false;
    z->is_infinity = false;

    curandState state;
    curand_init(0, 0, 0, &state);
    
    z->is_positive = curand(&state) % 2;

    size_t const blockSize = BLOCK_SIZE;
    size_t const gridSize = (z->size + blockSize) / blockSize;

    __shared__ bool* synchroTmp;
    cudaMalloc(&synchroTmp, sizeof(bool) * gridSize * blockSize);

    SYNCHRO((set_random_uintmax_t<<<gridSize, blockSize>>>(z->bits, z->size, synchroTmp)), synchroTmp, gridSize * blockSize);

    cudaFree(synchroTmp);

    synchro[idx] = true;
}

__global__ void z_cu_sign(z_cu_t const* z, int* sign, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (z->is_positive)
        *sign = 1;
    else
        *sign = -1;

    synchro[idx] = true;
}

__global__ void z_cu_sqrt(z_cu_t const* n, z_cu_t* sqrt, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ bool* synchroTmp;
    cudaMalloc(&synchroTmp, sizeof(bool));

    SYNCHRO((z_cu_from_c<<<1, 1>>>(sqrt, 0, synchroTmp)), synchroTmp, 1);

    int* cmp1;
    cudaMalloc(&cmp1, sizeof(int));
    int* cmp2;
    cudaMalloc(&cmp2, sizeof(int));

    SYNCHRO((z_cu_cmp_c<<<1, 1>>>(n, 0, cmp1, synchroTmp)), synchroTmp, 1);
    SYNCHRO((z_cu_cmp_c<<<1, 1>>>(n, 1, cmp2, synchroTmp)), synchroTmp, 1);

    if (*cmp1 < 0)
    {
        SYNCHRO((z_cu_set_nan<<<1, 1>>>(sqrt, synchroTmp)), synchroTmp, 1);
    }
    else if (!*cmp1 || !*cmp2 || n->is_nan || n->is_infinity)
    {
        SYNCHRO((z_cu_free<<<1, 1>>>(sqrt, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_copy<<<1, 1>>>(n, sqrt, synchroTmp)), synchroTmp, 1);
    }
    else
    {printf("sqrt0\n");
        SYNCHRO((z_cu_free<<<1, 1>>>(sqrt, synchroTmp)), synchroTmp, 1);

        z_cu_t* lo;
        cudaMalloc(&lo, sizeof(z_cu_t));
        SYNCHRO((z_cu_from_c<<<1, 1>>>(lo, 1, synchroTmp)), synchroTmp, 1);
        z_cu_t* hi;
        cudaMalloc(&hi, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(n, hi, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_from_c<<<1, 1>>>(sqrt, 1, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_cmp_z<<<1, 1>>>(lo, hi, cmp1, synchroTmp)), synchroTmp, 1);
        printf("sqrt1\n");
        while (*cmp1 <= 0)
        {printf("sqrt2\n");
            z_cu_t* mid;
            cudaMalloc(&mid, sizeof(z_cu_t));
            printf("sqrt3\n");
            SYNCHRO((z_cu_copy<<<1, 1>>>(lo, mid, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_add_z<<<1, 1>>>(mid, hi, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_rshift_c<<<1, 1>>>(mid, 1, synchroTmp)), synchroTmp, 1);
            printf("sqrt4\n");
            z_cu_t* mid2;
            cudaMalloc(&mid2, sizeof(z_cu_t));printf("sqrt4a\n");
            SYNCHRO((z_cu_copy<<<1, 1>>>(mid, mid2, synchroTmp)), synchroTmp, 1);printf("sqrt4b\n");
            SYNCHRO_((z_cu_mul_z<<<1, 1>>>(mid2, mid, synchroTmp)), synchroTmp, 1);printf("sqrt4c\n");
            printf("sqrt5\n");
            SYNCHRO((z_cu_cmp_z<<<1, 1>>>(mid2, n, cmp1, synchroTmp)), synchroTmp, 1);
            printf("sqrt6\n");
            if (*cmp1 <= 0)
            {
                SYNCHRO((z_cu_free<<<1, 1>>>(sqrt, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_copy<<<1, 1>>>(mid, sqrt, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_free<<<1, 1>>>(lo, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_copy<<<1, 1>>>(mid, lo, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_add_c<<<1, 1>>>(lo, 1, synchroTmp)), synchroTmp, 1);
            }
            else
            {
                SYNCHRO((z_cu_free<<<1, 1>>>(hi, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_copy<<<1, 1>>>(mid, hi, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_sub_c<<<1, 1>>>(hi, 1, synchroTmp)), synchroTmp, 1);
            }

            SYNCHRO((z_cu_free<<<1, 1>>>(mid, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_free<<<1, 1>>>(mid2, synchroTmp)), synchroTmp, 1);
            cudaFree(mid);
            cudaFree(mid2);

            SYNCHRO((z_cu_cmp_z<<<1, 1>>>(lo, hi, cmp1, synchroTmp)), synchroTmp, 1);
        }
        
        SYNCHRO((z_cu_free<<<1, 1>>>(lo, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(hi, synchroTmp)), synchroTmp, 1);
        cudaFree(lo);
        cudaFree(hi);
    }

    cudaFree(cmp1);
    cudaFree(cmp2);

    cudaFree(synchroTmp);

    synchro[idx] = true;
}
