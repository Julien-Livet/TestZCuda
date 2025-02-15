#include <assert.h>
#include <stdlib.h>
#include <time.h>

#include "z_cu_prime.cuh"

__global__ void z_cu_is_coprime(z_cu_t const* a, z_cu_t const* b, bool* result, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    z_cu_t* gcd;
    cudaMalloc(&gcd, sizeof(z_cu_t));

    __shared__ bool* synchroTmp;
    cudaMalloc(&synchroTmp, sizeof(bool));

    SYNCHRO((z_cu_gcd<<<1, 1>>>(a, b, gcd, synchroTmp)), synchroTmp, 1);

    int* cmp;
    cudaMalloc(&cmp, sizeof(int));

    SYNCHRO((z_cu_cmp_c<<<1, 1>>>(gcd, 1, cmp, synchroTmp)), synchroTmp, 1);

    *result = !*cmp;

    cudaFree(cmp);

    SYNCHRO((z_cu_free<<<1, 1>>>(gcd, synchroTmp)), synchroTmp, 1);

    cudaFree(synchroTmp);
    cudaFree(gcd);

    synchro[idx] = true;
}

//#include "primes_3_000_000.h"
#include "primes_100.h"

void cudaPrimes(unsigned int** p, size_t* size)
{
    unsigned int* tmp;
    cudaMalloc(&tmp, sizeof(unsigned int) * PRIMES_SIZE);
    *p = tmp;

    size_t const s = PRIMES_SIZE;

    cudaMemcpy(size, &s, sizeof(size_t), cudaMemcpyHostToDevice);

    cudaMemcpy(tmp, primes, sizeof(unsigned int) * PRIMES_SIZE, cudaMemcpyHostToDevice);
}

__global__ void lower_bound(unsigned int const* array, size_t first, size_t last,
                            unsigned int value, size_t* bound, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    long long count = last - first;

    while (count > 0)
    {
        size_t i = first;
        long long step = count / 2;
        i += step;

        if (array[i] < value)
        {
            first = ++i;
            count -= step + 1;
        }
        else
            count = step;
    }

    *bound = first;

    synchro[idx] = true;
}

__global__ void upper_bound(unsigned int const* array, size_t first, size_t last,
                            unsigned int value, size_t* bound, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    long long count = last - first;

    while (count > 0)
    {
        size_t i = first;
        long long step = count / 2;
        i += step;

        if (!(value < array[i]))
        {
            first = ++i;
            count -= step + 1;
        }
        else
            count = step;
    }

    *bound = first;

    synchro[idx] = true;
}

struct pair_struct
{
    size_t first;
    size_t second;
};

typedef struct pair_struct pair;

__global__ void equal_range(unsigned int const* array, size_t first, size_t last,
                            unsigned int value, pair* p, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ bool* synchroTmp;
    cudaMalloc(&synchroTmp, sizeof(bool));

    SYNCHRO((lower_bound<<<1, 1>>>(array, first, last, value, &p->first, synchroTmp)), synchroTmp, 1);
    SYNCHRO((upper_bound<<<1, 1>>>(array, first, last, value, &p->second, synchroTmp)), synchroTmp, 1);

    cudaFree(synchroTmp);

    synchro[idx] = true;
}

__global__ void reduction(z_cu_t const* t, z_cu_t const* R, z_cu_t const* n,
                          z_cu_t const* n_, z_cu_t* red, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ bool* synchroTmp;
    cudaMalloc(&synchroTmp, sizeof(bool));

    z_cu_t* m;
    cudaMalloc(&m, sizeof(z_cu_t));
    SYNCHRO((z_cu_div_r_z<<<1, 1>>>(t, R, m, synchroTmp)), synchroTmp, 1);
    SYNCHRO((z_cu_mul_z<<<1, 1>>>(m, n_, synchroTmp)), synchroTmp, 1);
    z_cu_t* mTmp;
    cudaMalloc(&mTmp, sizeof(z_cu_t));
    SYNCHRO((z_cu_div_r_z<<<1, 1>>>(m, R, mTmp, synchroTmp)), synchroTmp, 1);
    SYNCHRO((z_cu_free<<<1, 1>>>(m, synchroTmp)), synchroTmp, 1);
    *m = *mTmp;

    z_cu_t* x;
    cudaMalloc(&x, sizeof(z_cu_t));
    SYNCHRO((z_cu_copy<<<1, 1>>>(m, x, synchroTmp)), synchroTmp, 1);
    SYNCHRO((z_cu_mul_z<<<1, 1>>>(x, n, synchroTmp)), synchroTmp, 1);
    SYNCHRO((z_cu_add_z<<<1, 1>>>(x, t, synchroTmp)), synchroTmp, 1);
    z_cu_t* xTmp;
    cudaMalloc(&xTmp, sizeof(z_cu_t));
    SYNCHRO((z_cu_div_q_z<<<1, 1>>>(x, R, xTmp, synchroTmp)), synchroTmp, 1);
    SYNCHRO((z_cu_free<<<1, 1>>>(x, synchroTmp)), synchroTmp, 1);
    *x = *xTmp;

    SYNCHRO((z_cu_free<<<1, 1>>>(m, synchroTmp)), synchroTmp, 1);
    cudaFree(m);

    int* cmp;
    cudaMalloc(&cmp, sizeof(int));

    SYNCHRO((z_cu_cmp_z<<<1, 1>>>(x, n, cmp, synchroTmp)), synchroTmp, 1);

    if (*cmp >= 0)
    {
        SYNCHRO((z_cu_sub_z<<<1, 1>>>(x, n, synchroTmp)), synchroTmp, 1);
    }

    cudaFree(cmp);

    SYNCHRO((z_cu_copy<<<1, 1>>>(x, red, synchroTmp)), synchroTmp, 1);
    SYNCHRO((z_cu_free<<<1, 1>>>(x, synchroTmp)), synchroTmp, 1);
    cudaFree(x);

    cudaFree(synchroTmp);

    synchro[idx] = true;
}

__global__ void redmulmod(z_cu_t const* a, z_cu_t const* b, z_cu_t const* n,
                          z_cu_t const* R, z_cu_t const* n_, z_cu_t const* R2modn, z_cu_t* r, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ bool* synchroTmp;
    cudaMalloc(&synchroTmp, sizeof(bool));

    z_cu_t* tmp;
    cudaMalloc(&tmp, sizeof(z_cu_t));
    SYNCHRO((z_cu_copy<<<1, 1>>>(a, tmp, synchroTmp)), synchroTmp, 1);
    SYNCHRO((z_cu_mul_z<<<1, 1>>>(tmp, R2modn, synchroTmp)), synchroTmp, 1);

    z_cu_t* reda;
    cudaMalloc(&reda, sizeof(z_cu_t));
    SYNCHRO((reduction<<<1, 1>>>(tmp, R, n, n_, reda, synchroTmp)), synchroTmp, 1);

    SYNCHRO((z_cu_free<<<1, 1>>>(tmp, synchroTmp)), synchroTmp, 1);
    SYNCHRO((z_cu_copy<<<1, 1>>>(b, tmp, synchroTmp)), synchroTmp, 1);
    SYNCHRO((z_cu_mul_z<<<1, 1>>>(tmp, R2modn, synchroTmp)), synchroTmp, 1);

    z_cu_t* redb;
    cudaMalloc(&redb, sizeof(z_cu_t));
    SYNCHRO((reduction<<<1, 1>>>(tmp, R, n, n_, redb, synchroTmp)), synchroTmp, 1);

    SYNCHRO((z_cu_free<<<1, 1>>>(tmp, synchroTmp)), synchroTmp, 1);
    SYNCHRO((z_cu_copy<<<1, 1>>>(reda, tmp, synchroTmp)), synchroTmp, 1);
    SYNCHRO((z_cu_mul_z<<<1, 1>>>(tmp, redb, synchroTmp)), synchroTmp, 1);

    z_cu_t* redc;
    cudaMalloc(&redc, sizeof(z_cu_t));
    SYNCHRO((reduction<<<1, 1>>>(tmp, R, n, n_, redc, synchroTmp)), synchroTmp, 1);

    SYNCHRO((z_cu_free<<<1, 1>>>(tmp, synchroTmp)), synchroTmp, 1);
    cudaFree(tmp);

    SYNCHRO((reduction<<<1, 1>>>(redc, R, n, n_, r, synchroTmp)), synchroTmp, 1);

    SYNCHRO((z_cu_free<<<1, 1>>>(reda, synchroTmp)), synchroTmp, 1);
    SYNCHRO((z_cu_free<<<1, 1>>>(redb, synchroTmp)), synchroTmp, 1);
    SYNCHRO((z_cu_free<<<1, 1>>>(redc, synchroTmp)), synchroTmp, 1);
    cudaFree(reda);
    cudaFree(redb);
    cudaFree(redc);

    cudaFree(synchroTmp);

    synchro[idx] = true;
}

__global__ void mulmod(z_cu_t const* a, z_cu_t const* b, z_cu_t const* m, bool* res, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ bool* synchroTmp;
    cudaMalloc(&synchroTmp, sizeof(bool));

    z_cu_t* x;
    cudaMalloc(&x, sizeof(z_cu_t));
    SYNCHRO((z_cu_from_c<<<1, 1>>>(x, 0, synchroTmp)), synchroTmp, 1);
    z_cu_t* y;
    cudaMalloc(&y, sizeof(z_cu_t));
    SYNCHRO((z_cu_div_r_z<<<1, 1>>>(a, m, y, synchroTmp)), synchroTmp, 1);
    z_cu_t* bTmp;
    cudaMalloc(&bTmp, sizeof(z_cu_t));
    SYNCHRO((z_cu_copy<<<1, 1>>>(b, bTmp, synchroTmp)), synchroTmp, 1);

    int* cmp;
    cudaMalloc(&cmp, sizeof(int));

    SYNCHRO((z_cu_cmp_c<<<1, 1>>>(bTmp, 0, cmp, synchroTmp)), synchroTmp, 1);

    while (*cmp > 0)
    {
        z_cu_t* tmp;
        cudaMalloc(&tmp, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(bTmp, tmp, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_and_c<<<1, 1>>>(tmp, 1, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_cmp_c<<<1, 1>>>(tmp, 0, cmp, synchroTmp)), synchroTmp, 1);

        if (*cmp)
        {
            SYNCHRO((z_cu_add_z<<<1, 1>>>(x, y, synchroTmp)), synchroTmp, 1);

            z_cu_t* r;
            cudaMalloc(&r, sizeof(z_cu_t));
            SYNCHRO((z_cu_div_r_z<<<1, 1>>>(x, m, r, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_free<<<1, 1>>>(x, synchroTmp)), synchroTmp, 1);
            *x = *r;
        }

        SYNCHRO((z_cu_free<<<1, 1>>>(tmp, synchroTmp)), synchroTmp, 1);
        cudaFree(tmp);

        SYNCHRO((z_cu_lshift_c<<<1, 1>>>(y, 1, synchroTmp)), synchroTmp, 1);

        z_cu_t* r;
        cudaMalloc(&r, sizeof(z_cu_t));
        SYNCHRO((z_cu_div_r_z<<<1, 1>>>(y, m, r, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(y, synchroTmp)), synchroTmp, 1);
        *y = *r;
        cudaFree(r);

        SYNCHRO((z_cu_rshift_c<<<1, 1>>>(bTmp, 1, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_cmp_c<<<1, 1>>>(bTmp, 0, cmp, synchroTmp)), synchroTmp, 1);
    }

    z_cu_t* r;
    cudaMalloc(&r, sizeof(z_cu_t));
    SYNCHRO((z_cu_div_r_z<<<1, 1>>>(x, m, r, synchroTmp)), synchroTmp, 1);

    SYNCHRO((z_cu_cmp_c<<<1, 1>>>(r, 0, cmp, synchroTmp)), synchroTmp, 1);

    *res = *cmp;

    cudaFree(cmp);

    SYNCHRO((z_cu_free<<<1, 1>>>(r, synchroTmp)), synchroTmp, 1);
    SYNCHRO((z_cu_free<<<1, 1>>>(x, synchroTmp)), synchroTmp, 1);
    SYNCHRO((z_cu_free<<<1, 1>>>(y, synchroTmp)), synchroTmp, 1);
    SYNCHRO((z_cu_free<<<1, 1>>>(bTmp, synchroTmp)), synchroTmp, 1);
    cudaFree(r);
    cudaFree(x);
    cudaFree(y);
    cudaFree(bTmp);

    cudaFree(synchroTmp);

    synchro[idx] = true;
}

__global__ void modulo(z_cu_t const* base, z_cu_t const* e, z_cu_t const* m,
                       z_cu_t const* R, z_cu_t const* m_, z_cu_t const* R2modm, z_cu_t* res, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ bool* synchroTmp;
    cudaMalloc(&synchroTmp, sizeof(bool));

    z_cu_t* x;
    cudaMalloc(&x, sizeof(z_cu_t));
    SYNCHRO((z_cu_from_c<<<1, 1>>>(x, 1, synchroTmp)), synchroTmp, 1);
    z_cu_t* y;
    cudaMalloc(&y, sizeof(z_cu_t));
    SYNCHRO((z_cu_copy<<<1, 1>>>(base, y, synchroTmp)), synchroTmp, 1);
    z_cu_t* eTmp;
    cudaMalloc(&eTmp, sizeof(z_cu_t));
    SYNCHRO((z_cu_copy<<<1, 1>>>(e, eTmp, synchroTmp)), synchroTmp, 1);

    int* cmp;
    cudaMalloc(&cmp, sizeof(int));

    SYNCHRO((z_cu_cmp_c<<<1, 1>>>(eTmp, 0, cmp, synchroTmp)), synchroTmp, 1);

    while (*cmp > 0)
    {
        z_cu_t* tmp;
        cudaMalloc(&tmp, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(eTmp, tmp, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_and_c<<<1, 1>>>(tmp, 1, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_cmp_c<<<1, 1>>>(tmp, 0, cmp, synchroTmp)), synchroTmp, 1);

        if (*cmp)
        {
            z_cu_t* xTmp;
            cudaMalloc(&xTmp, sizeof(z_cu_t));
            SYNCHRO((redmulmod<<<1, 1>>>(x, y, m, R, m_, R2modm, xTmp, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_free<<<1, 1>>>(x, synchroTmp)), synchroTmp, 1);
            *x = *xTmp;

            SYNCHRO((z_cu_cmp_c<<<1, 1>>>(x, 0, cmp, synchroTmp)), synchroTmp, 1);

            while (*cmp < 0)
            {
                SYNCHRO((z_cu_add_z<<<1, 1>>>(x, m, synchroTmp)), synchroTmp, 1);

                SYNCHRO((z_cu_cmp_c<<<1, 1>>>(x, 0, cmp, synchroTmp)), synchroTmp, 1);
            }
        }

        SYNCHRO((z_cu_free<<<1, 1>>>(tmp, synchroTmp)), synchroTmp, 1);
        cudaFree(tmp);

        z_cu_t* yTmp;
        cudaMalloc(&yTmp, sizeof(z_cu_t));
        SYNCHRO((redmulmod<<<1, 1>>>(y, y, m, R, m_, R2modm, yTmp, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(y, synchroTmp)), synchroTmp, 1);
        *y = *yTmp;

        SYNCHRO((z_cu_cmp_c<<<1, 1>>>(y, 0, cmp, synchroTmp)), synchroTmp, 1);

        while (*cmp < 0)
        {
            SYNCHRO((z_cu_add_z<<<1, 1>>>(y, m, synchroTmp)), synchroTmp, 1);

            SYNCHRO((z_cu_cmp_c<<<1, 1>>>(y, 0, cmp, synchroTmp)), synchroTmp, 1);
        }

        SYNCHRO((z_cu_rshift_c<<<1, 1>>>(eTmp, 1, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_cmp_c<<<1, 1>>>(eTmp, 0, cmp, synchroTmp)), synchroTmp, 1);
    }

    cudaFree(cmp);

    SYNCHRO((z_cu_div_r_z<<<1, 1>>>(x, m, res, synchroTmp)), synchroTmp, 1);

    SYNCHRO((z_cu_free<<<1, 1>>>(x, synchroTmp)), synchroTmp, 1);
    SYNCHRO((z_cu_free<<<1, 1>>>(y, synchroTmp)), synchroTmp, 1);
    SYNCHRO((z_cu_free<<<1, 1>>>(eTmp, synchroTmp)), synchroTmp, 1);
    cudaFree(x);
    cudaFree(y);
    cudaFree(eTmp);

    cudaFree(synchroTmp);

    synchro[idx] = true;
}

__global__ void is_prime_trial_division(unsigned int const* primes, z_cu_t const* number,
                                        z_cu_t const* sqrtLimit, size_t size, bool* divisible, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size && !*divisible)
    {
        __shared__ bool* synchroTmp;
        cudaMalloc(&synchroTmp, sizeof(bool));
        
        z_cu_t* r;
        cudaMalloc(&r, sizeof(z_cu_t));
        SYNCHRO((z_cu_div_r_ui<<<1, 1>>>(number, primes[idx], r, synchroTmp)), synchroTmp, 1);
        
        int* cmp1;
        cudaMalloc(&cmp1, sizeof(int));
        SYNCHRO((z_cu_cmp_ui<<<1, 1>>>(sqrtLimit, primes[idx], cmp1, synchroTmp)), synchroTmp, 1);
        
        int* cmp2;
        cudaMalloc(&cmp2, sizeof(int));
        SYNCHRO((z_cu_cmp_c<<<1, 1>>>(r, 0, cmp2, synchroTmp)), synchroTmp, 1);
        
        if (*cmp1 >= 0 && !*cmp2)
            *divisible = true;
        
        cudaFree(cmp1);
        cudaFree(cmp2);
        
        SYNCHRO((z_cu_free<<<1, 1>>>(r, synchroTmp)), synchroTmp, 1);
        
        cudaFree(r);

        cudaFree(synchroTmp);
    }
    
    synchro[idx] = true;
}

__global__ void is_prime_miller_rabin_test(z_cu_t const* n, z_cu_t const* number,
                                           z_cu_t const* s, z_cu_t const* R, z_cu_t const* R_,
                                           z_cu_t const* m_, z_cu_t const* R2modm,
                                           size_t size, int* result, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size && *result)
    {
        __shared__ bool* synchroTmp;
        cudaMalloc(&synchroTmp, sizeof(bool));

    	z_cu_t* a;;
        cudaMalloc(&a, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(n, a, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_set_random<<<1, 1>>>(a, synchroTmp)), synchroTmp, 1);
        a->is_positive = true;
        z_cu_t* tmp;
        cudaMalloc(&tmp, sizeof(z_cu_t));
        SYNCHRO((z_cu_div_r_z<<<1, 1>>>(a, number, tmp, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(a, synchroTmp)), synchroTmp, 1);
        *a = *tmp;
        SYNCHRO((z_cu_add_c<<<1, 1>>>(a, 1, synchroTmp)), synchroTmp, 1);

        z_cu_t* temp;
        cudaMalloc(&temp, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(s, temp, synchroTmp)), synchroTmp, 1);
        z_cu_t* mod;
        cudaMalloc(&mod, sizeof(z_cu_t));
        SYNCHRO((modulo<<<1, 1>>>(a, temp, n, R, m_, R2modm, mod, synchroTmp)), synchroTmp, 1);

        int* cmp1;
        cudaMalloc(&cmp1, sizeof(int));
        SYNCHRO((z_cu_cmp_z<<<1, 1>>>(temp, number, cmp1, synchroTmp)), synchroTmp, 1);

        int* cmp2;
        cudaMalloc(&cmp2, sizeof(int));
        SYNCHRO((z_cu_cmp_c<<<1, 1>>>(mod, 0, cmp2, synchroTmp)), synchroTmp, 1);
        
        int* cmp3;
        cudaMalloc(&cmp3, sizeof(int));
        SYNCHRO((z_cu_cmp_z<<<1, 1>>>(mod, number, cmp3, synchroTmp)), synchroTmp, 1);

        while (*cmp1 && !*cmp2 && *cmp3)
        {
            bool* tmp;
            cudaMalloc(&tmp, sizeof(bool));
            SYNCHRO((mulmod<<<1, 1>>>(mod, mod, n, tmp, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_free<<<1, 1>>>(mod, synchroTmp)), synchroTmp, 1);

            if (*tmp)
            {
                SYNCHRO((z_cu_set_from_c<<<1, 1>>>(mod, 1, synchroTmp)), synchroTmp, 1);
            }
            else
            {
                SYNCHRO((z_cu_set_from_c<<<1, 1>>>(mod, 0, synchroTmp)), synchroTmp, 1);
            }

            cudaFree(tmp);

            SYNCHRO((z_cu_lshift_c<<<1, 1>>>(temp, 1, synchroTmp)), synchroTmp, 1);

            SYNCHRO((z_cu_cmp_z<<<1, 1>>>(temp, number, cmp1, synchroTmp)), synchroTmp, 1);
            SYNCHRO((z_cu_cmp_c<<<1, 1>>>(mod, 0, cmp2, synchroTmp)), synchroTmp, 1);
        }

        SYNCHRO((z_cu_copy<<<1, 1>>>(temp, tmp, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_and_c<<<1, 1>>>(tmp, 1, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_cmp_z<<<1, 1>>>(mod, number, cmp1, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_cmp_c<<<1, 1>>>(tmp, 0, cmp2, synchroTmp)), synchroTmp, 1);

        if (*cmp1 && !*cmp2)
            *result = 0;

        cudaFree(cmp1);
        cudaFree(cmp2);
        cudaFree(cmp3);

        SYNCHRO((z_cu_free<<<1, 1>>>(tmp, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(temp, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_free<<<1, 1>>>(mod, synchroTmp)), synchroTmp, 1);
    	SYNCHRO((z_cu_free<<<1, 1>>>(a, synchroTmp)), synchroTmp, 1);
        cudaFree(tmp);
        cudaFree(temp);
        cudaFree(mod);
        cudaFree(a);
    }

    synchro[idx] = true;
}

__global__ void z_cu_is_prime(z_cu_t const* n, size_t reps, int* result,
                              unsigned int const* p, size_t primes_size, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    assert(reps);

    int* cmp;
    cudaMalloc(&cmp, sizeof(int));

    __shared__ bool* synchroTmp;
    cudaMalloc(&synchroTmp, sizeof(bool));

    SYNCHRO((z_cu_cmp_c<<<1, 1>>>(n, 2, cmp, synchroTmp)), synchroTmp, 1);

    if (*cmp < 0)
        *result = 0;
    else if (!*cmp)
        *result = 2;
    else
    {
        z_cu_t* tmp;
        cudaMalloc(&tmp, sizeof(z_cu_t));

        SYNCHRO((z_cu_copy<<<1, 1>>>(n, tmp, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_and_c<<<1, 1>>>(tmp, 1, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_cmp_c<<<1, 1>>>(tmp, 0, cmp, synchroTmp)), synchroTmp, 1);
        
        if (!*cmp)
        {
            SYNCHRO((z_cu_free<<<1, 1>>>(tmp, synchroTmp)), synchroTmp, 1);
        
            cudaFree(tmp);

            *result = 0;
        }
        else
        {
            SYNCHRO((z_cu_free<<<1, 1>>>(tmp, synchroTmp)), synchroTmp, 1);

            cudaFree(tmp);
            
            bool found = false;

            bool* fits;
            cudaMalloc(&fits, sizeof(bool));
            SYNCHRO((z_cu_fits_ui<<<1, 1>>>(n, fits, synchroTmp)), synchroTmp, 1);
            
            if (*fits)
            {
                pair* pr;
                cudaMalloc(&pr, sizeof(pair));
                
                unsigned int* ui;
                cudaMalloc(&ui, sizeof(unsigned int));
                
                SYNCHRO((z_cu_to_ui<<<1, 1>>>(n, ui, synchroTmp)), synchroTmp, 1);
                
                SYNCHRO((equal_range<<<1, 1>>>(p, 0, primes_size, *ui, pr, synchroTmp)), synchroTmp, 1);
                
                if (pr->first != primes_size && p[pr->first] != *ui)
                    --pr->first;
                
                if (pr->first != pr->second && p[pr->first] == *ui)
                {
                    found = true;
                    *result = 2;
                }

                cudaFree(ui);
            }

            cudaFree(fits);

            //Trial divisions

            if (!found)
            {
                printf("Tial divisions\n");

                z_cu_t* sqrtLimit;
                cudaMalloc(&sqrtLimit, sizeof(z_cu_t));printf("here-1\n");
                SYNCHRO((z_cu_sqrt<<<1, 1>>>(n, sqrtLimit, synchroTmp)), synchroTmp, 1);
                printf("here0\n");
                bool* divisible(0);
                cudaMalloc(&divisible, sizeof(bool));
                *divisible = false;
                printf("here1\n");
                size_t const blockSize = BLOCK_SIZE;
                size_t const gridSize = (primes_size + blockSize) / blockSize;
                printf("here2\n");
                __shared__ bool* synch;
                cudaMalloc(&synch, sizeof(bool) * gridSize * blockSize);
                printf("here3\n");
                SYNCHRO((is_prime_trial_division<<<gridSize, blockSize>>>(p, n, sqrtLimit, primes_size, divisible, synch)), synch, gridSize * blockSize);
                printf("here4\n");
                cudaFree(synch);

                if (*divisible)
                {
                    *result = 0;
                    found = true;
                }
                
                SYNCHRO((z_cu_free<<<1, 1>>>(sqrtLimit, synchroTmp)), synchroTmp, 1);
                
                cudaFree(sqrtLimit);
                cudaFree(divisible);
            }

            //Miller-Rabin tests

            if (!found)
            {
                printf("Miller-Rabin tests\n");

                z_cu_t* s;
                cudaMalloc(&s, sizeof(z_cu_t));
                SYNCHRO((z_cu_copy<<<1, 1>>>(n, s, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_sub_c<<<1, 1>>>(s, 1, synchroTmp)), synchroTmp, 1);

                SYNCHRO((z_cu_copy<<<1, 1>>>(s, tmp, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_and_c<<<1, 1>>>(tmp, 1, synchroTmp)), synchroTmp, 1);

                SYNCHRO((z_cu_cmp_c<<<1, 1>>>(tmp, 0, cmp, synchroTmp)), synchroTmp, 1);

                while (!*cmp)
                {
                    SYNCHRO((z_cu_rshift_c<<<1, 1>>>(s, 1, synchroTmp)), synchroTmp, 1);
                    SYNCHRO((z_cu_free<<<1, 1>>>(tmp, synchroTmp)), synchroTmp, 1);
                    SYNCHRO((z_cu_copy<<<1, 1>>>(s, tmp, synchroTmp)), synchroTmp, 1);
                    SYNCHRO((z_cu_and_c<<<1, 1>>>(tmp, 1, synchroTmp)), synchroTmp, 1);

                    SYNCHRO((z_cu_cmp_c<<<1, 1>>>(tmp, 0, cmp, synchroTmp)), synchroTmp, 1);
                }

                SYNCHRO((z_cu_free<<<1, 1>>>(tmp, synchroTmp)), synchroTmp, 1);

                z_cu_t* number;
                cudaMalloc(&number, sizeof(z_cu_t));
                SYNCHRO((z_cu_copy<<<1, 1>>>(n, number, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_sub_c<<<1, 1>>>(number, 1, synchroTmp)), synchroTmp, 1);

                z_cu_t* m;
                cudaMalloc(&m, sizeof(z_cu_t));
                SYNCHRO((z_cu_copy<<<1, 1>>>(n, m, synchroTmp)), synchroTmp, 1);

                z_cu_t* R;
                cudaMalloc(&R, sizeof(z_cu_t));
                SYNCHRO((z_cu_from_c<<<1, 1>>>(R, 1, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_number<<<1, 1>>>(m, tmp, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_lshift_z<<<1, 1>>>(R, tmp, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_free<<<1, 1>>>(tmp, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_copy<<<1, 1>>>(R, tmp, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_sub_z<<<1, 1>>>(tmp, m, synchroTmp)), synchroTmp, 1);

                SYNCHRO((z_cu_cmp_c<<<1, 1>>>(tmp, 0, cmp, synchroTmp)), synchroTmp, 1);

                assert(*cmp > 0);
                SYNCHRO((z_cu_free<<<1, 1>>>(tmp, synchroTmp)), synchroTmp, 1);

                SYNCHRO((z_cu_copy<<<1, 1>>>(m, tmp, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_and_c<<<1, 1>>>(tmp, 1, synchroTmp)), synchroTmp, 1);

                SYNCHRO((z_cu_cmp_c<<<1, 1>>>(tmp, 0, cmp, synchroTmp)), synchroTmp, 1);

                if (!*cmp)
                {
                    SYNCHRO((z_cu_add_c<<<1, 1>>>(R, 1, synchroTmp)), synchroTmp, 1);
                }

                SYNCHRO((z_cu_free<<<1, 1>>>(tmp, synchroTmp)), synchroTmp, 1);

                bool* is_coprime;
                cudaMalloc(&is_coprime, sizeof(bool));

                SYNCHRO((z_cu_is_coprime<<<1, 1>>>(m, R, is_coprime, synchroTmp)), synchroTmp, 1);

                while (!*is_coprime)
                {
                    SYNCHRO((z_cu_copy<<<1, 1>>>(m, tmp, synchroTmp)), synchroTmp, 1);
                    SYNCHRO((z_cu_and_c<<<1, 1>>>(tmp, 1, synchroTmp)), synchroTmp, 1);

                    SYNCHRO((z_cu_cmp_c<<<1, 1>>>(tmp, 0, cmp, synchroTmp)), synchroTmp, 1);

                    if (!*cmp)
                    {
                        SYNCHRO((z_cu_sub_c<<<1, 1>>>(R, 1, synchroTmp)), synchroTmp, 1);
                    }

                    SYNCHRO((z_cu_free<<<1, 1>>>(tmp, synchroTmp)), synchroTmp, 1);

                    SYNCHRO((z_cu_lshift_c<<<1, 1>>>(R, 1, synchroTmp)), synchroTmp, 1);

                    SYNCHRO((z_cu_copy<<<1, 1>>>(m, tmp, synchroTmp)), synchroTmp, 1);
                    SYNCHRO((z_cu_and_c<<<1, 1>>>(tmp, 1, synchroTmp)), synchroTmp, 1);

                    SYNCHRO((z_cu_cmp_c<<<1, 1>>>(tmp, 0, cmp, synchroTmp)), synchroTmp, 1);

                    if (!*cmp)
                    {
                        SYNCHRO((z_cu_add_c<<<1, 1>>>(R, 1, synchroTmp)), synchroTmp, 1);
                    }

                    SYNCHRO((z_cu_free<<<1, 1>>>(tmp, synchroTmp)), synchroTmp, 1);

                    SYNCHRO((z_cu_is_coprime<<<1, 1>>>(m, R, is_coprime, synchroTmp)), synchroTmp, 1);
                }

                z_cu_t* R2modm;
                cudaMalloc(&R2modm, sizeof(z_cu_t));
                SYNCHRO((z_cu_copy<<<1, 1>>>(R, R2modm, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_mul_z<<<1, 1>>>(R2modm, R, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_div_r_z<<<1, 1>>>(R2modm, m, tmp, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_free<<<1, 1>>>(R2modm, synchroTmp)), synchroTmp, 1);
                *R2modm = *tmp;
                z_cu_t* R_;
                cudaMalloc(&R_, sizeof(z_cu_t));
                SYNCHRO((z_cu_from_c<<<1, 1>>>(R_, 0, synchroTmp)), synchroTmp, 1);
                z_cu_t* m_;
                cudaMalloc(&m_, sizeof(z_cu_t));
                SYNCHRO((z_cu_from_c<<<1, 1>>>(m_, 0, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_copy<<<1, 1>>>(m, tmp, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_neg<<<1, 1>>>(tmp, synchroTmp)), synchroTmp, 1);
                z_cu_t* d;
                cudaMalloc(&d, sizeof(z_cu_t));
                SYNCHRO((z_cu_gcd_extended<<<1, 1>>>(R, tmp, R_, m_, d, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_free<<<1, 1>>>(tmp, synchroTmp)), synchroTmp, 1);
                cudaFree(tmp);

                int* cmp1;
                cudaMalloc(&cmp1, sizeof(int));
                SYNCHRO((z_cu_cmp_c<<<1, 1>>>(d, 1, cmp1, synchroTmp)), synchroTmp, 1);

                int* cmp2;
                cudaMalloc(&cmp2, sizeof(int));
                SYNCHRO((z_cu_cmp_c<<<1, 1>>>(d, -1, cmp2, synchroTmp)), synchroTmp, 1);

                assert(!*cmp1 || !*cmp2);

                cudaFree(cmp1);
                cudaFree(cmp2);

                z_cu_t* tmp1;
                cudaMalloc(&tmp1, sizeof(z_cu_t));
                SYNCHRO((z_cu_copy<<<1, 1>>>(R, tmp1, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_mul_z<<<1, 1>>>(tmp1, R_, synchroTmp)), synchroTmp, 1);
                z_cu_t* tmp2;
                cudaMalloc(&tmp2, sizeof(z_cu_t));
                SYNCHRO((z_cu_copy<<<1, 1>>>(m, tmp2, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_mul_z<<<1, 1>>>(tmp2, m_, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_sub_z<<<1, 1>>>(tmp1, tmp2, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_cmp_z<<<1, 1>>>(tmp1, d, cmp, synchroTmp)), synchroTmp, 1);
                assert(*cmp == 0);
                SYNCHRO((z_cu_free<<<1, 1>>>(tmp1, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_free<<<1, 1>>>(tmp2, synchroTmp)), synchroTmp, 1);
                cudaFree(tmp1);
                cudaFree(tmp2);

                SYNCHRO((z_cu_cmp_c<<<1, 1>>>(d, -1, cmp, synchroTmp)), synchroTmp, 1);

                if (!*cmp)
                {
                    SYNCHRO((z_cu_neg<<<1, 1>>>(R_, synchroTmp)), synchroTmp, 1);
                    SYNCHRO((z_cu_neg<<<1, 1>>>(m_, synchroTmp)), synchroTmp, 1);
                }

                *result = 1;

                size_t const blockSize = BLOCK_SIZE;
                size_t const gridSize = (PRIMES_SIZE + blockSize) / blockSize;
                        
                __shared__ bool* synch;
                cudaMalloc(&synch, sizeof(bool) * gridSize * blockSize);
  
                SYNCHRO((is_prime_miller_rabin_test<<<gridSize, blockSize>>>(n, number, s, R, R_, m_,
                                                                             R2modm, reps, result, synch)), synch, gridSize * blockSize);
                            
                cudaFree(synch);

                SYNCHRO((z_cu_free<<<1, 1>>>(R_, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_free<<<1, 1>>>(m_, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_free<<<1, 1>>>(d, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_free<<<1, 1>>>(s, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_free<<<1, 1>>>(m, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_free<<<1, 1>>>(R, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_free<<<1, 1>>>(R2modm, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_free<<<1, 1>>>(number, synchroTmp)), synchroTmp, 1);
                cudaFree(R_);
                cudaFree(m_);
                cudaFree(d);
                cudaFree(s);
                cudaFree(m);
                cudaFree(R);
                cudaFree(R2modm);
                cudaFree(number);
            }
        }
    }

    cudaFree(synchroTmp);
    cudaFree(cmp);

    synchro[idx] = true;
}

__global__ void next_prime(z_cu_t const* n, z_cu_t* prime, size_t size,
                           unsigned int const* p, size_t primes_size, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ bool* synchroTmp;
    cudaMalloc(&synchroTmp, sizeof(bool));

    size_t start = blockIdx.x * blockDim.x + threadIdx.x + 1;

    z_cu_t* number;
    cudaMalloc(&number, sizeof(z_cu_t));

    int* cmp;
    cudaMalloc(&cmp, sizeof(int));
    SYNCHRO((z_cu_cmp_z<<<1, 1>>>(number, prime, cmp, synchroTmp)), synchroTmp, 1);

    while (*cmp < 0)
    {
        SYNCHRO((z_cu_free<<<1, 1>>>(number, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_set_from_ull<<<1, 1>>>(number, start, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_mul_c<<<1, 1>>>(number, 2, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_add_z<<<1, 1>>>(number, n, synchroTmp)), synchroTmp, 1);
        
        int* result;
        cudaMalloc(&result, sizeof(int));
        SYNCHRO((z_cu_is_prime<<<1, 1>>>(number, 25, result, p, primes_size, synchroTmp)), synchroTmp, 1);

        if (*result)
        {
            SYNCHRO((z_cu_cmp_z<<<1, 1>>>(number, prime, cmp, synchroTmp)), synchroTmp, 1);

            if (*cmp < 0)
            {
                SYNCHRO((z_cu_free<<<1, 1>>>(prime, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_set_from_z<<<1, 1>>>(prime, number, synchroTmp)), synchroTmp, 1);
            }
        }
        else
            start += size;

        cudaFree(result);

        SYNCHRO((z_cu_cmp_z<<<1, 1>>>(number, prime, cmp, synchroTmp)), synchroTmp, 1);
    }

    cudaFree(number);

    cudaFree(synchroTmp);

    synchro[idx] = true;
}

__global__ void z_cu_next_prime(z_cu_t const* n, z_cu_t* prime,
                                unsigned int const* p, size_t primes_size, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ bool* synchroTmp;
    cudaMalloc(&synchroTmp, sizeof(bool));

    int* cmp;
    cudaMalloc(&cmp, sizeof(int));

    SYNCHRO((z_cu_cmp_c<<<1, 1>>>(n, 2, cmp, synchroTmp)), synchroTmp, 1);

    bool* fits;
    cudaMalloc(&fits, sizeof(bool));

    SYNCHRO((z_cu_fits_ui<<<1, 1>>>(n, fits, synchroTmp)), synchroTmp, 1);

    bool found = false;

    if (n->is_nan)
    {
        found = true;
        SYNCHRO((z_cu_copy<<<1, 1>>>(n, prime, synchroTmp)), synchroTmp, 1);
    }
    else if (*cmp < 0)
    {
        found = true;
        SYNCHRO((z_cu_from_c<<<1, 1>>>(prime, 2, synchroTmp)), synchroTmp, 1);
    }
    else if (!*cmp)
    {
        found = true;
        SYNCHRO((z_cu_from_c<<<1, 1>>>(prime, 3, synchroTmp)), synchroTmp, 1);
    }
    else if (n->is_infinity)
    {
        found = true;
        SYNCHRO((z_cu_set_nan<<<1, 1>>>(prime, synchroTmp)), synchroTmp, 1);
    }
    else if (*fits)
    {
        pair* pr;
        cudaMalloc(&pr, sizeof(pair));

        unsigned int* ui;
        cudaMalloc(&ui, sizeof(unsigned int));

        SYNCHRO((z_cu_to_ui<<<1, 1>>>(n, ui, synchroTmp)), synchroTmp, 1);

        SYNCHRO((equal_range<<<1, 1>>>(p, 0, primes_size, *ui, pr, synchroTmp)), synchroTmp, 1);

        if (pr->first != primes_size && p[pr->first] != *ui)
            --pr->first;

        if (pr->first != pr->second && pr->second != primes_size)
        {
            SYNCHRO((z_cu_from_ui<<<1, 1>>>(prime, p[pr->second], synchroTmp)), synchroTmp, 1);
            found = true;
        }

        cudaFree(pr);
        cudaFree(ui);
    }
    
    if (!found)
    {
        SYNCHRO((z_cu_copy<<<1, 1>>>(n, prime, synchroTmp)), synchroTmp, 1);
        z_cu_t* tmp;
        cudaMalloc(&tmp, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(prime, tmp, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_and_c<<<1, 1>>>(tmp, 2, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_cmp_c<<<1, 1>>>(tmp, 0, cmp, synchroTmp)), synchroTmp, 1);

        if (*cmp)
        {
            SYNCHRO((z_cu_add_c<<<1, 1>>>(prime, 2, synchroTmp)), synchroTmp, 1);
        }
        else
        {
            SYNCHRO((z_cu_add_c<<<1, 1>>>(prime, 1, synchroTmp)), synchroTmp, 1);
        }

        z_cu_t* n;
        cudaMalloc(&n, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(prime, n, synchroTmp)), synchroTmp, 1);

        __shared__ bool* synch;
        cudaMalloc(&synch, sizeof(bool) * BLOCK_SIZE * BLOCK_SIZE);

        SYNCHRO((next_prime<<<BLOCK_SIZE, BLOCK_SIZE>>>(n, prime, BLOCK_SIZE, p, primes_size, synch)), synch, BLOCK_SIZE * BLOCK_SIZE);

        cudaFree(synch);

        SYNCHRO((z_cu_free<<<1, 1>>>(n, synchroTmp)), synchroTmp, 1);
        cudaFree(n);

        SYNCHRO((z_cu_free<<<1, 1>>>(tmp, synchroTmp)), synchroTmp, 1);
        cudaFree(tmp);
    }
    
    cudaFree(cmp);
    cudaFree(fits);

    cudaFree(synchroTmp);

    synchro[idx] = true;
}

__global__ void previous_prime(z_cu_t const* n, z_cu_t* prime, size_t size,
                               unsigned int const* p, size_t primes_size, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ bool* synchroTmp;
    cudaMalloc(&synchroTmp, sizeof(bool));

    size_t start = blockIdx.x * blockDim.x + threadIdx.x + 1;

    z_cu_t* number;
    cudaMalloc(&number, sizeof(z_cu_t));

    int* cmp;
    cudaMalloc(&cmp, sizeof(int));
    SYNCHRO((z_cu_cmp_z<<<1, 1>>>(number, prime, cmp, synchroTmp)), synchroTmp, 1);

    while (*cmp > 0)
    {
        SYNCHRO((z_cu_free<<<1, 1>>>(number, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_set_from_ull<<<1, 1>>>(number, start, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_mul_c<<<1, 1>>>(number, 2, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_neg<<<1, 1>>>(number, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_add_z<<<1, 1>>>(number, n, synchroTmp)), synchroTmp, 1);
        
        int* result;
        cudaMalloc(&result, sizeof(int));
        SYNCHRO((z_cu_is_prime<<<1, 1>>>(number, 25, result, p, primes_size, synchroTmp)), synchroTmp, 1);

        if (*result)
        {
            SYNCHRO((z_cu_cmp_z<<<1, 1>>>(number, prime, cmp, synchroTmp)), synchroTmp, 1);

            if (*cmp > 0)
            {
                SYNCHRO((z_cu_free<<<1, 1>>>(prime, synchroTmp)), synchroTmp, 1);
                SYNCHRO((z_cu_set_from_z<<<1, 1>>>(prime, number, synchroTmp)), synchroTmp, 1);
            }
        }
        else
            start += size;

        cudaFree(result);

        SYNCHRO((z_cu_cmp_z<<<1, 1>>>(number, prime, cmp, synchroTmp)), synchroTmp, 1);
    }

    cudaFree(number);

    cudaFree(synchroTmp);

    synchro[idx] = true;
}


__global__ void z_cu_previous_prime(z_cu_t const* n, z_cu_t* prime,
                                    unsigned int const* p, size_t primes_size, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ bool* synchroTmp;
    cudaMalloc(&synchroTmp, sizeof(bool));

    int* cmp1;
    cudaMalloc(&cmp1, sizeof(int));

    SYNCHRO((z_cu_cmp_c<<<1, 1>>>(n, 2, cmp1, synchroTmp)), synchroTmp, 1);

    int* cmp2;
    cudaMalloc(&cmp2, sizeof(int));

    SYNCHRO((z_cu_cmp_c<<<1, 1>>>(n, 3, cmp2, synchroTmp)), synchroTmp, 1);

    bool* fits;
    cudaMalloc(&fits, sizeof(bool));

    SYNCHRO((z_cu_fits_ui<<<1, 1>>>(n, fits, synchroTmp)), synchroTmp, 1);

    bool found = false;

    if (n->is_nan)
    {
        found = true;
        SYNCHRO((z_cu_copy<<<1, 1>>>(n, prime, synchroTmp)), synchroTmp, 1);
    }
    else if (n->is_infinity || *cmp1 < 0)
    {
        found = true;
        SYNCHRO((z_cu_set_nan<<<1, 1>>>(prime, synchroTmp)), synchroTmp, 1);
    }
    else if (!*cmp1)
    {
        found = true;
        SYNCHRO((z_cu_from_c<<<1, 1>>>(prime, 2, synchroTmp)), synchroTmp, 1);
    }
    else if (!*cmp2)
    {
        found = true;
        SYNCHRO((z_cu_from_c<<<1, 1>>>(prime, 2, synchroTmp)), synchroTmp, 1);
    }
    else if (*fits)
    {
        pair* pr;
        cudaMalloc(&pr, sizeof(pair));

        unsigned int* ui;
        cudaMalloc(&ui, sizeof(unsigned int));

        SYNCHRO((z_cu_to_ui<<<1, 1>>>(n, ui, synchroTmp)), synchroTmp, 1);

        SYNCHRO((equal_range<<<1, 1>>>(p, 0, primes_size, *ui, pr, synchroTmp)), synchroTmp, 1);

        if (pr->first != primes_size && p[pr->first] != *ui)
            --pr->first;

        if (pr->first != pr->second && pr->second != primes_size)
        {
            found = true;
            
            if (p[pr->first] == *ui)
            {
                SYNCHRO((z_cu_from_ui<<<1, 1>>>(prime, p[pr->first - 1], synchroTmp)), synchroTmp, 1);
            }
            else
            {
                SYNCHRO((z_cu_from_ui<<<1, 1>>>(prime, p[pr->first], synchroTmp)), synchroTmp, 1);
            }
        }

        cudaFree(pr);
        cudaFree(ui);
    }

    if (!found)
    {
        SYNCHRO((z_cu_copy<<<1, 1>>>(n, prime, synchroTmp)), synchroTmp, 1);
        z_cu_t* tmp;
        cudaMalloc(&tmp, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(prime, tmp, synchroTmp)), synchroTmp, 1);
        SYNCHRO((z_cu_and_c<<<1, 1>>>(tmp, 2, synchroTmp)), synchroTmp, 1);

        SYNCHRO((z_cu_cmp_c<<<1, 1>>>(tmp, 0, cmp1, synchroTmp)), synchroTmp, 1);

        if (*cmp1)
        {
            SYNCHRO((z_cu_sub_c<<<1, 1>>>(prime, 2, synchroTmp)), synchroTmp, 1);
        }
        else
        {
            SYNCHRO((z_cu_sub_c<<<1, 1>>>(prime, 1, synchroTmp)), synchroTmp, 1);
        }

        z_cu_t* n;
        cudaMalloc(&n, sizeof(z_cu_t));
        SYNCHRO((z_cu_copy<<<1, 1>>>(prime, n, synchroTmp)), synchroTmp, 1);

        __shared__ bool* synch;
        cudaMalloc(&synch, sizeof(bool) * BLOCK_SIZE * BLOCK_SIZE);

        SYNCHRO((previous_prime<<<BLOCK_SIZE, BLOCK_SIZE>>>(n, prime, BLOCK_SIZE, p, primes_size, synch)), synch, BLOCK_SIZE * BLOCK_SIZE);

        cudaFree(synch);

        SYNCHRO((z_cu_free<<<1, 1>>>(n, synchroTmp)), synchroTmp, 1);
        cudaFree(n);

        SYNCHRO((z_cu_free<<<1, 1>>>(tmp, synchroTmp)), synchroTmp, 1);
        cudaFree(tmp);
    }

    cudaFree(cmp1);
    cudaFree(cmp2);
    cudaFree(fits);

    cudaFree(synchroTmp);

    synchro[idx] = true;
}
