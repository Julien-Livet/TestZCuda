#ifndef Z_CU_T_CUH
#define Z_CU_T_CUH

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>//TODO: to remove
#include <stdint.h>

#include <cuda.h>
#include <cuda_runtime.h>

typedef uintmax_t z_cu_type;
typedef uintmax_t longest_type;

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

#define BLOCK_SIZE 1024

struct z_cu_t_struct
{
    bool is_positive;
    z_cu_type* bits;
    size_t size;
    bool is_nan;
    bool is_infinity;
    bool is_auto_adjust;
};

typedef struct z_cu_t_struct z_cu_t;

__global__ void z_cu_abs(z_cu_t* z, bool* synchro);
__global__ void z_cu_adjust(z_cu_t* z, bool* synchro);
__global__ void z_cu_copy(z_cu_t const* from, z_cu_t* to, bool* synchro);
//z_cu_t z_cu_factorial(z_cu_t n);
__global__ void z_cu_free(z_cu_t* z, bool* synchro);
void z_cu_free2(z_cu_t* z);
__global__ void z_cu_gcd(z_cu_t const* a, z_cu_t const* b, z_cu_t* gcd, bool* synchro);
__global__ void z_cu_gcd_extended(z_cu_t const* a, z_cu_t const* b, z_cu_t* u, z_cu_t* v, z_cu_t* gcd, bool* synchro);
//z_cu_t z_cu_infinity();
__global__ void z_cu_invert(z_cu_t* z, bool* synchro);
__global__ void z_cu_is_even(z_cu_t const* z, bool* even, bool* synchro);
//bool z_cu_is_negative(z_cu_t z);
//bool z_cu_is_null(z_cu_t z);
__global__ void z_cu_is_odd(z_cu_t const* z, bool* odd, bool* synchro);
//bool z_cu_is_positive(z_cu_t z);
//z_cu_t z_cu_max(z_cu_t a, z_cu_t b);
//z_cu_t z_cu_min(z_cu_t a, z_cu_t b);
//z_cu_t z_cu_nan();
__global__ void z_cu_neg(z_cu_t* z, bool* synchro);
__global__ void z_cu_number(z_cu_t const* z, z_cu_t* number, bool* synchro);
///__global__ void z_cu_printf(z_cu_t z, size_t base);
__global__ void z_cu_printf_bits(z_cu_t const* z, bool* synchro);
__global__ void z_cu_printf_bytes(z_cu_t const* z, bool* synchro);
__global__ void z_cu_set_auto_adjust(z_cu_t* z, bool is_auto_adjust, bool* synchro);
__global__ void z_cu_set_infinity(z_cu_t* z, bool* synchro);
__global__ void z_cu_set_nan(z_cu_t* z, bool* synchro);
__global__ void z_cu_set_negative(z_cu_t* z);
__global__ void z_cu_set_positive(z_cu_t* z);
__global__ void z_cu_set_precision(z_cu_t* z, size_t precision, bool* synchro);
__global__ void z_cu_set_random(z_cu_t* z, bool* synchro);
__global__ void z_cu_sign(z_cu_t const* z, int* sign, bool* synchro);
__global__ void z_cu_sqrt(z_cu_t const* n, z_cu_t* sqrt, bool* synchro);

#define SYNCHRO(instruction, synchro, count)    \
{                                               \
    for (size_t i = 0; i < count; ++i)          \
        synchro[i] = false;                     \
}                                               \
                                                \
instruction;                                    \
                                                \
{                                               \
    size_t c = 0;                               \
                                                \
    while (c != count)                          \
    {                                           \
        __syncthreads();                        \
                                                \
        c = 0;                                  \
                                                \
        for (size_t i = 0; i < count; ++i)      \
        {                                       \
            if (synchro[i])                     \
                ++c;                            \
        }                                       \
    }                                           \
}

#define SYNCHRO_(instruction, synchro, count)    \
{                                               \
    for (size_t i = 0; i < count; ++i)          \
        synchro[i] = false;                     \
}                                               \
printf("lol0\n");                                                \
instruction;                                    \
printf("lol1\n");\
{                                               \
    size_t c = 0;                               \
    printf("lol2\n");                                                \
    while (c != count)                          \
    {                                           \
        __syncthreads();                        \
                                                \
        c = 0;                                  \
                                                \
        for (size_t i = 0; i < count; ++i)      \
        {                                       \
            if (synchro[i])                     \
                ++c;                            \
        }                                       \
    }                                           \
    printf("lol3\n");\
}

#include "z_cu_add.cuh"
#include "z_cu_and.cuh"
#include "z_cu_cmp.cuh"
#include "z_cu_div_q.cuh"
#include "z_cu_div_qr.cuh"
#include "z_cu_div_r.cuh"
#include "z_cu_fits.cuh"
#include "z_cu_from.cuh"
#include "z_cu_lshift.cuh"
#include "z_cu_mul.cuh"
#include "z_cu_or.cuh"
#include "z_cu_pow.cuh"
#include "z_cu_powm.cuh"
#include "z_cu_prime.cuh"
#include "z_cu_rshift.cuh"
#include "z_cu_set_from.cuh"
#include "z_cu_sub.cuh"
#include "z_cu_to.cuh"
#include "z_cu_xor.cuh"

#endif // Z_CU_T_H
