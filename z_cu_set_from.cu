#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "z_cu_set_from.cuh"

#define Z_CU_SET_FROM(suffix, type)                                         \
__global__ void z_cu_set_from_##suffix(z_cu_t* z, type n, bool* synchro)    \
{                                                                           \
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;               \
                                                                            \
    assert(z);                                                              \
                                                                            \
    z->is_positive = (n >= 0);                                              \
                                                                            \
    size_t size = sizeof(type) / sizeof(z_cu_type);                         \
                                                                            \
    if (sizeof(type) % sizeof(z_cu_type))                                   \
        ++size;                                                             \
                                                                            \
    if (size > z->size)                                                     \
    {                                                                       \
        z->size = size;                                                     \
        cudaFree(z->bits);                                                  \
        z->bits = 0;                                                        \
        cudaMalloc(&z->bits, z->size * sizeof(z_cu_type));                  \
    }                                                                       \
                                                                            \
    z->is_nan = false;                                                      \
    z->is_infinity = false;                                                 \
                                                                            \
    if (n < 0)                                                              \
        n = -n;                                                             \
                                                                            \
    assert(z->bits);                                                        \
                                                                            \
    size_t const s = sizeof(type);                                          \
                                                                            \
    memset((char*)(z->bits) + s, 0, z->size * sizeof(z_cu_type) - s);       \
    memcpy(z->bits, &n, s);                                                 \
                                                                            \
    synchro[idx] = true;                                                    \
}

Z_CU_SET_FROM(c, char)
Z_CU_SET_FROM(i, int)
Z_CU_SET_FROM(l, long)
Z_CU_SET_FROM(ll, long long)
Z_CU_SET_FROM(s, short)

#define Z_CU_SET_FROM_UNSIGNED(suffix, type)                                \
__global__ void z_cu_set_from_##suffix(z_cu_t* z, type n, bool* synchro)    \
{                                                                           \
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;               \
                                                                            \
    assert(z);                                                              \
                                                                            \
    z->is_positive = true;                                                  \
                                                                            \
    size_t size = sizeof(type) / sizeof(z_cu_type);                         \
                                                                            \
    if (sizeof(type) % sizeof(z_cu_type))                                   \
        ++size;                                                             \
                                                                            \
    if (size > z->size)                                                     \
    {                                                                       \
        z->size = size;                                                     \
        cudaFree(z->bits);                                                  \
        z->bits = 0;                                                        \
        cudaMalloc(&z->bits, z->size * sizeof(z_cu_type));                  \
    }                                                                       \
                                                                            \
    z->is_nan = false;                                                      \
    z->is_infinity = false;                                                 \
                                                                            \
    assert(z->bits);                                                        \
                                                                            \
    size_t const s = sizeof(type);                                          \
                                                                            \
    memset((char*)(z->bits) + s, 0, z->size * sizeof(z_cu_type) - s);       \
    memcpy(z->bits, &n, s);                                                 \
                                                                            \
    synchro[idx] = true;                                                    \
}

Z_CU_SET_FROM_UNSIGNED(uc, unsigned char)
Z_CU_SET_FROM_UNSIGNED(ui, unsigned int)
Z_CU_SET_FROM_UNSIGNED(ul, unsigned long)
Z_CU_SET_FROM_UNSIGNED(ull, unsigned long long)
Z_CU_SET_FROM_UNSIGNED(us, unsigned short)

__global__ void z_cu_set_from_data(z_cu_t* z, void const* data, size_t size, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    assert(z);

    assert(data);

    z->is_positive = true;

    size_t s = size / sizeof(z_cu_type);

    if (size % sizeof(z_cu_type))
        ++s;

    if (s > z->size)
    {
        z->size = s;
        cudaFree(z->bits);
        z->bits = 0;
        cudaMalloc(&z->bits, z->size * sizeof(z_cu_type));
    }

    z->is_nan = false;
    z->is_infinity = false;

    if (size)
        assert(z->bits);

    memset((char*)(z->bits) + size, 0, z->size * sizeof(z_cu_type) - size);
    memcpy(z->bits, data, size);

    for (size_t i = 0; i < size / 2; ++i)
    {
        char tmp = *((char*)(z->bits) + size - 1 - i);
        *((char*)(z->bits) + size - 1 - i) = *((char*)(z->bits) + i);
        *((char*)(z->bits) + i) = tmp;
    }

    synchro[idx] = true;
}

__global__ void z_cu_set_from_z(z_cu_t* z, z_cu_t const* n, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    assert(z);
    assert(n);

    z->is_positive = n->is_positive;
    cudaFree(z->bits);
    z->bits = 0;
    cudaMalloc(&z->bits, n->size * sizeof(z_cu_type));
    assert(z->bits);
    memcpy(z->bits, n->bits, n->size * sizeof(z_cu_type));
    z->size = n->size;
    z->is_nan = n->is_nan;
    z->is_infinity = n->is_infinity;
    z->is_auto_adjust = n->is_auto_adjust;

    synchro[idx] = true;
}
