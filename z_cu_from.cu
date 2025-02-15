#include <assert.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#include "z_cu_from.cuh"

#define Z_CU_FROM(suffix, type)                                             \
__global__ void z_cu_from_##suffix(z_cu_t* z, type n, bool* synchro)        \
{                                                                           \
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;               \
                                                                            \
    z->is_positive = (n >= 0);                                              \
    z->size = sizeof(type) / sizeof(z_cu_type);                             \
                                                                            \
    if (sizeof(type) % sizeof(z_cu_type))                                   \
        ++z->size;                                                          \
                                                                            \
    z->bits = 0;                                                            \
    cudaMalloc(&z->bits, z->size * sizeof(z_cu_type));                      \
                                                                            \
    z->is_nan = false;                                                      \
    z->is_infinity = false;                                                 \
    z->is_auto_adjust = true;                                               \
                                                                            \
    if (n < 0)                                                              \
        n = -n;                                                             \
                                                                            \
    assert(z->bits);                                                        \
                                                                            \
    size_t const size = sizeof(type);                                       \
                                                                            \
    memset((char*)(z->bits) + size, 0, z->size * sizeof(z_cu_type) - size); \
    memcpy(z->bits, &n, size);                                              \
                                                                            \
    synchro[idx] = true;                                                    \
}

Z_CU_FROM(c, char)
Z_CU_FROM(i, int)
Z_CU_FROM(l, long)
Z_CU_FROM(ll, long long)
Z_CU_FROM(s, short)

#define Z_CU_FROM_UNSIGNED(suffix, type)                                    \
__global__ void z_cu_from_##suffix(z_cu_t* z, type n, bool* synchro)        \
{                                                                           \
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;               \
                                                                            \
    z->is_positive = true;                                                  \
    z->size = sizeof(type) / sizeof(z_cu_type);                             \
                                                                            \
    if (sizeof(type) % sizeof(z_cu_type))                                   \
        ++z->size;                                                          \
                                                                            \
    z->bits = 0;                                                            \
    cudaMalloc(&z->bits, z->size * sizeof(z_cu_type));                      \
                                                                            \
    z->is_nan = false;                                                      \
    z->is_infinity = false;                                                 \
    z->is_auto_adjust = true;                                               \
                                                                            \
    assert(z->bits);                                                        \
                                                                            \
    size_t const size = sizeof(type);                                       \
                                                                            \
    memset((char*)(z->bits) + size, 0, z->size * sizeof(z_cu_type) - size); \
    memcpy(z->bits, &n, size);                                              \
                                                                            \
    synchro[idx] = true;                                                    \
}

Z_CU_FROM_UNSIGNED(uc, unsigned char)
Z_CU_FROM_UNSIGNED(ui, unsigned int)
Z_CU_FROM_UNSIGNED(ul, unsigned long)
Z_CU_FROM_UNSIGNED(ull, unsigned long long)
Z_CU_FROM_UNSIGNED(us, unsigned short)

__global__ void z_cu_from_data(z_cu_t* z, void const* data, size_t size, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    assert(data);

    z->is_positive = true;
    z->size = size / sizeof(z_cu_type);

    if (size % sizeof(z_cu_type))
        ++z->size;

    z->bits = 0;
    cudaMalloc(&z->bits, z->size * sizeof(z_cu_type));
    z->is_nan = false;
    z->is_infinity = false;
    z->is_auto_adjust = true;

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

void z_cu_from_z(z_t const* from, z_cu_t* to)
{
    cudaMemcpy(&to->is_auto_adjust, &from->is_auto_adjust, sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(&to->is_infinity, &from->is_infinity, sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(&to->is_nan, &from->is_nan, sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(&to->is_positive, &from->is_positive, sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(&to->size, &from->size, sizeof(size_t), cudaMemcpyHostToDevice);
    z_type* tmp;
    cudaMalloc(&tmp, sizeof(z_type) * from->size);
    cudaMemcpy(&to->bits, &tmp, sizeof(z_type*), cudaMemcpyHostToDevice);
    cudaMemcpy(tmp, from->bits, sizeof(z_type) * from->size, cudaMemcpyHostToDevice);
}

void z_from_z_cu(z_cu_t const* from, z_t* to)
{
    cudaMemcpy(&to->is_auto_adjust, &from->is_auto_adjust, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(&to->is_infinity, &from->is_infinity, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(&to->is_nan, &from->is_nan, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(&to->is_positive, &from->is_positive, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(&to->size, &from->size, sizeof(size_t), cudaMemcpyDeviceToHost);
    to->bits = (z_type*)malloc(sizeof(z_type) * to->size);
    z_type* tmp;
    cudaMemcpy(&tmp, &from->bits, sizeof(z_type*), cudaMemcpyDeviceToHost);
    cudaMemcpy(to->bits, tmp, sizeof(z_type) * to->size, cudaMemcpyDeviceToHost);
    cudaFree(tmp);
}
