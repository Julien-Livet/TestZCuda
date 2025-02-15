#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "z_cu_xor.cuh"

#define Z_CU_XOR(suffix, type)                                                      \
__global__ void z_cu_xor_##suffix(z_cu_t* lhs, type rhs, bool* synchro)             \
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
    SYNCHRO((z_cu_xor_z<<<1, 1>>>(lhs, other, synchroTmp)), synchroTmp, 1);         \
                                                                                    \
    SYNCHRO((z_cu_free<<<1, 1>>>(other, synchroTmp)), synchroTmp, 1);               \
                                                                                    \
    cudaFree(synchroTmp);                                                           \
    cudaFree(other);                                                                \
                                                                                    \
    synchro[idx] = true;                                                            \
}

Z_CU_XOR(c, char)
Z_CU_XOR(i, int)
Z_CU_XOR(l, long)
Z_CU_XOR(ll, long long)
Z_CU_XOR(s, short)
Z_CU_XOR(uc, unsigned char)
Z_CU_XOR(ui, unsigned int)
Z_CU_XOR(ul, unsigned long)
Z_CU_XOR(ull, unsigned long long)
Z_CU_XOR(us, unsigned short)

__global__ void z_cu_xor_z(z_cu_t* lhs, z_cu_t const* rhs, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    assert(lhs);
    assert(rhs);

    if (lhs->size < rhs->size)
    {
        z_cu_type* bits = lhs->bits;
        lhs->bits = 0;
        cudaMalloc(&lhs->bits, rhs->size * sizeof(z_cu_type));
        assert(lhs->bits);
        memset(lhs->bits, 0, (rhs->size - lhs->size) * sizeof(z_cu_type));
        memcpy((char*)(lhs->bits) + (rhs->size - lhs->size) * sizeof(z_cu_type), bits, lhs->size * sizeof(z_cu_type));
        lhs->size = rhs->size;
        cudaFree(bits);
    }
    else if (lhs->size > rhs->size)
    {
        for (size_t i = 0; i < lhs->size - rhs->size; ++i)
            lhs->bits[i] ^= 0;
    }

    for (size_t i = 0; i < MIN(lhs->size, rhs->size); ++i)
        lhs->bits[lhs->size - 1 - i] ^= rhs->bits[rhs->size - 1 - i];



    if (lhs->is_auto_adjust)
    {
        __shared__ bool* synchroTmp;
        cudaMalloc(&synchroTmp, sizeof(bool));

        SYNCHRO((z_cu_adjust<<<1, 1>>>(lhs, synchroTmp)), synchroTmp, 1);

        cudaFree(synchroTmp);
    }

    synchro[idx] = true;
}
