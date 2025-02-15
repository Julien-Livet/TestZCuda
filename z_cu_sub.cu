#include <assert.h>

#include "z_cu_sub.cuh"

#define Z_CU_SUB(suffix, type)                                                      \
__global__ void z_cu_sub_##suffix(z_cu_t* lhs, type rhs, bool* synchro)             \
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
    SYNCHRO((z_cu_sub_z<<<1, 1>>>(lhs, other, synchroTmp)), synchroTmp, 1);         \
                                                                                    \
    SYNCHRO((z_cu_free<<<1, 1>>>(other, synchroTmp)), synchroTmp, 1);               \
                                                                                    \
    cudaFree(synchroTmp);                                                           \
    cudaFree(other);                                                                \
                                                                                    \
    synchro[idx] = true;                                                            \
}

Z_CU_SUB(c, char)
Z_CU_SUB(i, int)
Z_CU_SUB(l, long)
Z_CU_SUB(ll, long long)
Z_CU_SUB(s, short)
Z_CU_SUB(uc, unsigned char)
Z_CU_SUB(ui, unsigned int)
Z_CU_SUB(ul, unsigned long)
Z_CU_SUB(ull, unsigned long long)
Z_CU_SUB(us, unsigned short)

__global__ void z_cu_sub_z(z_cu_t* lhs, z_cu_t const* rhs, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    assert(lhs);

    __shared__ bool* synchroTmp;
    cudaMalloc(&synchroTmp, sizeof(bool));

    z_cu_t* other;
    cudaMalloc(&other, sizeof(z_cu_t));
    SYNCHRO((z_cu_copy<<<1, 1>>>(rhs, other, synchroTmp)), synchroTmp, 1);

    other->is_positive = !other->is_positive;

    SYNCHRO((z_cu_add_z<<<1, 1>>>(lhs, other, synchroTmp)), synchroTmp, 1);
    
    SYNCHRO((z_cu_free<<<1, 1>>>(other, synchroTmp)), synchroTmp, 1);
    cudaFree(other);

    cudaFree(synchroTmp);

    synchro[idx] = true;
}
