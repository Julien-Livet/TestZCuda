#include <stdlib.h>

#include "z_cu_div_r.cuh"

#define Z_CU_DIV_R(suffix, type)                                                            \
__global__ void z_cu_div_r_##suffix(z_cu_t const* lhs, type rhs, z_cu_t* q, bool* synchro)  \
{                                                                                           \
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;                               \
                                                                                            \
    z_cu_t* other = 0;                                                                      \
    cudaMalloc(&other, sizeof(z_cu_t));                                                     \
                                                                                            \
    __shared__ bool* synchroTmp;                                                            \
    cudaMalloc(&synchroTmp, sizeof(bool));                                                  \
                                                                                            \
    SYNCHRO((z_cu_from_##suffix<<<1, 1>>>(other, rhs, synchroTmp)), synchroTmp, 1);         \
                                                                                            \
    SYNCHRO((z_cu_div_r_z<<<1, 1>>>(lhs, other, q, synchroTmp)), synchroTmp, 1);            \
                                                                                            \
    SYNCHRO((z_cu_free<<<1, 1>>>(other, synchroTmp)), synchroTmp, 1);                       \
                                                                                            \
    cudaFree(synchroTmp);                                                                   \
    cudaFree(other);                                                                        \
                                                                                            \
    synchro[idx] = true;                                                                    \
}

Z_CU_DIV_R(c, char)
Z_CU_DIV_R(i, int)
Z_CU_DIV_R(l, long)
Z_CU_DIV_R(ll, long long)
Z_CU_DIV_R(s, short)
Z_CU_DIV_R(uc, unsigned char)
Z_CU_DIV_R(ui, unsigned int)
Z_CU_DIV_R(ul, unsigned long)
Z_CU_DIV_R(ull, unsigned long long)
Z_CU_DIV_R(us, unsigned short)

__global__ void z_cu_div_r_z(z_cu_t const* lhs, z_cu_t const* rhs, z_cu_t* r, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ bool* synchroTmp;
    cudaMalloc(&synchroTmp, sizeof(bool));

    z_cu_t *q = 0;
    cudaMalloc(&q, sizeof(z_cu_t));
    SYNCHRO((z_cu_from_c<<<1, 1>>>(q, 0, synchroTmp)), synchroTmp, 1);
    SYNCHRO((z_cu_from_c<<<1, 1>>>(r, 0, synchroTmp)), synchroTmp, 1);

    SYNCHRO((z_cu_div_qr_z<<<1, 1>>>(lhs, rhs, q, r, synchroTmp)), synchroTmp, 1);

    SYNCHRO((z_cu_free<<<1, 1>>>(q, synchroTmp)), synchroTmp, 1);
    cudaFree(q);

    cudaFree(synchroTmp);

    synchro[idx] = true;
}
