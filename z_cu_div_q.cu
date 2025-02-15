#include "z_cu_div_q.cuh"

#define Z_CU_DIV_Q(suffix, type)                                                            \
__global__ void z_cu_div_q_##suffix(z_cu_t const* lhs, type rhs, z_cu_t* q, bool* synchro)  \
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
    SYNCHRO((z_cu_div_q_z<<<1, 1>>>(lhs, other, q, synchroTmp)), synchroTmp, 1);            \
                                                                                            \
    SYNCHRO((z_cu_free<<<1, 1>>>(other, synchroTmp)), synchroTmp, 1);                       \
                                                                                            \
    cudaFree(synchroTmp);                                                                   \
    cudaFree(other);                                                                        \
                                                                                            \
    synchro[idx] = true;                                                                    \
}

Z_CU_DIV_Q(c, char)
Z_CU_DIV_Q(i, int)
Z_CU_DIV_Q(l, long)
Z_CU_DIV_Q(ll, long long)
Z_CU_DIV_Q(s, short)
Z_CU_DIV_Q(uc, unsigned char)
Z_CU_DIV_Q(ui, unsigned int)
Z_CU_DIV_Q(ul, unsigned long)
Z_CU_DIV_Q(ull, unsigned long long)
Z_CU_DIV_Q(us, unsigned short)

__global__ void z_cu_div_q_z(z_cu_t const* lhs, z_cu_t const* rhs, z_cu_t* q, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ bool* synchroTmp;
    cudaMalloc(&synchroTmp, sizeof(bool));

    SYNCHRO((z_cu_from_c<<<1, 1>>>(q, 0, synchroTmp)), synchroTmp, 1);

    z_cu_t *r = 0;
    cudaMalloc(&r, sizeof(z_cu_t));
    SYNCHRO((z_cu_from_c<<<1, 1>>>(r, 0, synchroTmp)), synchroTmp, 1);

    SYNCHRO((z_cu_div_qr_z<<<1, 1>>>(lhs, rhs, q, r, synchroTmp)), synchroTmp, 1);

    int* cmp1;
    cudaMalloc(&cmp1, sizeof(int));
    int* cmp2;
    cudaMalloc(&cmp2, sizeof(int));

    SYNCHRO((z_cu_cmp_c<<<1, 1>>>(lhs, 0, cmp1, synchroTmp)), synchroTmp, 1);
    SYNCHRO((z_cu_cmp_c<<<1, 1>>>(rhs, 0, cmp2, synchroTmp)), synchroTmp, 1);

    if (*cmp1 < 0 && *cmp2 < 0)
    {
        SYNCHRO((z_cu_cmp_c<<<1, 1>>>(r, 0, cmp1, synchroTmp)), synchroTmp, 1);

        if (*cmp1)
        {
            SYNCHRO((z_cu_sub_c<<<1, 1>>>(q, 1, synchroTmp)), synchroTmp, 1);
        }
    }
    else if (*cmp1 > 0 && *cmp2 < 0)
    {
        SYNCHRO((z_cu_cmp_c<<<1, 1>>>(r, 0, cmp1, synchroTmp)), synchroTmp, 1);

        if (*cmp1)
        {
            SYNCHRO((z_cu_add_c<<<1, 1>>>(q, 1, synchroTmp)), synchroTmp, 1);
        }
    }
    else if (*cmp1 < 0 && *cmp2 > 0)
    {
        SYNCHRO((z_cu_cmp_c<<<1, 1>>>(r, 0, cmp1, synchroTmp)), synchroTmp, 1);

        if (*cmp1)
        {
            SYNCHRO((z_cu_add_c<<<1, 1>>>(q, 1, synchroTmp)), synchroTmp, 1);
        }
    }

    cudaFree(cmp1);
    cudaFree(cmp2);

    SYNCHRO((z_cu_free<<<1, 1>>>(r, synchroTmp)), synchroTmp, 1);
    cudaFree(r);

    cudaFree(synchroTmp);

    synchro[idx] = true;
}
