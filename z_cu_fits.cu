#include "z_cu_fits.cuh"

#define Z_CU_FITS(suffix, type)                                                     \
__global__ void z_cu_fits_##suffix(z_cu_t const* z, bool* fits, bool* synchro)      \
{                                                                                   \
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;                       \
                                                                                    \
    type* n;                                                                        \
    cudaMalloc(&n, sizeof(type));                                                   \
                                                                                    \
    __shared__ bool* synchroTmp;                                                    \
    cudaMalloc(&synchroTmp, sizeof(bool));                                          \
                                                                                    \
    SYNCHRO((z_cu_to_##suffix<<<1, 1>>>(z, n, synchroTmp)), synchroTmp, 1);         \
                                                                                    \
    int* cmp;                                                                       \
    cudaMalloc(&cmp, sizeof(int));                                                  \
                                                                                    \
    SYNCHRO((z_cu_cmp_##suffix<<<1, 1>>>(z, *n, cmp, synchroTmp)), synchroTmp, 1);  \
                                                                                    \
    *fits = !*cmp;                                                                  \
                                                                                    \
    cudaFree(n);                                                                    \
    cudaFree(cmp);                                                                  \
    cudaFree(synchroTmp);                                                           \
                                                                                    \
    synchro[idx] = true;                                                            \
}

Z_CU_FITS(c, char)
Z_CU_FITS(i, int)
Z_CU_FITS(l, long)
Z_CU_FITS(ll, long long)
Z_CU_FITS(s, short)
Z_CU_FITS(uc, unsigned char)
Z_CU_FITS(ui, unsigned int)
Z_CU_FITS(ul, unsigned long)
Z_CU_FITS(ull, unsigned long long)
Z_CU_FITS(us, unsigned short)
