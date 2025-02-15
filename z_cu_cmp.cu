#include "z_cu_cmp.cuh"

#define Z_CU_CMP(suffix, type)                                                          \
__global__ void z_cu_cmp_##suffix(z_cu_t const* lhs, type rhs, int* cmp, bool* synchro) \
{                                                                                       \
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;                           \
                                                                                        \
    z_cu_t* other = 0;                                                                  \
    cudaMalloc(&other, sizeof(z_cu_t));                                                 \
                                                                                        \
    __shared__ bool* synchroTmp;                                                        \
    cudaMalloc(&synchroTmp, sizeof(bool));                                              \
                                                                                        \
    SYNCHRO((z_cu_from_##suffix<<<1, 1>>>(other, rhs, synchroTmp)), synchroTmp, 1);     \
                                                                                        \
    SYNCHRO((z_cu_cmp_z<<<1, 1>>>(lhs, other, cmp, synchroTmp)), synchroTmp, 1);        \
                                                                                        \
    SYNCHRO((z_cu_free<<<1, 1>>>(other, synchroTmp)), synchroTmp, 1);                   \
                                                                                        \
    cudaFree(synchroTmp);                                                               \
    cudaFree(other);                                                                    \
                                                                                        \
    synchro[idx] = true;                                                                \
}

Z_CU_CMP(c, char)
Z_CU_CMP(i, int)
Z_CU_CMP(l, long)
Z_CU_CMP(ll, long long)
Z_CU_CMP(s, short)
Z_CU_CMP(uc, unsigned char)
Z_CU_CMP(ui, unsigned int)
Z_CU_CMP(ul, unsigned long)
Z_CU_CMP(ull, unsigned long long)
Z_CU_CMP(us, unsigned short)

__global__ void z_cu_cmp_z(z_cu_t const* lhs, z_cu_t const* rhs, int* cmp, bool* synchro)
{
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (lhs->is_nan && rhs->is_nan)
        *cmp = 0;
    else if (lhs->is_nan || rhs->is_nan)
        *cmp = 2;
    else if (rhs->is_infinity)
    {
        if (lhs->is_infinity)
        {
            if (rhs->is_positive == lhs->is_positive)
                *cmp = 0;
            else if (rhs->is_positive && !lhs->is_positive)
                *cmp = -1;
            else
                *cmp = 1;
        }
        else if (rhs->is_positive)
            *cmp = -1;
        else
            *cmp = 1;
    }
    else if (lhs->is_infinity)
    {
        if (rhs->is_infinity)
        {
            if (rhs->is_positive == lhs->is_positive)
                *cmp = 0;
            else if (lhs->is_positive && !rhs->is_positive)
                *cmp = 1;
            else
                *cmp = -1;
        }
        else if (lhs->is_positive)
            *cmp = 1;
        else
            *cmp = -1;
    }
    else
    {
        *cmp = 0;

        bool found = false;

        if (lhs->size != rhs->size)
        {
            if (lhs->size > rhs->size)
            {
                for (size_t i = 0; i < lhs->size - rhs->size; ++i)
                {
                    if (lhs->bits[i])
                    {
                        found = true;
                        *cmp = lhs->is_positive;
                        break;
                    }
                }
            }
            else
            {
                for (size_t i = 0; i < rhs->size - lhs->size; ++i)
                {
                    if (rhs->bits[i])
                    {
                        found = true;
                        *cmp = -rhs->is_positive;
                        break;
                    }
                }
            }
        }

        if (!found)
        {
            size_t i1 = lhs->size - MIN(lhs->size, rhs->size);
            size_t i2 = rhs->size - MIN(lhs->size, rhs->size);

            for (size_t i = 0; i < MIN(lhs->size, rhs->size); ++i)
            {
                if (lhs->is_positive && rhs->is_positive)
                {
                    if (lhs->bits[i1] > rhs->bits[i2])
                    {
                        *cmp = 1;
                        break;
                    }
                    else if (lhs->bits[i1] < rhs->bits[i2])
                    {
                        *cmp = -1;
                        break;
                    }
                }
                else if (!lhs->is_positive && !rhs->is_positive)
                {
                    if (lhs->bits[i1] > rhs->bits[i2])
                    {
                        *cmp = -1;
                        break;
                    }
                    else if (lhs->bits[i1] < rhs->bits[i2])
                    {
                        *cmp = 1;
                        break;
                    }
                }
                else if (lhs->is_positive && !rhs->is_positive)
                {
                    if (lhs->bits[i1] != rhs->bits[i2])
                    {
                        *cmp = 1;
                        break;
                    }
                }
                else if (!lhs->is_positive && rhs->is_positive)
                {
                    if (lhs->bits[i1] != rhs->bits[i2])
                    {
                        *cmp = -1;
                        break;
                    }
                }

                ++i1;
                ++i2;
            }
        }
    }

    synchro[idx] = true;
}
