#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "z_cu_to.cuh"

#define Z_CU_TO(suffix, type)                                               \
__global__ void z_cu_to_##suffix(z_cu_t const* z, type* n, bool* synchro)   \
{                                                                           \
    size_t const idx = blockIdx.x * blockDim.x + threadIdx.x;               \
                                                                            \
    memcpy(n, z->bits, MIN(sizeof(type), z->size * sizeof(z_cu_type)));     \
                                                                            \
    if (!z->is_positive)                                                    \
        *n = -*n;                                                           \
                                                                            \
    synchro[idx] = true;                                                    \
}

Z_CU_TO(c, char)
Z_CU_TO(i, int)
Z_CU_TO(l, long)
Z_CU_TO(ll, long long)
Z_CU_TO(s, short)
Z_CU_TO(uc, unsigned char)
Z_CU_TO(ui, unsigned int)
Z_CU_TO(ul, unsigned long)
Z_CU_TO(ull, unsigned long long)
Z_CU_TO(us, unsigned short)

/*
char* filled_str(unsigned long long n, size_t width, char fillChar)
{
    char* s = (char*)malloc(100 + 1);
    sprintf(s, "%llu", n);

    if (strlen(s) < width)
    {
        char* sTmp = (char*)malloc(width + 1);
        memset(sTmp, fillChar, width);
        strcpy(sTmp + width - strlen(s), s);
        free(s);
        s = sTmp;
    }

    return s;
}

__global__ void z_cu_to_str(z_cu_t z, size_t base, char** s)
{
    if (!base)
        base = 10;

    assert(2 <= base && base <= 62);

    char* s = NULL;

    if (z_cu_is_nan(z))
    {
        if (!z_cu_is_positive(z))
        {
            s = (char*)malloc(5);
            strcpy(s, "-nan");
        }
        else
        {
            s = (char*)malloc(4);
            strcpy(s, "nan");
        }

        return s;
    }
    else if (z_cu_is_infinity(z))
    {
        if (!z_cu_is_positive(z))
        {
            s = (char*)malloc(5);
            strcpy(s, "-inf");
        }
        else
        {
            s = (char*)malloc(4);
            strcpy(s, "inf");
        }

        return s;
    }

    unsigned long long size = 2 + z.size * sizeof(z_cu_type) * 8 + (z.is_positive ? 0 : 1) + 1;
    s = (char*)malloc(size);
    memset(s, 0, size);

    if (base == 2)
    {
        for (size_t j = 0; j < z.size; ++j)
        {
            z_cu_type b = z.bits[z.size - 1 - j];

            for (size_t i = 0; i < sizeof(z_cu_type) * 8; ++i)
            {
                char* sTmp = (char*)malloc(strlen(s) + 1);
                memcpy(sTmp, s, strlen(s) + 1);

                sprintf(s, "%d%s", (b & 1 ? 1 : 0), sTmp);
                b >>= 1;

                free(sTmp);
            }
        }

        char* sTmp = (char*)malloc(strlen(s) + 1);
        memcpy(sTmp, s, strlen(s) + 1);

        sprintf(s, "0b%s", sTmp);

        free(sTmp);
    }
    else if (base == 8)
    {
        z_cu_t number = z_cu_abs(z);

        if (!z_cu_cmp_c(number, 0))
            strcpy(s, "0");

        while (z_cu_cmp_c(number, 0))
        {
            z_cu_t tmp = z_cu_div_r_c(number, 8);

            char* sTmp = (char*)malloc(strlen(s) + 1);
            memcpy(sTmp, s, strlen(s) + 1);

            sprintf(s, "%d%s", z_cu_to_s(tmp), sTmp);

            free(sTmp);

            z_cu_free(&tmp);

            z_cu_t q = z_cu_div_q_c(number, 8);
            z_cu_free(&number);
            number = q;
        }

        char* sTmp = (char*)malloc(strlen(s) + 1);
        memcpy(sTmp, s, strlen(s) + 1);

        sprintf(s, "0o%s", sTmp);

        free(sTmp);

        z_cu_free(&number);
    }
    else if (base == 10)
    {
        z_cu_t number = z_cu_abs(z);

        if (z.size == 1)
            sprintf(s, "%llu", z.bits[0]);
        else
        {
            if (!z_cu_cmp_c(number, 0))
                strcpy(s, "0");

            z_cu_type zero = 0;
            z_cu_type const n = log10(~zero);
            z_cu_type const b = pow(10, n);

            while (z_cu_cmp_c(number, 0))
            {
                z_cu_t tmp = z_cu_div_r_ull(number, b);

                char* s1Tmp = filled_str(z_cu_to_ull(tmp), n, '0');
                char* sTmp = (char*)malloc(strlen(s) + 1);
                memcpy(sTmp, s, strlen(s) + 1);
                sprintf(s, "%s%s", s1Tmp, sTmp);

                free(s1Tmp);
                free(sTmp);

                z_cu_free(&tmp);

                z_cu_t q = z_cu_div_q_ull(number, b);
                z_cu_free(&number);
                number = q;
            }

            size_t i = 0;

            while (s[i] == '0' && i < strlen(s))
                ++i;

            if (i == strlen(s))
                i = strlen(s) - 1;

            char* sTmp = (char*)malloc(strlen(s) + 1);
            memcpy(sTmp, s + i, strlen(s) + 1 - i);
            free(s);
            s = sTmp;
        }

        z_cu_free(&number);
    }
    else if (2 < base && base < 16)
    {
        z_cu_t number = z_cu_abs(z);

        if (z.size == 1)
            sprintf(s, "%llu", z.bits[0]);
        else
        {
            if (!z_cu_cmp_c(number, 0))
                strcpy(s, "0");

            while (z_cu_cmp_c(number, 0))
            {
                z_cu_t tmp = z_cu_div_r_ull(number, base);

                char* sTmp = (char*)malloc(strlen(s) + 1);
                memcpy(sTmp, s, strlen(s) + 1);

                sprintf(s, "%d%s", z_cu_to_s(tmp), sTmp);

                free(sTmp);

                z_cu_free(&tmp);

                z_cu_t q = z_cu_div_q_ull(number, base);
                z_cu_free(&number);
                number = q;
            }
        }

        z_cu_free(&number);
    }
    else if (base == 16)
    {
        z_cu_t number = z_cu_abs(z);

        if (!z_cu_cmp_c(number, 0))
            strcpy(s, "0");

        while (z_cu_cmp_c(number, 0))
        {
            z_cu_t tmp = z_cu_div_r_c(number, 16);

            if (z_cu_cmp_c(tmp, 10) < 0)
            {
                char* sTmp = (char*)malloc(strlen(s) + 1);
                memcpy(sTmp, s, strlen(s) + 1);

                sprintf(s, "%d%s", z_cu_to_s(tmp), sTmp);

                free(sTmp);
            }
            else
            {
                char* sTmp = (char*)malloc(strlen(s) + 1);
                memcpy(sTmp, s, strlen(s) + 1);

                sprintf(s, "%c%s", 'a' + z_cu_to_s(tmp) - 10, sTmp);

                free(sTmp);
            }

            z_cu_t q = z_cu_div_q_c(number, 16);
            z_cu_free(&number);
            number = q;

            z_cu_free(&tmp);
        }

        char* sTmp = (char*)malloc(strlen(s) + 1);
        memcpy(sTmp, s, strlen(s) + 1);

        sprintf(s, "0x%s", sTmp);

        free(sTmp);

        z_cu_free(&number);
    }
    else if (base <= 62)
    {
        z_cu_t number = z_cu_abs(z);

        if (!z_cu_cmp_c(number, 0))
            strcpy(s, "0");

        while (z_cu_cmp_c(number, 0))
        {
            z_cu_t tmp = z_cu_div_r_ull(number, base);

            if (z_cu_cmp_c(tmp, 10) < 0)
            {
                char* sTmp = (char*)malloc(strlen(s) + 1);
                memcpy(sTmp, s, strlen(s) + 1);

                sprintf(s, "%d%s", z_cu_to_s(tmp), sTmp);

                free(sTmp);
            }
            else if (z_cu_cmp_c(tmp, -36) < 0)
            {
                char* sTmp = (char*)malloc(strlen(s) + 1);
                memcpy(sTmp, s, strlen(s) + 1);

                sprintf(s, "%c%s", 'a' + z_cu_to_s(tmp) - 10, sTmp);

                free(sTmp);
            }
            else
            {
                char* sTmp = (char*)malloc(strlen(s) + 1);
                memcpy(sTmp, s, strlen(s) + 1);

                sprintf(s, "%c%s", 'A' + z_cu_to_s(tmp) - 36, sTmp);

                free(sTmp);
            }

            z_cu_t q = z_cu_div_q_ull(number, base);
            z_cu_free(&number);
            number = q;

            z_cu_free(&tmp);
        }

        z_cu_free(&number);
    }

    if (!z.size && (!strlen(s) && s[strlen(s) - 1] != '0'))
    {
        char* sTmp = (char*)malloc(strlen(s) + 1);
        memcpy(sTmp, s, strlen(s) + 1);

        sprintf(s, "%s0", sTmp);

        free(sTmp);
    }

    if (!z_cu_is_positive(z))
    {
        char* sTmp = (char*)malloc(strlen(s) + 1);
        memcpy(sTmp, s, strlen(s) + 1);

        sprintf(s, "-%s", sTmp);

        free(sTmp);
    }

    return s;
}
*/
