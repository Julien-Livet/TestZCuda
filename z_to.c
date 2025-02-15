#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "z_to.h"

#define Z_TO(suffix, type)                                          \
type z_to_##suffix(z_t z)                                           \
{                                                                   \
    type n = 0;                                                     \
                                                                    \
    memcpy(&n, z.bits, MIN(sizeof(type), z.size * sizeof(z_type))); \
                                                                    \
    if (!z_is_positive(z))                                          \
        n = -n;                                                     \
                                                                    \
    return n;                                                       \
}

Z_TO(c, char)
Z_TO(i, int)
Z_TO(l, long)
Z_TO(ll, long long)
Z_TO(s, short)
Z_TO(uc, unsigned char)
Z_TO(ui, unsigned int)
Z_TO(ul, unsigned long)
Z_TO(ull, unsigned long long)
Z_TO(us, unsigned short)

char* filled_str(unsigned long long n, size_t width, char fillChar)
{
    char* s = malloc(100 + 1);
    sprintf(s, "%llu", n);

    if (strlen(s) < width)
    {
        char* sTmp = malloc(width + 1);
        memset(sTmp, fillChar, width);
        strcpy(sTmp + width - strlen(s), s);
        free(s);
        s = sTmp;
    }

    return s;
}

char* z_to_str(z_t z, size_t base)
{
    if (!base)
        base = 10;

    assert(2 <= base && base <= 62);

    char* s = NULL;

    if (z_is_nan(z))
    {
        if (!z_is_positive(z))
        {
            s = malloc(5);
            strcpy(s, "-nan");
        }
        else
        {
            s = malloc(4);
            strcpy(s, "nan");
        }

        return s;
    }
    else if (z_is_infinity(z))
    {
        if (!z_is_positive(z))
        {
            s = malloc(5);
            strcpy(s, "-inf");
        }
        else
        {
            s = malloc(4);
            strcpy(s, "inf");
        }

        return s;
    }

    unsigned long long size = 2 + z.size * sizeof(z_type) * 8 + (z.is_positive ? 0 : 1) + 1;
    s = malloc(size);
    memset(s, 0, size);

    if (base == 2)
    {
        for (size_t j = 0; j < z.size; ++j)
        {
            z_type b = z.bits[z.size - 1 - j];

            for (size_t i = 0; i < sizeof(z_type) * 8; ++i)
            {
                char* sTmp = malloc(strlen(s) + 1);
                memcpy(sTmp, s, strlen(s) + 1);

                sprintf(s, "%d%s", (b & 1 ? 1 : 0), sTmp);
                b >>= 1;

                free(sTmp);
            }
        }

        char* sTmp = malloc(strlen(s) + 1);
        memcpy(sTmp, s, strlen(s) + 1);

        sprintf(s, "0b%s", sTmp);

        free(sTmp);
    }
    else if (base == 8)
    {
        z_t number = z_abs(z);

        if (!z_cmp_c(number, 0))
            strcpy(s, "0");

        while (z_cmp_c(number, 0))
        {
            z_t tmp = z_div_r_c(number, 8);

            char* sTmp = malloc(strlen(s) + 1);
            memcpy(sTmp, s, strlen(s) + 1);

            sprintf(s, "%d%s", z_to_s(tmp), sTmp);

            free(sTmp);

            z_free(&tmp);

            z_t q = z_div_q_c(number, 8);
            z_free(&number);
            number = q;
        }

        char* sTmp = malloc(strlen(s) + 1);
        memcpy(sTmp, s, strlen(s) + 1);

        sprintf(s, "0o%s", sTmp);

        free(sTmp);

        z_free(&number);
    }
    else if (base == 10)
    {
        z_t number = z_abs(z);

        if (z.size == 1)
            sprintf(s, "%llu", z.bits[0]);
        else
        {
            if (!z_cmp_c(number, 0))
                strcpy(s, "0");

            z_type zero = 0;
            z_type const n = log10(~zero);
            z_type const b = pow(10, n);

            while (z_cmp_c(number, 0))
            {
                z_t tmp = z_div_r_ull(number, b);

                char* s1Tmp = filled_str(z_to_ull(tmp), n, '0');
                char* sTmp = malloc(strlen(s) + 1);
                memcpy(sTmp, s, strlen(s) + 1);
                sprintf(s, "%s%s", s1Tmp, sTmp);

                free(s1Tmp);
                free(sTmp);

                z_free(&tmp);

                z_t q = z_div_q_ull(number, b);
                z_free(&number);
                number = q;
            }

            size_t i = 0;

            while (s[i] == '0' && i < strlen(s))
                ++i;

            if (i == strlen(s))
                i = strlen(s) - 1;

            char* sTmp = malloc(strlen(s) + 1);
            memcpy(sTmp, s + i, strlen(s) + 1 - i);
            free(s);
            s = sTmp;
        }

        z_free(&number);
    }
    else if (2 < base && base < 16)
    {
        z_t number = z_abs(z);

        if (z.size == 1)
            sprintf(s, "%llu", z.bits[0]);
        else
        {
            if (!z_cmp_c(number, 0))
                strcpy(s, "0");

            while (z_cmp_c(number, 0))
            {
                z_t tmp = z_div_r_ull(number, base);

                char* sTmp = malloc(strlen(s) + 1);
                memcpy(sTmp, s, strlen(s) + 1);

                sprintf(s, "%d%s", z_to_s(tmp), sTmp);

                free(sTmp);

                z_free(&tmp);

                z_t q = z_div_q_ull(number, base);
                z_free(&number);
                number = q;
            }
        }

        z_free(&number);
    }
    else if (base == 16)
    {
        z_t number = z_abs(z);

        if (!z_cmp_c(number, 0))
            strcpy(s, "0");

        while (z_cmp_c(number, 0))
        {
            z_t tmp = z_div_r_c(number, 16);

            if (z_cmp_c(tmp, 10) < 0)
            {
                char* sTmp = malloc(strlen(s) + 1);
                memcpy(sTmp, s, strlen(s) + 1);

                sprintf(s, "%d%s", z_to_s(tmp), sTmp);

                free(sTmp);
            }
            else
            {
                char* sTmp = malloc(strlen(s) + 1);
                memcpy(sTmp, s, strlen(s) + 1);

                sprintf(s, "%c%s", 'a' + z_to_s(tmp) - 10, sTmp);

                free(sTmp);
            }

            z_t q = z_div_q_c(number, 16);
            z_free(&number);
            number = q;

            z_free(&tmp);
        }

        char* sTmp = malloc(strlen(s) + 1);
        memcpy(sTmp, s, strlen(s) + 1);

        sprintf(s, "0x%s", sTmp);

        free(sTmp);

        z_free(&number);
    }
    else if (base <= 62)
    {
        z_t number = z_abs(z);

        if (!z_cmp_c(number, 0))
            strcpy(s, "0");

        while (z_cmp_c(number, 0))
        {
            z_t tmp = z_div_r_ull(number, base);

            if (z_cmp_c(tmp, 10) < 0)
            {
                char* sTmp = malloc(strlen(s) + 1);
                memcpy(sTmp, s, strlen(s) + 1);

                sprintf(s, "%d%s", z_to_s(tmp), sTmp);

                free(sTmp);
            }
            else if (z_cmp_c(tmp, -36) < 0)
            {
                char* sTmp = malloc(strlen(s) + 1);
                memcpy(sTmp, s, strlen(s) + 1);

                sprintf(s, "%c%s", 'a' + z_to_s(tmp) - 10, sTmp);

                free(sTmp);
            }
            else
            {
                char* sTmp = malloc(strlen(s) + 1);
                memcpy(sTmp, s, strlen(s) + 1);

                sprintf(s, "%c%s", 'A' + z_to_s(tmp) - 36, sTmp);

                free(sTmp);
            }

            z_t q = z_div_q_ull(number, base);
            z_free(&number);
            number = q;

            z_free(&tmp);
        }

        z_free(&number);
    }

    if (!z.size && (!strlen(s) && s[strlen(s) - 1] != '0'))
    {
        char* sTmp = malloc(strlen(s) + 1);
        memcpy(sTmp, s, strlen(s) + 1);

        sprintf(s, "%s0", sTmp);

        free(sTmp);
    }

    if (!z_is_positive(z))
    {
        char* sTmp = malloc(strlen(s) + 1);
        memcpy(sTmp, s, strlen(s) + 1);

        sprintf(s, "-%s", sTmp);

        free(sTmp);
    }

    return s;
}
