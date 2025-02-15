#include <assert.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#include "z_from.h"

#define Z_FROM(suffix, type)                                            \
z_t z_from_##suffix(type n)                                             \
{                                                                       \
    z_t z;                                                              \
                                                                        \
    z.is_positive = (n >= 0);                                           \
    z.size = sizeof(type) / sizeof(z_type);                             \
                                                                        \
    if (sizeof(type) % sizeof(z_type))                                  \
        ++z.size;                                                       \
                                                                        \
    z.bits = malloc(z.size * sizeof(z_type));                           \
                                                                        \
    z.is_nan = false;                                                   \
    z.is_infinity = false;                                              \
    z.is_auto_adjust = true;                                            \
                                                                        \
    if (n < 0)                                                          \
        n = -n;                                                         \
                                                                        \
    assert(z.bits);                                                     \
                                                                        \
    size_t const size = sizeof(type);                                   \
                                                                        \
    memset((void*)(z.bits) + size, 0, z.size * sizeof(z_type) - size);  \
    memcpy(z.bits, &n, size);                                           \
                                                                        \
    return z;                                                           \
}

Z_FROM(c, char)
Z_FROM(i, int)
Z_FROM(l, long)
Z_FROM(ll, long long)
Z_FROM(s, short)
Z_FROM(uc, unsigned char)
Z_FROM(ui, unsigned int)
Z_FROM(ul, unsigned long)
Z_FROM(ull, unsigned long long)
Z_FROM(us, unsigned short)

z_t z_from_data(void const* data, size_t size)
{
    assert(data);

    z_t z;

    z.is_positive = true;
    z.size = size / sizeof(z_type);

    if (size % sizeof(z_type))
        ++z.size;

    z.bits = malloc(z.size * sizeof(z_type));
    z.is_nan = false;
    z.is_infinity = false;
    z.is_auto_adjust = true;

    if (size)
        assert(z.bits);

    memset((void*)(z.bits) + size, 0, z.size * sizeof(z_type) - size);
    memcpy(z.bits, data, size);

    for (size_t i = 0; i < size / 2; ++i)
    {
        char tmp = *((char*)(z.bits) + size - 1 - i);
        *((char*)(z.bits) + size - 1 - i) = *((char*)(z.bits) + i);
        *((char*)(z.bits) + i) = tmp;
    }

    return z;
}

z_t z_from_str(char const* n, size_t base)
{
    assert(n);

    z_t z = z_from_c(0);

    char* nTmp = malloc(strlen(n) + 1);
    memset(nTmp, 0, strlen(n) + 1);

    size_t j = 0;

    for (size_t i = 0; i < strlen(n); ++i)
    {
        if (!isspace(n[i]) && n[i] != '\'')
        {
            nTmp[j] = tolower(n[i]);
            ++j;
        }
    }

    j = 0;

    if (nTmp[j] == '-')
    {
        z.is_positive = false;
        ++j;
    }

    if (!base)
    {
        char const* s = nTmp + j + 2;

        if (s[0] == 'b' || (s[0] == '0' && s[1] == 'b'))
            base = 2;
        else if (s[0] == 'o' || (s[0] == '0' && s[1] == 'o'))
            base = 8;
        else if (s[0] == 'x' || (s[0] == '0' && s[1] == 'x'))
            base = 16;
        else
            base = 10;
    }

    assert(2 <= base && base <= 62);

    char const* str = nTmp + j;

    if (!strcmp(str, "nan"))
        z_set_nan(&z);
    else if (!strcmp(str, "inf"))
        z_set_infinity(&z);
    else
    {
        bool const is_positive = z.is_positive;

        if (base == 2)
        {
            if (str[0] == 'b')
                ++j;
            else if (str[0] == '0' && str[1] == 'b')
                j += 2;

            while (j < strlen(str))
            {
                if (str[j] == '1')
                {
                    z_lshift_c(&z, 1);
                    z_or_c(&z, 1);
                }
                else if (str[j] == '0')
                    z_lshift_c(&z, 1);

                ++j;
            }
        }
        else if (base == 8)
        {
            if (str[0] == 'o')
                ++j;
            else if (str[0] == '0' && str[1] == 'o')
                j += 2;

            size_t otherJ = strlen(nTmp) - 1;
            z_t p = z_from_c(1);

            while (otherJ != j - 1)
            {
                if ('0' <= nTmp[otherJ] && nTmp[otherJ] <= '7')
                {
                    z_t tmp = z_copy(p);
                    z_mul_c(&tmp, nTmp[otherJ] - '0');
                    z_add_z(&z, tmp);
                    z_free(&tmp);
                    z_mul_ull(&p, base);
                }

                --otherJ;
            }

            z_free(&p);
        }
        else if (base <= 10)
        {
            size_t otherJ = strlen(nTmp) - 1;
            z_t p = z_from_c(1);

            while (otherJ != j - 1)
            {
                if ('0' <= nTmp[otherJ] && nTmp[otherJ] <= '0' + base)
                {
                    z_t tmp = z_copy(p);
                    z_mul_c(&tmp, nTmp[otherJ] - '0');
                    z_add_z(&z, tmp);
                    z_free(&tmp);
                    z_mul_ull(&p, base);
                }

                --otherJ;
            }

            z_free(&p);
        }
        else if (base < 16)
        {
            size_t otherJ = strlen(nTmp) - 1;
            z_t p = z_from_c(1);

            while (otherJ != j - 1)
            {
                if ('0' <= nTmp[otherJ] && nTmp[otherJ] <= '9')
                {
                    z_t tmp = z_copy(p);
                    z_mul_c(&tmp, nTmp[otherJ] - '0');
                    z_add_z(&z, tmp);
                    z_free(&tmp);
                    z_mul_ull(&p, base);
                }
                else if ('a' <= nTmp[otherJ] && nTmp[otherJ] <= 'a' + base - 10)
                {
                    z_t tmp = z_copy(p);
                    z_mul_c(&tmp, nTmp[otherJ] - 'a' + 10);
                    z_add_z(&z, tmp);
                    z_free(&tmp);
                    z_mul_ull(&p, base);
                }

                --otherJ;
            }

            z_free(&p);
        }
        else if (base == 16)
        {
            if (str[0] == 'x')
                ++j;
            else if (str[0] == '0' && str[1] == 'x')
                j += 2;

            size_t otherJ = strlen(nTmp) - 1;
            z_t p = z_from_c(1);

            while (otherJ != j - 1)
            {
                if ('0' <= nTmp[otherJ] && nTmp[otherJ] <= '9')
                {
                    z_t tmp = z_copy(p);
                    z_mul_c(&tmp, nTmp[otherJ] - '0');
                    z_add_z(&z, tmp);
                    z_free(&tmp);
                    z_mul_ull(&p, base);
                }
                else if ('a' <= nTmp[otherJ] && nTmp[otherJ] <= 'f')
                {
                    z_t tmp = z_copy(p);
                    z_mul_c(&tmp, nTmp[otherJ] - 'a' + 10);
                    z_add_z(&z, tmp);
                    z_free(&tmp);
                    z_mul_ull(&p, base);
                }

                --otherJ;
            }

            z_free(&p);
        }
        else// if (base <= 62)
        {
            size_t otherJ = strlen(nTmp) - 1;
            z_t p = z_from_c(1);

            while (otherJ != j - 1)
            {
                if ('0' <= nTmp[otherJ] && nTmp[otherJ] <= '9')
                {
                    z_t tmp = z_copy(p);
                    z_mul_c(&tmp, nTmp[otherJ] - '0');
                    z_add_z(&z, tmp);
                    z_free(&tmp);
                    z_mul_ull(&p, base);
                }
                else if ('a' <= nTmp[otherJ] && nTmp[otherJ] <= 'z')
                {
                    z_t tmp = z_copy(p);
                    z_mul_c(&tmp, nTmp[otherJ] - 'a' + 10);
                    z_add_z(&z, tmp);
                    z_free(&tmp);
                    z_mul_ull(&p, base);
                }
                else if ('A' <= nTmp[otherJ] && nTmp[otherJ] <= 'Z')
                {
                    z_t tmp = z_copy(p);
                    z_mul_c(&tmp, nTmp[otherJ] - 'A' + 36);
                    z_add_z(&z, tmp);
                    z_free(&tmp);
                    z_mul_ull(&p, base);
                }

                --otherJ;
            }

            z_free(&p);
        }

        z.is_positive = is_positive;
    }

    free(nTmp);

    z_adjust(&z);

    return z;
}
