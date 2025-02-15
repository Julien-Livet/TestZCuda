#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "z_lshift.h"

#define Z_LSHIFT(suffix, type)              \
void z_lshift_##suffix(z_t* lhs, type rhs)  \
{                                           \
    assert(lhs);                            \
                                            \
    z_t other = z_from_##suffix(rhs);       \
                                            \
    z_lshift_z(lhs, other);                 \
                                            \
    z_free(&other);                         \
}

Z_LSHIFT(c, char)
Z_LSHIFT(i, int)
Z_LSHIFT(l, long)
Z_LSHIFT(ll, long long)
Z_LSHIFT(s, short)
Z_LSHIFT(uc, unsigned char)
Z_LSHIFT(ui, unsigned int)
Z_LSHIFT(ul, unsigned long)
Z_LSHIFT(ull, unsigned long long)
Z_LSHIFT(us, unsigned short)

void z_lshift_z(z_t* lhs, z_t rhs)
{
    assert(lhs);

    assert(z_cmp_c(rhs, 0) >= 0);

    if (!z_cmp_c(*lhs, 0) || !z_cmp_c(rhs, 0))
        return;
    else if (z_is_nan(*lhs) || z_is_nan(rhs) || z_is_infinity(*lhs) || z_is_infinity(rhs))
    {
        z_set_nan(lhs);

        return;
    }

    unsigned short const us = sizeof(z_type) * 8;
    z_t n = z_div_q_us(rhs, us);

    {
        z_type* bits = malloc((z_to_ull(n) + lhs->size) * sizeof(z_type));
        memset((void*)(bits) + lhs->size * sizeof(z_type), 0, z_to_ull(n) * sizeof(z_type));
        memcpy(bits, lhs->bits, lhs->size * sizeof(z_type));
        free(lhs->bits);
        lhs->bits = bits;
        lhs->size += z_to_ull(n);
    }

    z_t other = z_copy(rhs);
    z_t nTmp = z_copy(n);
    z_mul_us(&nTmp, us);
    z_sub_z(&other, nTmp);
    z_free(&nTmp);

    z_type* bits = malloc((lhs->size + 1) * sizeof(z_type));
    memset(bits, 0, sizeof(z_type));
    memcpy((void*)(bits) + sizeof(z_type), lhs->bits, lhs->size * sizeof(z_type));
    free(lhs->bits);
    lhs->bits = bits;
    ++lhs->size;

    unsigned long long const shift = z_to_ull(other);

    if (shift)
    {
        for (size_t i = 1; i < lhs->size; ++i)
        {
            longest_type const s = sizeof(z_type) * 8;

            if ((lhs->bits[i] >> (s - shift)))
                lhs->bits[i - 1] |= (lhs->bits[i] >> (s - shift));

            lhs->bits[i] <<= shift;
        }
    }

    z_free(&n);
    z_free(&other);

    if (z_is_auto_adjust(*lhs))
        z_adjust(lhs);
}
