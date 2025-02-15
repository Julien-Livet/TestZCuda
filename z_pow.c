#include <assert.h>

#include "z_pow.h"

#define Z_POW(suffix, type)             \
z_t z_pow_##suffix(z_t base, type exp)  \
{                                       \
    z_t e = z_from_##suffix(exp);       \
                                        \
    z_t p = z_pow_z(base, e);           \
                                        \
    z_free(&e);                         \
                                        \
    return p;                           \
}

Z_POW(c, char)
Z_POW(i, int)
Z_POW(l, long)
Z_POW(ll, long long)
Z_POW(s, short)
Z_POW(uc, unsigned char)
Z_POW(ui, unsigned int)
Z_POW(ul, unsigned long)
Z_POW(ull, unsigned long long)
Z_POW(us, unsigned short)

z_t z_pow_z(z_t base, z_t exp)
{
    assert(z_cmp_c(exp, 0) >= 0);

    z_t pow = z_from_c(0);

    if (z_is_infinity(base) || z_is_nan(base))
    {
        z_free(&pow);
        pow = z_copy(base);

        return pow;
    }
    else if (z_is_nan(exp) || z_is_infinity(exp))
    {
        z_free(&pow);
        pow = z_copy(exp);

        return pow;
    }
    else if (z_cmp_c(base, 0) < 0)
    {
        z_free(&pow);

        z_t base_abs = z_abs(base);

        pow = z_pow_z(base_abs, exp);

        z_free(&base_abs);

        z_t a = z_copy(exp);
        z_and_c(&a, 1);

        if (z_cmp_c(a, 0))
            pow.is_positive = !pow.is_positive;
            
        z_free(&a);

        return pow;
    }

    if (z_cmp_c(base, 2))
    {
        z_free(&pow);

        pow = z_from_c(1);
        z_rshift_z(&pow, exp);

        return pow;
    }

    z_free(&pow);

    z_t result = z_from_c(1);
    z_t e = z_copy(exp);
    z_t b = z_copy(base);

    for (;;)
    {
        z_t a = z_copy(e);
        z_and_c(&a, 1);

        if (z_cmp_c(a, 0))
            z_mul_z(&result, b);

        z_free(&a);

        z_rshift_c(&e, 1);

        if (!z_cmp_c(e, 0))
            break;

	z_t bTmp = z_copy(b);

        z_mul_z(&b, bTmp);
        
        z_free(&bTmp);
    }

    z_free(&e);
    z_free(&b);

    return result;
}
