#include <assert.h>

#include "z_powm.h"

#define Z_POWM(suffix, type)                        \
z_t z_powm_##suffix(z_t base, type exp, type mod)   \
{                                                   \
    z_t e = z_from_##suffix(exp);                   \
    z_t m = z_from_##suffix(mod);                   \
                                                    \
    z_t p = z_powm_z(base, e, m);                   \
                                                    \
    z_free(&e);                                     \
    z_free(&m);                                     \
                                                    \
    return p;                                       \
}

Z_POWM(c, char)
Z_POWM(i, int)
Z_POWM(l, long)
Z_POWM(ll, long long)
Z_POWM(s, short)
Z_POWM(uc, unsigned char)
Z_POWM(ui, unsigned int)
Z_POWM(ul, unsigned long)
Z_POWM(ull, unsigned long long)
Z_POWM(us, unsigned short)

z_t z_powm_z(z_t base, z_t exp, z_t mod)
{
    assert(z_cmp_c(exp, 0) >= 0);

    z_t result = z_from_c(1);
    z_t base_mod = z_div_r_z(base, mod);
    z_t e = z_copy(exp);

    while (z_cmp_c(e, 0) > 0)
    {
        z_t a = z_copy(e);
        z_and_c(&a, 1);

        if (!z_cmp_c(a, 1))
        {
            z_mul_z(&result, base_mod);

            z_t r = z_div_r_z(result, mod);

            z_free(&result);
            result = r;
        }

        z_free(&a);

	z_t base_modTmp = z_copy(base_mod);

        z_mul_z(&base_mod, base_modTmp);
        
        z_free(&base_modTmp);

        z_t r = z_div_r_z(base_mod, mod);

        z_free(&base_mod);
        base_mod = r;

        z_rshift_c(&e, 1);
    }

    z_free(&base_mod);
    z_free(&e);

    return result;
}
