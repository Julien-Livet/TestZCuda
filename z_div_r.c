#include <stdlib.h>

#include "z_div_r.h"

#define Z_DIV_R(suffix, type)           \
z_t z_div_r_##suffix(z_t lhs, type rhs) \
{                                       \
    z_t other = z_from_##suffix(rhs);   \
                                        \
    z_t r = z_div_r_z(lhs, other);      \
                                        \
    z_free(&other);                     \
                                        \
    return r;                           \
}

Z_DIV_R(c, char)
Z_DIV_R(i, int)
Z_DIV_R(l, long)
Z_DIV_R(ll, long long)
Z_DIV_R(s, short)
Z_DIV_R(uc, unsigned char)
Z_DIV_R(ui, unsigned int)
Z_DIV_R(ul, unsigned long)
Z_DIV_R(ull, unsigned long long)
Z_DIV_R(us, unsigned short)

z_t z_div_r_z(z_t lhs, z_t rhs)
{
    z_t q = z_from_c(0);
    z_t r = z_from_c(0);

    z_div_qr_z(lhs, rhs, &q, &r);

    z_free(&q);

    return r;
}
