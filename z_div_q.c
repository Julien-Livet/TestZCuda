#include "z_div_q.h"

#define Z_DIV_Q(suffix, type)           \
z_t z_div_q_##suffix(z_t lhs, type rhs) \
{                                       \
    z_t other = z_from_##suffix(rhs);   \
                                        \
    z_t q = z_div_q_z(lhs, other);      \
                                        \
    z_free(&other);                     \
                                        \
    return q;                           \
}

Z_DIV_Q(c, char)
Z_DIV_Q(i, int)
Z_DIV_Q(l, long)
Z_DIV_Q(ll, long long)
Z_DIV_Q(s, short)
Z_DIV_Q(uc, unsigned char)
Z_DIV_Q(ui, unsigned int)
Z_DIV_Q(ul, unsigned long)
Z_DIV_Q(ull, unsigned long long)
Z_DIV_Q(us, unsigned short)

z_t z_div_q_z(z_t lhs, z_t rhs)
{
    z_t q = z_from_c(0);
    z_t r = z_from_c(0);

    z_div_qr_z(lhs, rhs, &q, &r);

    if (z_cmp_c(lhs, 0) < 0 && z_cmp_c(rhs, 0) < 0)
    {
        if (z_cmp_c(r, 0))
            z_sub_c(&q, 1);
    }
    else if (z_cmp_c(lhs, 0) > 0 && z_cmp_c(rhs, 0) < 0)
    {
        if (z_cmp_c(r, 0))
            z_add_c(&q, 1);
    }
    else if (z_cmp_c(lhs, 0) < 0 && z_cmp_c(rhs, 0) > 0)
    {
        if (z_cmp_c(r, 0))
            z_add_c(&q, 1);
    }

    z_free(&r);

    return q;
}
