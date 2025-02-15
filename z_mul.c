#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "z_mul.h"

size_t number(longest_type n)
{
    size_t number = 0;

    while (n)
    {
        ++number;
        n >>= 1;
    }

    return number;
}

#define Z_MUL(suffix, type)             \
void z_mul_##suffix(z_t* lhs, type rhs) \
{                                       \
    assert(lhs);                        \
                                        \
    z_t other = z_from_##suffix(rhs);   \
                                        \
    z_mul_z(lhs, other);                \
                                        \
    z_free(&other);                     \
}

Z_MUL(c, char)
Z_MUL(i, int)
Z_MUL(l, long)
Z_MUL(ll, long long)
Z_MUL(s, short)
Z_MUL(uc, unsigned char)
Z_MUL(ui, unsigned int)
Z_MUL(ul, unsigned long)
Z_MUL(ull, unsigned long long)
Z_MUL(us, unsigned short)

void z_mul_z(z_t* lhs, z_t rhs)
{
    assert(lhs);

    if (z_is_negative(rhs))
    {
        z_neg(lhs);

        z_t other = z_copy(rhs);
        z_neg(&other);

        z_mul_z(lhs, other);

        z_free(&other);
    }
    else if (z_is_nan(*lhs) || z_is_nan(rhs))
        z_set_nan(lhs);
    else if (!z_cmp_c(*lhs, 0) || !z_cmp_c(rhs, 0))
        z_set_from_c(lhs, 0);
    else if (z_is_infinity(*lhs) || z_is_infinity(rhs))
    {
        z_set_infinity(lhs);

        if (z_is_negative(rhs))
            lhs->is_positive = !lhs->is_positive;
    }
    else
    {
        if (z_is_positive(*lhs) && z_is_positive(rhs))
        {
            z_t rhsTmp = z_copy(rhs);
            z_and_c(&rhsTmp, 1);

            if (z_fits_ull(*lhs) && z_fits_ull(rhs))
            {
                unsigned long long const a = z_to_ull(*lhs);
                unsigned long long const b = z_to_ull(rhs);
                unsigned long long const ab = a * b;

                if (ab / b == a)
                    z_set_from_ull(lhs, ab);
                else
                {
                    //Karatsuba algorithm
                    size_t n = MAX(number(a), number(b));
                    if (n % 2)
                        ++n;
                    size_t const m = n / 2;
                    longest_type zero = 0;
                    z_t x0 = z_from_ull((~zero >> (sizeof(longest_type) * 8 - m)) & a);
                    z_t x1 = z_from_ull((((~zero >> (sizeof(longest_type) * 8 - m)) << m) & a) >> m);
                    z_t y0 = z_from_ull((~zero >> (sizeof(longest_type) * 8 - m)) & b);
                    z_t y1 = z_from_ull((((~zero >> (sizeof(longest_type) * 8 - m)) << m) & b) >> m);

                    z_t z0 = z_copy(x0);
                    z_mul_z(&z0, y0);
                    z_t z1a = z_copy(x1);
                    z_mul_z(&z1a, y0);
                    z_t z1b = z_copy(x0);
                    z_mul_z(&z1b, y1);
                    z_t z1 = z_copy(z1a);
                    z_add_z(&z1, z1b);
                    z_t z2 = z_copy(x1);
                    z_mul_z(&z2, y1);

                    //xy = z2 * 2^(2 * m) + z1 * 2^m + z0
                    z_lshift_ull(&z1, m);
                    z_lshift_ull(&z2, 2 * m);
                    z_set_from_z(lhs, z0);
                    z_add_z(lhs, z1);
                    z_add_z(lhs, z2);

                    z_free(&x0);
                    z_free(&x1);
                    z_free(&y0);
                    z_free(&y1);
                    z_free(&z0);
                    z_free(&z1a);
                    z_free(&z1b);
                    z_free(&z1);
                    z_free(&z2);
                }
            }
            else if (!z_cmp_c(rhsTmp, 0))
            {
                z_t r = z_copy(rhs);
                z_t shift = z_from_c(0);
                z_t rTmp = z_copy(r);
                z_and_c(&rTmp, 1);

                while (!z_cmp_c(rTmp, 0))
                {
                    z_rshift_c(&r, 1);
                    z_add_c(&shift, 1);
                    z_free(&rTmp);
                    rTmp = z_copy(r);
                    z_and_c(&rTmp, 1);
                }

                z_lshift_z(lhs, shift);
                z_mul_z(lhs, r);

                z_free(&r);
                z_free(&rTmp);
                z_free(&shift);
            }
            else
            {
                //Karatsuba algorithm
                //x = x1 * 2^m + x0
                //y = y1 * 2^m + y0

                z_t tmp = z_number(*lhs);
                z_t n1 = z_from_c(0);
                z_t r = z_from_c(0);
                z_div_qr_ull(tmp, sizeof(z_type) * 8, &n1, &r);
                if (z_cmp_c(r, 0))
                    z_add_c(&n1, 1);
                z_free(&tmp);
                z_free(&r);
                r = z_div_r_c(n1, 2);
                if (z_cmp_c(r, 0))
                    z_add_c(&n1, 1);
                z_free(&r);

                tmp = z_number(rhs);
                z_t n2 = z_from_c(0);
                r = z_from_c(0);
                z_div_qr_ull(tmp, sizeof(z_type) * 8, &n2, &r);
                if (z_cmp_c(r, 0))
                    z_add_c(&n2, 1);
                z_free(&tmp);
                z_free(&r);
                r = z_div_r_c(n2, 2);
                if (z_cmp_c(r, 0))
                    z_add_c(&n2, 1);
                z_free(&r);

                size_t const n = MAX(z_to_ull(n1), z_to_ull(n2));
                size_t const m = n / 2;

                z_t x0 = z_from_c(0);
                z_set_precision(&x0, m);
                memcpy((void*)(x0.bits) + (x0.size - MIN(lhs->size, m)) * sizeof(z_type),
                       (void*)(lhs->bits) + (lhs->size - MIN(lhs->size, m)) * sizeof(z_type),
                       MIN(lhs->size, m) * sizeof(z_type));

                z_t x1 = z_from_c(0);
                z_set_precision(&x1, m);
                if (MIN(lhs->size, 2 * m) >= m)
                    memcpy((void*)(x1.bits) + (x1.size - (MIN(lhs->size, 2 * m) - m)) * sizeof(z_type),
                           (void*)(lhs->bits) + (lhs->size - MIN(lhs->size, 2 * m)) * sizeof(z_type),
                           (MIN(lhs->size, 2 * m) - m) * sizeof(z_type));

                z_t y0 = z_from_c(0);
                z_set_precision(&y0, m);
                memcpy((void*)(y0.bits) + (y0.size - MIN(rhs.size, m)) * sizeof(z_type),
                       (void*)(rhs.bits) + (rhs.size - MIN(rhs.size, m)) * sizeof(z_type),
                       MIN(rhs.size, m) * sizeof(z_type));

                z_t y1 = z_from_c(0);
                z_set_precision(&y1, m);
                if (MIN(rhs.size, 2 * m) >= m)
                    memcpy((void*)(y1.bits) + (y1.size - (MIN(rhs.size, 2 * m) - m)) * sizeof(z_type),
                           (void*)(rhs.bits) + (rhs.size - MIN(rhs.size, 2 * m)) * sizeof(z_type),
                           (MIN(rhs.size, 2 * m) - m) * sizeof(z_type));

                z_t z0 = z_copy(x0);
                z_mul_z(&z0, y0);
                z_t z1a = z_copy(x1);
                z_mul_z(&z1a, y0);
                z_t z1b = z_copy(x0);
                z_mul_z(&z1b, y1);
                z_t z1 = z_copy(z1a);
                z_add_z(&z1, z1b);
                z_t z2 = z_copy(x1);
                z_mul_z(&z2, y1);

                //o = m * 8 * sizeof(z_type)
                //xy = z2 * 2^(2 * o) + z1 * 2^o + z0

                z_set_from_z(lhs, z0);

                z_t w1 = z_from_c(0);
                z_set_precision(&w1, z1.size + m);
                memcpy(w1.bits, z1.bits, z1.size * sizeof(z_type));

                z_add_z(lhs, w1);

                z_t w2 = z_from_c(0);
                z_set_precision(&w2, z1.size + 2 * m);
                memcpy(w2.bits, z2.bits, z2.size * sizeof(z_type));

                z_add_z(lhs, w2);

                z_free(&x0);
                z_free(&x1);
                z_free(&y0);
                z_free(&y1);
                z_free(&z0);
                z_free(&z1a);
                z_free(&z1b);
                z_free(&z1);
                z_free(&z2);
                z_free(&w1);
                z_free(&w2);
            }

            z_free(&rhsTmp);
        }
        else
        {
            z_t other = z_copy(rhs);
            z_neg(&other);

            z_mul_z(lhs, other);
            z_neg(lhs);

            z_free(&other);
        }
    }

    if (z_is_auto_adjust(*lhs))
        z_adjust(lhs);
}
