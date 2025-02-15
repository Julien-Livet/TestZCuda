#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "z_t.h"

z_t z_abs(z_t z)
{
    z_t res = z_copy(z);

    res.is_positive = true;

    return res;
}

void z_adjust(z_t* z)
{
    assert(z);

    if (!z->size)
        return;

    assert(z->bits);

    size_t i = 0;

    while (!z->bits[i] && i < z->size)
        ++i;

    if (i == z->size)
        i = z->size - 1;

    if (i != 0)
    {
        z_type* bits = malloc((z->size - i) * sizeof(z_type));
        memcpy(bits, (void*)(z->bits) + i * sizeof(z_type), (z->size - i) * sizeof(z_type));
        free(z->bits);
        z->bits = bits;
        z->size = z->size - i;
    }
}

z_t z_copy(z_t z)
{
    z_t other = z;

    other.bits = malloc(other.size * sizeof(z_type));
    memcpy(other.bits, z.bits, other.size * sizeof(z_type));

    return other;
}

z_t z_factorial(z_t n)
{
    z_t res = z_from_c(0);

    if (z_is_nan(n) || z_is_infinity(n))
    {
        z_free(&res);
        res = z_copy(n);

        return res;
    }

    assert(z_cmp_c(n, 0) >= 0);

    if (!z_cmp_c(n, 0))
    {
        z_free(&res);
        res = z_from_c(1);

        return res;
    }

    z_free(&res);
    res = z_copy(n);

    z_t n_1 = z_copy(n);
    z_sub_c(&n_1, 1);

    z_t f = z_factorial(n_1);

    z_mul_z(&res, f);

    z_free(&n_1);
    z_free(&f);

    return res;
}

void z_free(z_t* z)
{
    assert(z);

    free(z->bits);

    z->bits = NULL;
    z->size = 0;
    z->is_nan = true;
    z->is_infinity = false;
}

z_t z_gcd(z_t a, z_t b)
{
    if (z_is_nan(a) || z_is_nan(b) || z_is_infinity(a) || z_is_infinity(b))
        return z_nan();
    else if (z_cmp_c(a, 0) < 0)
    {
        z_t aTmp = z_abs(a);

        z_t gcd = z_gcd(aTmp, b);

        z_free(&aTmp);

        return gcd;
    }
    else if (z_cmp_c(b, 0) < 0)
    {
        z_t bTmp = z_abs(b);

        z_t gcd = z_gcd(a, bTmp);

        z_free(&bTmp);

        return gcd;
    }
    else if (z_cmp_z(a, b) < 0)
        return z_gcd(b, a);
    else if (!z_cmp_c(a, 0))
        return z_copy(b);
    else if (!z_cmp_c(b, 0))
        return z_copy(a);
    else if (z_is_even(a) && z_is_even(b))
    {
        z_t aTmp = z_copy(a);
        z_rshift_c(&aTmp, 1);
        z_t bTmp = z_copy(b);
        z_rshift_c(&bTmp, 1);

        z_t gcd = z_gcd(aTmp, bTmp);
        z_rshift_c(&gcd, 1);

        z_free(&aTmp);
        z_free(&bTmp);

        return gcd;
    }
    else if (z_is_odd(a) && z_is_even(b))
    {
        z_t bTmp = z_copy(b);
        z_rshift_c(&bTmp, 1);

        z_t gcd = z_gcd(a, bTmp);

        z_free(&bTmp);

        return gcd;
    }
    else if (z_is_even(a) && z_is_odd(b))
    {
        z_t aTmp = z_copy(a);
        z_rshift_c(&aTmp, 1);

        z_t gcd = z_gcd(aTmp, b);

        z_free(&aTmp);

        return gcd;
    }
    else //if (z_is_odd(a) && z_is_odd(b))
    {
        z_t aTmp = z_copy(a);
        z_sub_z(&aTmp, b);
        z_rshift_c(&aTmp, 1);

        z_t gcd = z_gcd(aTmp, b);

        z_free(&aTmp);

        return gcd;
    }
}

z_t z_gcd_extended(z_t a, z_t b, z_t* u, z_t* v)
{
    assert(u && v);

    if (z_is_nan(a) || z_is_nan(b) || z_is_infinity(a) || z_is_infinity(b))
    {
        z_set_nan(u);
        z_set_nan(v);

        return z_nan();
    }

    if (!z_cmp_c(a, 0) && !z_cmp_c(b, 0))
        return z_from_c(0);

    z_t r1 = z_copy(a);
    z_t u1 = z_from_c(1);
    z_t v1 = z_from_c(0);
    z_t r2 = z_copy(b);
    z_t u2 = z_from_c(0);
    z_t v2 = z_from_c(1);
    z_t q = z_from_c(0);
    z_t r_temp = z_from_c(0);
    z_t u_temp = z_from_c(0);
    z_t v_temp = z_from_c(0);

    while (z_cmp_c(r2, 0))
    {
        z_free(&q);
        q = z_div_q_z(r1, r2);

        z_free(&r_temp);
        r_temp = z_copy(r2);
        z_neg(&r_temp);
        z_mul_z(&r_temp, q);
        z_add_z(&r_temp, r1);

        z_free(&u_temp);
        u_temp = z_copy(u2);
        z_neg(&u_temp);
        z_mul_z(&u_temp, q);
        z_add_z(&u_temp, u1);

        z_free(&v_temp);
        v_temp = z_copy(v2);
        z_neg(&v_temp);
        z_mul_z(&v_temp, q);
        z_add_z(&v_temp, v1);

        z_free(&r1);
        r1 = z_copy(r2);
        z_free(&u1);
        u1 = z_copy(u2);
        z_free(&v1);
        v1 = z_copy(v2);
        z_free(&r2);
        r2 = z_copy(r_temp);
        z_free(&u2);
        u2 = z_copy(u_temp);
        z_free(&v2);
        v2 = z_copy(v_temp);
    }

    z_free(u);
    *u = z_copy(u1);
    z_free(v);
    *v = z_copy(v1);

    z_free(&u1);
    z_free(&v1);
    z_free(&r2);
    z_free(&u2);
    z_free(&v2);
    z_free(&q);
    z_free(&r_temp);
    z_free(&u_temp);
    z_free(&v_temp);

    return r1;
}

z_t z_infinity()
{
    z_t z;

    z.is_positive = true;
    z.bits = NULL;
    z.size = 0;
    z.is_nan = false;
    z.is_infinity = true;
    z.is_auto_adjust = true;

    return z;
}

void z_invert(z_t* z)
{
    assert(z);

    assert(z->bits);

    for (size_t i = 0; i < z->size; ++i)
        z->bits[i] = ~z->bits[i];
    
    z->is_positive = !z->is_positive;
}

bool z_is_auto_adjust(z_t z)
{
    return z.is_auto_adjust;
}

bool z_is_nan(z_t z)
{
    return z.is_nan;
}

bool z_is_even(z_t z)
{
    if (!z.size)
        return false;

    return !(z.bits[z.size - 1] & 1);
}

bool z_is_infinity(z_t z)
{
    return z.is_infinity;
}

bool z_is_negative(z_t z)
{
    return !z.is_positive;
}

bool z_is_null(z_t z)
{
    return !z_cmp_c(z, 0);
}

bool z_is_odd(z_t z)
{
    if (!z.size)
        return false;

    return z.bits[z.size - 1] & 1;
}

bool z_is_positive(z_t z)
{
    return z.is_positive;
}

z_t z_max(z_t a, z_t b)
{
    if (z_cmp_z(a, b) > 0)
        return a;
    else
        return b;
}

z_t z_min(z_t a, z_t b)
{
    if (z_cmp_z(a, b) < 0)
        return a;
    else
        return b;
}

z_t z_nan()
{
    z_t z;

    z.is_positive = true;
    z.bits = NULL;
    z.size = 0;
    z.is_nan = true;
    z.is_infinity = false;
    z.is_auto_adjust = true;

    return z;
}

void z_neg(z_t* z)
{
    assert(z);

    z->is_positive = !z->is_positive;
}

z_t z_number(z_t z)
{
    z_t number = z_from_c(0);

    if (z_is_nan(z) || z_is_infinity(z))
        return number;

    size_t i = 0;

    while (!z.bits[i] && i < z.size)
        ++i;

    if (i < z.size)
    {
        z_type b = z.bits[i];

        while (b)
        {
            z_add_c(&number, 1);
            b >>= 1;
        }

        z_add_ull(&number, (z.size - i - 1) * sizeof(z_type) * 8);
    }

    return number;
}

size_t z_precision(z_t z)
{
    return z.size;
}

void z_printf(z_t z, size_t base)
{
    char* s = z_to_str(z, base);

    printf(s);

    free(s);
}

void z_printf_bits(z_t z)
{
    for (size_t i = 0; i < z.size; ++i)
        printf("%llu ", z.bits[i]);
}

void z_printf_bytes(z_t z)
{
    for (size_t i = 0; i < z.size * sizeof(z_type); ++i)
        printf("%d ", *((char*)(z.bits) + i));
}

void z_set_auto_adjust(z_t* z, bool is_auto_adjust)
{
    assert(z);

    z->is_auto_adjust = is_auto_adjust;
}

void z_set_infinity(z_t* z)
{
    assert(z);

    free(z->bits);
    z->bits = NULL;
    z->size = 0;
    z->is_nan = false;
    z->is_infinity = true;
}

void z_set_nan(z_t* z)
{
    assert(z);

    free(z->bits);
    z->bits = NULL;
    z->size = 0;
    z->is_nan = true;
    z->is_infinity = false;
}

void z_set_negative(z_t* z)
{
    assert(z);

    z->is_positive = false;
}

void z_set_positive(z_t* z)
{
    assert(z);

    z->is_positive = true;
}

void z_set_precision(z_t* z, size_t precision)
{
    assert(z);
    assert(precision);

    if (precision == z->size)
        return;

    z_type* bits = z->bits;
    z->bits = malloc(precision * sizeof(z_type));

    if (precision > z->size)
    {
        memset(z->bits, 0, (precision - z->size) * sizeof(z_type));
        memcpy((void*)(z->bits) + (precision - z->size) * sizeof(z_type), bits, z->size * sizeof(z_type));
    }
    else
        memcpy(z->bits, (void*)(bits) + (z->size - precision) * sizeof(z_type), precision * sizeof(z_type));

    z->size = precision;
    free(bits);
}

void z_set_random(z_t* z)
{
    assert(z);

    z->is_nan = false;
    z->is_infinity = false;
    z->is_positive = rand() % 2;

    for (size_t i = 0; i < z->size; ++i)
    {
        int const n = rand();

        if (sizeof(z_type) <= sizeof(n))
            z->bits[i] = n;
        else
        {
            z->bits[i] = 0;

            size_t const jMax = sizeof(z_type) / sizeof(n);

            for (size_t j = 0; j < jMax; ++j)
            {
                z->bits[i] <<= sizeof(n) * 8;
                z->bits[i] |= rand();
            }
        }
    }
}

int z_sign(z_t z)
{
    if (z_is_positive(z))
        return 1;
    else
        return -1;
}

z_t z_sqrt(z_t n)
{
    z_t sqrt = z_from_c(0);

    if (z_cmp_c(n, 0) < 0)
    {
        z_set_nan(&sqrt);

        return sqrt;
    }
    else if (!z_cmp_c(n, 0) || !z_cmp_c(n, 1) || z_is_nan(n) || z_is_infinity(n))
    {
        z_free(&sqrt);

        sqrt = z_copy(n);

        return sqrt;
    }

    z_free(&sqrt);

    z_t lo = z_from_c(1);
    z_t hi = z_copy(n);
    z_t res = z_from_c(1);

    while (z_cmp_z(lo, hi) <= 0)
    {
        z_t mid = z_copy(lo);
        z_add_z(&mid, hi);
        z_rshift_c(&mid, 1);

        z_t mid2 = z_copy(mid);
        z_mul_z(&mid2, mid);

        if (z_cmp_z(mid2, n) <= 0)
        {
            z_free(&res);
            res = z_copy(mid);
            z_free(&lo);
            lo = z_copy(mid);
            z_add_c(&lo, 1);
        }
        else
        {
            z_free(&hi);
            hi = z_copy(mid);
            z_sub_c(&hi, 1);
        }

        z_free(&mid);
        z_free(&mid2);
    }

    z_free(&lo);
    z_free(&hi);

    return res;
}
