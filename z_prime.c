#include <assert.h>
#include <stdlib.h>
#include <time.h>

#include "z_prime.h"

bool z_is_coprime(z_t a, z_t b)
{
    z_t gcd = z_gcd(a, b);

    bool res = !z_cmp_c(gcd, 1);

    z_free(&gcd);

    return res;
}

#include "primes_3_000_000.h"

size_t lower_bound(unsigned int const* array, size_t first, size_t last, unsigned int value)
{
    long long count = last - first;

    while (count > 0)
    {
        size_t i = first;
        long long step = count / 2;
        i += step;

        if (array[i] < value)
        {
            first = ++i;
            count -= step + 1;
        }
        else
            count = step;
    }

    return first;
}

size_t upper_bound(unsigned int const* array, size_t first, size_t last, unsigned int value)
{
    long long count = last - first;

    while (count > 0)
    {
        size_t i = first;
        long long step = count / 2;
        i += step;

        if (!(value < array[i]))
        {
            first = ++i;
            count -= step + 1;
        }
        else
            count = step;
    }

    return first;
}

struct pair_struct
{
    size_t first;
    size_t second;
};

typedef struct pair_struct pair;

pair equal_range(unsigned int const* array, size_t first, size_t last, unsigned int value)
{
    pair const p = {lower_bound(array, first, last, value), upper_bound(array, first, last, value)};

    return p;
}

z_t reduction(z_t t, z_t R, z_t n, z_t n_)
{
    z_t m = z_div_r_z(t, R);
    z_mul_z(&m, n_);
    z_t mTmp = z_div_r_z(m, R);
    z_free(&m);
    m = mTmp;

    z_t x = z_copy(m);
    z_mul_z(&x, n);
    z_add_z(&x, t);
    z_t xTmp = z_div_q_z(x, R);
    z_free(&x);
    x = xTmp;

    z_free(&m);

    if (z_cmp_z(x, n) < 0)
        return x;
    else
    {
        z_sub_z(&x, n);

        return x;
    }
}

z_t redmulmod(z_t a, z_t b, z_t n, z_t R, z_t n_, z_t R2modn)
{
    z_t tmp = z_copy(a);
    z_mul_z(&tmp, R2modn);

    z_t reda = reduction(tmp, R, n, n_);

    z_free(&tmp);
    tmp = z_copy(b);
    z_mul_z(&tmp, R2modn);

    z_t redb = reduction(tmp, R, n, n_);

    z_free(&tmp);
    tmp = z_copy(reda);
    z_mul_z(&tmp, redb);

    z_t redc = reduction(tmp, R, n, n_);

    z_free(&tmp);

    z_t r = reduction(redc, R, n, n_);

    z_free(&reda);
    z_free(&redb);
    z_free(&redc);

    return r;
}

bool mulmod(z_t a, z_t b, z_t m)
{
    z_t x = z_from_c(0);
    z_t y = z_div_r_z(a, m);
    z_t bTmp = z_copy(b);

    while (z_cmp_c(bTmp, 0) > 0)
    {
        z_t tmp = z_copy(bTmp);
        z_and_c(&tmp, 1);

        if (z_cmp_c(tmp, 0))
        {
            z_add_z(&x, y);

            z_t r = z_div_r_z(x, m);
            z_free(&x);
            x = r;
        }

        z_free(&tmp);

        z_lshift_c(&y, 1);

        z_t r = z_div_r_z(y, m);
        z_free(&y);
        y = r;

        z_rshift_c(&bTmp, 1);
    }

    z_t r = z_div_r_z(x, m);

    bool res = z_cmp_c(r, 0);

    z_free(&r);
    z_free(&x);
    z_free(&y);
    z_free(&bTmp);

    return res;
}

z_t modulo(z_t base, z_t e, z_t m, z_t R, z_t m_, z_t R2modm)
{
    z_t x = z_from_c(1);
    z_t y = z_copy(base);
    z_t eTmp = z_copy(e);

    while (z_cmp_c(eTmp, 0) > 0)
    {
        z_t tmp = z_copy(eTmp);
        z_and_c(&tmp, 1);

        if (z_cmp_c(tmp, 0))
        {
            z_t xTmp = redmulmod(x, y, m, R, m_, R2modm);
            z_free(&x);
            x = xTmp;

            while (z_cmp_c(x, 0) < 0)
                z_add_z(&x, m);
        }

        z_free(&tmp);

        z_t yTmp = redmulmod(y, y, m, R, m_, R2modm);
        z_free(&y);
        y = yTmp;

        while (z_cmp_c(y, 0) < 0)
            z_add_z(&y, m);

        z_rshift_c(&eTmp, 1);
    }

    z_t res = z_div_r_z(x, m);

    z_free(&x);
    z_free(&y);
    z_free(&eTmp);

    return res;
}

int z_is_prime(z_t n, size_t reps)
{
    assert(reps);

    if (z_cmp_c(n, 2) < 0)
        return 0;
    else if (!z_cmp_c(n, 2))
        return 2;

    z_t tmp = z_copy(n);
    z_and_c(&tmp, 1);

    if (!z_cmp_c(tmp, 0))
    {
        z_free(&tmp);

        return 0;
    }

    z_free(&tmp);

    if (z_fits_ui(n))
    {
        pair p = equal_range(primes, 0, PRIMES_SIZE, z_to_ui(n));

        if (p.first != PRIMES_SIZE && primes[p.first] != z_to_ui(n))
            --p.first;

        if (p.first != p.second && primes[p.first] == z_to_ui(n))
            return 2;
    }

    //Trial divisions

    {
        z_t sqrtLimit = z_sqrt(n);
        size_t i = 0;

        while (i < PRIMES_SIZE && z_cmp_ui(sqrtLimit, primes[i]) >= 0)
        {
            z_t r = z_div_r_ui(n, primes[i]);

            if (!z_cmp_c(r, 0))
            {
                z_free(&r);
                z_free(&sqrtLimit);

                return 0;
            }

            z_free(&r);

            ++i;
        }
        
        z_free(&sqrtLimit);

        if (i != PRIMES_SIZE)
            return 2;
    }

    //Miller-Rabin tests

    z_t s = z_copy(n);
    z_sub_c(&s, 1);

    tmp = z_copy(s);
    z_and_c(&tmp, 1);

    while (!z_cmp_c(tmp, 0))
    {
        z_rshift_c(&s, 1);
        z_free(&tmp);
        tmp = z_copy(s);
        z_and_c(&tmp, 1);
    }

    z_free(&tmp);

    z_t number = z_copy(n);
    z_sub_c(&number, 1);

    z_t m = z_copy(n);

    z_t R = z_from_c(1);
    tmp = z_number(m);
    z_lshift_z(&R, tmp);
    z_free(&tmp);
    tmp = z_copy(R);
    z_sub_z(&tmp, m);

    assert(z_cmp_c(tmp, 0) > 0);
    z_free(&tmp);

    tmp = z_copy(m);
    z_and_c(&tmp, 1);

    if (!z_cmp_c(tmp, 0))
        z_add_c(&R, 1);

    z_free(&tmp);

    while (!z_is_coprime(m, R))
    {
        tmp = z_copy(m);
        z_and_c(&tmp, 1);

        if (!z_cmp_c(tmp, 0))
            z_sub_c(&R, 1);

        z_free(&tmp);

        z_lshift_c(&R, 1);

        tmp = z_copy(m);
        z_and_c(&tmp, 1);

        if (!z_cmp_c(tmp, 0))
            z_add_c(&R, 1);

        z_free(&tmp);
    }

    z_t R2modm = z_copy(R);
    z_mul_z(&R2modm, R);
    tmp = z_div_r_z(R2modm, m);
    z_free(&R2modm);
    R2modm = tmp;
    z_t R_ = z_from_c(0);
    z_t m_ = z_from_c(0);
    tmp = z_copy(m);
    z_neg(&tmp);
    z_t d = z_gcd_extended(R, tmp, &R_, &m_);
    z_free(&tmp);

    assert(!z_cmp_c(d, 1) || !z_cmp_c(d, -1));

    z_t tmp1 = z_copy(R);
    z_mul_z(&tmp1, R_);
    z_t tmp2 = z_copy(m);
    z_mul_z(&tmp2, m_);
    z_sub_z(&tmp1, tmp2);
    assert(z_cmp_z(tmp1, d) == 0);
    z_free(&tmp1);
    z_free(&tmp2);

    if (!z_cmp_c(d, -1))
    {
        z_neg(&R_);
        z_neg(&m_);
    }

    srand(time(NULL));
    z_t a = z_from_c(0);

    int res = 1;

    for (size_t i = 0; i < reps; ++i)
    {
        z_free(&a);
        a = z_copy(n);
        z_set_random(&a);
        z_set_positive(&a);
        tmp = z_div_r_z(a, number);
        z_free(&a);
        a = tmp;
        z_add_c(&a, 1);

        z_t temp = z_copy(s);
        z_t mod = modulo(a, temp, n, R, m_, R2modm);

        while (z_cmp_z(temp, number) && !z_cmp_c(mod, 0) && z_cmp_z(mod, number))
        {
            bool tmp = mulmod(mod, mod, n);
            z_free(&mod);
            if (tmp)
                z_set_from_c(&mod, 1);
            else
                z_set_from_c(&mod, 0);

            z_lshift_c(&temp, 1);
        }

        tmp = z_copy(temp);
        z_and_c(&tmp, 1);

        if (z_cmp_z(mod, number) && !z_cmp_c(tmp, 0))
        {
            res = 0;
            break;
        }

        z_free(&tmp);
        z_free(&temp);
        z_free(&mod);
    }

    z_free(&R_);
    z_free(&m_);
    z_free(&d);
    z_free(&s);
    z_free(&m);
    z_free(&R);
    z_free(&number);
    z_free(&a);

    return res;
}

z_t z_next_prime(z_t n)
{
    if (z_is_nan(n))
        return z_copy(n);
    else if (z_cmp_c(n, 2) < 0)
        return z_from_c(2);
    else if (!z_cmp_c(n, 2))
        return z_from_c(3);
    else if (z_is_infinity(n))
        return z_nan();
    else if (z_fits_ui(n))
    {
        pair p = equal_range(primes, 0, PRIMES_SIZE, z_to_ui(n));

        if (p.first != PRIMES_SIZE && primes[p.first] != z_to_ui(n))
            --p.first;

        if (p.first != p.second && p.second != PRIMES_SIZE)
            return z_from_ui(primes[p.second]);
    }

    z_t nTmp = z_copy(n);
    z_t tmp = z_copy(nTmp);
    z_and_c(&tmp, 2);

    if (z_cmp_c(tmp, 0))
        z_add_c(&nTmp, 2);
    else
        z_add_c(&nTmp, 1);

    while (!z_is_prime(nTmp, 25))
        z_add_c(&nTmp, 2);

    z_free(&tmp);

    return nTmp;
}

z_t z_previous_prime(z_t n)
{
    if (z_is_nan(n))
        return z_copy(n);
    else if (z_is_infinity(n) || z_cmp_c(n, 2) < 0)
        return z_nan();
    else if (!z_cmp_c(n, 2))
        return z_from_c(2);
    else if (!z_cmp_c(n, 3))
        return z_from_c(2);
    else if (z_fits_ui(n))
    {
        pair p = equal_range(primes, 0, PRIMES_SIZE, z_to_ui(n));

        if (p.first != PRIMES_SIZE && primes[p.first] != z_to_ui(n))
            --p.first;

        if (p.first != p.second && p.second != PRIMES_SIZE)
        {
            if (primes[p.first] == z_to_ui(n))
                return z_from_ui(primes[p.first - 1]);
            else
                return z_from_ui(primes[p.first]);
        }
    }

    z_t nTmp = z_copy(n);
    z_t tmp = z_copy(nTmp);
    z_and_c(&tmp, 2);

    if (z_cmp_c(tmp, 0))
        z_sub_c(&nTmp, 2);
    else
        z_sub_c(&nTmp, 1);

    while (!z_is_prime(nTmp, 25))
        z_sub_c(&nTmp, 2);

    z_free(&tmp);

    return nTmp;
}
