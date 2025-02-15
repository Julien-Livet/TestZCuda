#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "z_add.h"

#define Z_ADD(suffix, type)             \
void z_add_##suffix(z_t* lhs, type rhs) \
{                                       \
    assert(lhs);                        \
                                        \
    z_t other = z_from_##suffix(rhs);   \
                                        \
    z_add_z(lhs, other);                \
                                        \
    z_free(&other);                     \
}

Z_ADD(c, char)
Z_ADD(i, int)
Z_ADD(l, long)
Z_ADD(ll, long long)
Z_ADD(s, short)
Z_ADD(uc, unsigned char)
Z_ADD(ui, unsigned int)
Z_ADD(ul, unsigned long)
Z_ADD(ull, unsigned long long)
Z_ADD(us, unsigned short)

void z_add_z(z_t* lhs, z_t rhs)
{
    assert(lhs);

    if (z_is_nan(*lhs) || z_is_nan(rhs))
        z_set_nan(lhs);
    else if (z_is_infinity(*lhs) || z_is_infinity(rhs))
    {
        if ((z_is_positive(*lhs) && z_is_negative(rhs))
            || (z_is_negative(*lhs) && z_is_positive(rhs)))
            z_set_nan(lhs);
        else
        {
            if (z_is_infinity(rhs))
                lhs->is_positive = rhs.is_positive;

            z_set_infinity(lhs);
        }
    }
    else
    {
        if ((z_is_positive(*lhs) && z_is_positive(rhs))
            || (z_is_negative(*lhs) && z_is_negative(rhs)))
        {
            z_type carry = 0;

            z_type const* a = lhs->bits;
            z_type const* b = rhs.bits;

            assert(a && b);

            size_t n = MAX(lhs->size, rhs.size);

            z_type* result = malloc(n * sizeof(z_type));

            assert(result);

            for (size_t i = 0; i < n; ++i)
            {
                z_type const bit_a = (i < lhs->size) ? a[lhs->size - 1 - i] : 0;
                z_type const bit_b = (i < rhs.size) ? b[rhs.size - 1 - i] : 0;
                z_type const sum = bit_a + bit_b + carry;

                carry = (sum < bit_a || sum < bit_b);

                result[i] = sum;
            }

            if (carry)
            {
                z_type* r = malloc((n + 1) * sizeof(z_type));
                memcpy(r, result, n * sizeof(z_type));
                r[n] = 1;
                free(result);
                result = r;
                ++n;
            }

            for (size_t i = 0; i < n / 2; ++i)
            {
                z_type tmp = result[i];
                result[i] = result[n - 1 - i];
                result[n - 1 - i] = tmp;
            }

            free(lhs->bits);
            lhs->bits = result;
            lhs->size = n;
        }
        else
        {
            z_type* other_bits = malloc(rhs.size * sizeof(z_type));
            memcpy(other_bits, rhs.bits, rhs.size * sizeof(z_type));
            z_type other_size = rhs.size;

            if (z_is_positive(*lhs))
            {
                z_t neg = z_copy(rhs);
                z_neg(&neg);

                if (z_cmp_z(*lhs, neg) < 0)
                {
                    lhs->is_positive = false;
                    free(other_bits);
                    other_bits = malloc(lhs->size * sizeof(z_type));
                    memcpy(other_bits, lhs->bits, lhs->size * sizeof(z_type));
                    other_size = lhs->size;
                    free(lhs->bits);
                    lhs->bits = malloc(rhs.size * sizeof(z_type));
                    memcpy(lhs->bits, rhs.bits, rhs.size * sizeof(z_type));
                    lhs->size = rhs.size;
                }

                z_free(&neg);
            }
            else
            {
                z_t neg = z_copy(rhs);
                z_neg(&neg);

                if (z_cmp_z(*lhs, neg) > 0)
                {
                    lhs->is_positive = true;
                    free(other_bits);
                    other_bits = malloc(lhs->size * sizeof(z_type));
                    memcpy(other_bits, lhs->bits, lhs->size * sizeof(z_type));
                    other_size = lhs->size;
                    free(lhs->bits);
                    lhs->bits = malloc(rhs.size * sizeof(z_type));
                    memcpy(lhs->bits, rhs.bits, rhs.size * sizeof(z_type));
                    lhs->size = rhs.size;
                }

                z_free(&neg);
            }

            z_type const* a = lhs->bits;
            z_type const* b = other_bits;

            assert(a && b);

            size_t n = MAX(lhs->size, other_size);

            z_type* result = malloc(n * sizeof(z_type));

            assert(result);

            for (size_t i = n - 1; i <= n - 1; --i)
            {
                size_t const ia = lhs->size - 1 - i;
                size_t const ib = rhs.size - 1 - i;

                z_type const bit_a = ia < lhs->size ? a[ia] : 0;
                z_type const bit_b = ib < rhs.size ? b[ib] : 0;

                z_type bit_result = bit_a - bit_b;

                if (bit_a < bit_b)
                {
                    for (size_t j = i - 1; j <= i - 1; --j)
                    {
                        bool const stop = (result[j] > 0);

                        result[j] -= 1;

                        if (stop)
                            break;
                    }

                    bit_result = -1;
                    bit_result -= bit_b;
                    bit_result += bit_a + 1;
                }

                result[i] = bit_result;
            }

            free(lhs->bits);
            lhs->bits = result;
            lhs->size = n;
            free(other_bits);
        }
    }

    if (z_is_auto_adjust(*lhs))
        z_adjust(lhs);
}
