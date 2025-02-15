#include "z_cmp.h"

#define Z_CMP(suffix, type)             \
int z_cmp_##suffix(z_t lhs, type rhs)   \
{                                       \
    z_t other = z_from_##suffix(rhs);   \
                                        \
    int cmp = z_cmp_z(lhs, other);      \
                                        \
    z_free(&other);                     \
                                        \
    return cmp;                         \
}

Z_CMP(c, char)
Z_CMP(i, int)
Z_CMP(l, long)
Z_CMP(ll, long long)
Z_CMP(s, short)
Z_CMP(uc, unsigned char)
Z_CMP(ui, unsigned int)
Z_CMP(ul, unsigned long)
Z_CMP(ull, unsigned long long)
Z_CMP(us, unsigned short)

int z_cmp_z(z_t lhs, z_t rhs)
{
    if (z_is_nan(lhs) && z_is_nan(rhs))
        return 0;
    else if (z_is_nan(lhs) || z_is_nan(rhs))
        return 2;
    else if (z_is_infinity(rhs))
    {
        if (z_is_infinity(lhs))
        {
            if (z_is_positive(rhs) == z_is_positive(lhs))
                return 0;
            else if (z_is_positive(rhs) && !z_is_positive(lhs))
                return -1;
            else
                return 1;
        }
        else if (z_is_positive(rhs))
            return -1;
        else
            return 1;
    }
    else if (z_is_infinity(lhs))
    {
        if (z_is_infinity(rhs))
        {
            if (z_is_positive(rhs) == z_is_positive(lhs))
                return 0;
            else if (z_is_positive(lhs) && !z_is_positive(rhs))
                return 1;
            else
                return -1;
        }
        else if (z_is_positive(lhs))
            return 1;
        else
            return -1;
    }

    if (lhs.size != rhs.size)
    {
        if (lhs.size > rhs.size)
        {
            for (size_t i = 0; i < lhs.size - rhs.size; ++i)
            {
                if (lhs.bits[i])
                    return z_is_positive(lhs);
            }
        }
        else
        {
            for (size_t i = 0; i < rhs.size - lhs.size; ++i)
            {
                if (rhs.bits[i])
                    return -z_is_positive(rhs);
            }
        }
    }

    size_t i1 = lhs.size - MIN(lhs.size, rhs.size);
    size_t i2 = rhs.size - MIN(lhs.size, rhs.size);

    for (size_t i = 0; i < MIN(lhs.size, rhs.size); ++i)
    {
        if (z_is_positive(lhs) && z_is_positive(rhs))
        {
            if (lhs.bits[i1] > rhs.bits[i2])
                return 1;
            else if (lhs.bits[i1] < rhs.bits[i2])
                return -1;
        }
        else if (!z_is_positive(lhs) && !z_is_positive(rhs))
        {
            if (lhs.bits[i1] > rhs.bits[i2])
                return -1;
            else if (lhs.bits[i1] < rhs.bits[i2])
                return 1;
        }
        else if (z_is_positive(lhs) && !z_is_positive(rhs))
        {
            if (lhs.bits[i1] != rhs.bits[i2])
                return 1;
        }
        else if (!z_is_positive(lhs) && z_is_positive(rhs))
        {
            if (lhs.bits[i1] != rhs.bits[i2])
                return -1;
        }

        ++i1;
        ++i2;
    }

    return 0;
}
