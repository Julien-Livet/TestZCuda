#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "z_xor.h"

#define Z_XOR(suffix, type)             \
void z_xor_##suffix(z_t* lhs, type rhs) \
{                                       \
    assert(lhs);                        \
                                        \
    z_t other = z_from_##suffix(rhs);   \
                                        \
    z_xor_z(lhs, other);                \
                                        \
    z_free(&other);                     \
}

Z_XOR(c, char)
Z_XOR(i, int)
Z_XOR(l, long)
Z_XOR(ll, long long)
Z_XOR(s, short)
Z_XOR(uc, unsigned char)
Z_XOR(ui, unsigned int)
Z_XOR(ul, unsigned long)
Z_XOR(ull, unsigned long long)
Z_XOR(us, unsigned short)

void z_xor_z(z_t* lhs, z_t rhs)
{
    assert(lhs);

    if (lhs->size < rhs.size)
    {
        z_type* bits = lhs->bits;
        lhs->bits = malloc(rhs.size * sizeof(z_type));
        memset(lhs->bits, 0, (rhs.size - lhs->size) * sizeof(z_type));
        memcpy((void*)(lhs->bits) + (rhs.size - lhs->size) * sizeof(z_type), bits, lhs->size * sizeof(z_type));
        lhs->size = rhs.size;
        free(bits);
    }
    else if (lhs->size > rhs.size)
    {
        for (size_t i = 0; i < lhs->size - rhs.size; ++i)
            lhs->bits[i] ^= 0;
    }

    for (size_t i = 0; i < MIN(lhs->size, rhs.size); ++i)
        lhs->bits[lhs->size - 1 - i] ^= rhs.bits[rhs.size - 1 - i];

    if (z_is_auto_adjust(*lhs))
        z_adjust(lhs);
}
