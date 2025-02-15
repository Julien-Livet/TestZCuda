#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "z_or.h"

#define Z_OR(suffix, type)              \
void z_or_##suffix(z_t* lhs, type rhs)  \
{                                       \
    assert(lhs);                        \
                                        \
    z_t other = z_from_##suffix(rhs);   \
                                        \
    z_or_z(lhs, other);                 \
                                        \
    z_free(&other);                     \
}

Z_OR(c, char)
Z_OR(i, int)
Z_OR(l, long)
Z_OR(ll, long long)
Z_OR(s, short)
Z_OR(uc, unsigned char)
Z_OR(ui, unsigned int)
Z_OR(ul, unsigned long)
Z_OR(ull, unsigned long long)
Z_OR(us, unsigned short)

void z_or_z(z_t* lhs, z_t rhs)
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

    for (size_t i = 0; i < MIN(lhs->size, rhs.size); ++i)
        lhs->bits[lhs->size - 1 - i] |= rhs.bits[rhs.size - 1 - i];

    if (z_is_auto_adjust(*lhs))
        z_adjust(lhs);
}
