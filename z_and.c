#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "z_and.h"

#define Z_AND(suffix, type)             \
void z_and_##suffix(z_t* lhs, type rhs) \
{                                       \
    assert(lhs);                        \
                                        \
    z_t other = z_from_##suffix(rhs);   \
                                        \
    z_and_z(lhs, other);                \
                                        \
    z_free(&other);                     \
}

Z_AND(c, char)
Z_AND(i, int)
Z_AND(l, long)
Z_AND(ll, long long)
Z_AND(s, short)
Z_AND(uc, unsigned char)
Z_AND(ui, unsigned int)
Z_AND(ul, unsigned long)
Z_AND(ull, unsigned long long)
Z_AND(us, unsigned short)

void z_and_z(z_t* lhs, z_t rhs)
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
        memset(lhs->bits, 0, (lhs->size - rhs.size) * sizeof(z_type));

    for (size_t i = 0; i < MIN(lhs->size, rhs.size); ++i)
        lhs->bits[lhs->size - 1 - i] &= rhs.bits[rhs.size - 1 - i];

    if (z_is_auto_adjust(*lhs))
        z_adjust(lhs);
}
