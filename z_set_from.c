#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "z_set_from.h"

#define Z_SET_FROM(suffix, type)                                    \
void z_set_from_##suffix(z_t* z, type n)                            \
{                                                                   \
    assert(z);                                                      \
                                                                    \
    z->is_positive = (n >= 0);                                      \
                                                                    \
    size_t size = sizeof(type) / sizeof(z_type);                    \
                                                                    \
    if (sizeof(type) % sizeof(z_type))                              \
        ++size;                                                     \
                                                                    \
    if (size > z->size)                                             \
    {                                                               \
        z->size = size;                                             \
        free(z->bits);                                              \
        z->bits = malloc(z->size * sizeof(z_type));                 \
    }                                                               \
                                                                    \
    z->is_nan = false;                                              \
    z->is_infinity = false;                                         \
                                                                    \
    if (n < 0)                                                      \
        n = -n;                                                     \
                                                                    \
    assert(z->bits);                                                \
                                                                    \
    size_t const s = sizeof(type);                                  \
                                                                    \
    memset((void*)(z->bits) + s, 0, z->size * sizeof(z_type) - s);  \
    memcpy(z->bits, &n, s);                                         \
}

Z_SET_FROM(c, char)
Z_SET_FROM(i, int)
Z_SET_FROM(l, long)
Z_SET_FROM(ll, long long)
Z_SET_FROM(s, short)
Z_SET_FROM(uc, unsigned char)
Z_SET_FROM(ui, unsigned int)
Z_SET_FROM(ul, unsigned long)
Z_SET_FROM(ull, unsigned long long)
Z_SET_FROM(us, unsigned short)

void z_set_from_data(z_t* z, void const* data, size_t size)
{
    assert(z);

    assert(data);

    z->is_positive = true;

    size_t s = size / sizeof(z_type);

    if (size % sizeof(z_type))
        ++s;

    if (s > z->size)
    {
        z->size = s;
        free(z->bits);
        z->bits = malloc(z->size * sizeof(z_type));
    }

    z->is_nan = false;
    z->is_infinity = false;

    if (size)
        assert(z->bits);

    memset((void*)(z->bits) + size, 0, z->size * sizeof(z_type) - size);
    memcpy(z->bits, data, size);

    for (size_t i = 0; i < size / 2; ++i)
    {
        char tmp = *((char*)(z->bits) + size - 1 - i);
        *((char*)(z->bits) + size - 1 - i) = *((char*)(z->bits) + i);
        *((char*)(z->bits) + i) = tmp;
    }
}

void z_set_from_str(z_t* z, char const* n, size_t base)
{
    assert(z);

    z_free(z);

    *z = z_from_str(n, base);
}

void z_set_from_z(z_t* z, z_t n)
{
    assert(z);

    z->is_positive = n.is_positive;
    free(z->bits);
    z->bits = malloc(n.size * sizeof(z_type));
    memcpy(z->bits, n.bits, n.size * sizeof(z_type));
    z->size = n.size;
    z->is_nan = n.is_nan;
    z->is_infinity = n.is_infinity;
    z->is_auto_adjust = n.is_auto_adjust;
}
