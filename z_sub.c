#include <assert.h>

#include "z_sub.h"

#define Z_SUB(suffix, type)             \
void z_sub_##suffix(z_t* lhs, type rhs) \
{                                       \
    assert(lhs);                        \
                                        \
    z_t other = z_from_##suffix(rhs);   \
                                        \
    z_sub_z(lhs, other);                \
                                        \
    z_free(&other);                     \
}

Z_SUB(c, char)
Z_SUB(i, int)
Z_SUB(l, long)
Z_SUB(ll, long long)
Z_SUB(s, short)
Z_SUB(uc, unsigned char)
Z_SUB(ui, unsigned int)
Z_SUB(ul, unsigned long)
Z_SUB(ull, unsigned long long)
Z_SUB(us, unsigned short)

void z_sub_z(z_t* lhs, z_t rhs)
{
    assert(lhs);

    rhs.is_positive = !rhs.is_positive;

    z_add_z(lhs, rhs);
}
