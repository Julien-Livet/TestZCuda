#include "z_fits.h"

#define Z_FITS(suffix, type)        \
bool z_fits_##suffix(z_t z)         \
{                                   \
    type n = z_to_##suffix(z);      \
                                    \
    return !z_cmp_##suffix(z, n);   \
}

Z_FITS(c, char)
Z_FITS(i, int)
Z_FITS(l, long)
Z_FITS(ll, long long)
Z_FITS(s, short)
Z_FITS(uc, unsigned char)
Z_FITS(ui, unsigned int)
Z_FITS(ul, unsigned long)
Z_FITS(ull, unsigned long long)
Z_FITS(us, unsigned short)
