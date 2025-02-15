#ifndef Z_POW_H
#define Z_POW_H

#include "z_t.h"

z_t z_pow_c(z_t base, char exp);
z_t z_pow_i(z_t base, int exp);
z_t z_pow_l(z_t base, long exp);
z_t z_pow_ll(z_t base, long long exp);
z_t z_pow_s(z_t base, short exp);
z_t z_pow_z(z_t base, z_t exp);
z_t z_pow_uc(z_t base, unsigned char exp);
z_t z_pow_ui(z_t base, unsigned int exp);
z_t z_pow_ul(z_t base, unsigned long exp);
z_t z_pow_ull(z_t base, unsigned long long exp);
z_t z_pow_us(z_t base, unsigned short exp);

#endif // Z_POW_H
