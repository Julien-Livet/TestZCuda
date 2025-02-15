#ifndef Z_POWM_H
#define Z_POWM_H

#include "z_t.h"

z_t z_powm_c(z_t base, char exp, char mod);
z_t z_powm_i(z_t base, int exp, int mod);
z_t z_powm_l(z_t base, long exp, long mod);
z_t z_powm_ll(z_t base, long long exp, long long mod);
z_t z_powm_s(z_t base, short exp, short mod);
z_t z_powm_z(z_t base, z_t exp, z_t mod);
z_t z_powm_uc(z_t base, unsigned char exp, unsigned char mod);
z_t z_powm_ui(z_t base, unsigned int exp, unsigned int mod);
z_t z_powm_ul(z_t base, unsigned long exp, unsigned long mod);
z_t z_powm_ull(z_t base, unsigned long long exp, unsigned long long mod);
z_t z_powm_us(z_t base, unsigned short exp, unsigned short mod);

#endif // Z_POWM_H
