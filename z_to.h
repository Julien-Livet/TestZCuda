#ifndef Z_TO_H
#define Z_TO_H

#include "z_t.h"

char z_to_c(z_t z);
int z_to_i(z_t z);
long z_to_l(z_t z);
long long z_to_ll(z_t z);
short z_to_s(z_t z);
char* z_to_str(z_t z, size_t base);
unsigned char z_to_uc(z_t z);
unsigned int z_to_ui(z_t z);
unsigned long z_to_ul(z_t z);
unsigned long long z_to_ull(z_t z);
unsigned short z_to_us(z_t z);

#endif // Z_TO_H
