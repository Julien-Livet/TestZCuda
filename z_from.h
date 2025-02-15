#ifndef Z_FROM_H
#define Z_FROM_H

#include "z_t.h"

z_t z_from_c(char n);
z_t z_from_data(void const* data, size_t size);
z_t z_from_i(int n);
z_t z_from_l(long n);
z_t z_from_ll(long long n);
z_t z_from_s(short n);
z_t z_from_str(char const* n, size_t base);
z_t z_from_uc(unsigned char n);
z_t z_from_ui(unsigned int n);
z_t z_from_ul(unsigned long n);
z_t z_from_ull(unsigned long long n);
z_t z_from_us(unsigned short n);

#endif // Z_FROM_H
