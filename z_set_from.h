#ifndef Z_SET_FROM_H
#define Z_SET_FROM_H

#include "z_t.h"

void z_set_from_c(z_t* z, char n);
void z_set_from_data(z_t* z, void const* data, size_t size);
void z_set_from_i(z_t* z, int n);
void z_set_from_l(z_t* z, long n);
void z_set_from_ll(z_t* z, long long n);
void z_set_from_s(z_t* z, short n);
void z_set_from_str(z_t* z, char const* n, size_t base);
void z_set_from_uc(z_t* z, unsigned char n);
void z_set_from_ui(z_t* z, unsigned int n);
void z_set_from_ul(z_t* z, unsigned long n);
void z_set_from_ull(z_t* z, unsigned long long n);
void z_set_from_us(z_t* z, unsigned short n);
void z_set_from_z(z_t* z, z_t n);

#endif // Z_SET_FROM_H
