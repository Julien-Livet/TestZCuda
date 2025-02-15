#ifndef Z_RSHIFT_H
#define Z_RSHIFT_H

#include "z_t.h"

void z_rshift_uc(z_t* lhs, unsigned char rhs);
void z_rshift_ui(z_t* lhs, unsigned int rhs);
void z_rshift_ul(z_t* lhs, unsigned long rhs);
void z_rshift_ull(z_t* lhs, unsigned long long rhs);
void z_rshift_us(z_t* lhs, unsigned short rhs);
void z_rshift_c(z_t* lhs, char rhs);
void z_rshift_i(z_t* lhs, int rhs);
void z_rshift_l(z_t* lhs, long rhs);
void z_rshift_ll(z_t* lhs, long long rhs);
void z_rshift_s(z_t* lhs, short rhs);
void z_rshift_z(z_t* lhs, z_t rhs);

#endif // Z_RSHIFT_H
