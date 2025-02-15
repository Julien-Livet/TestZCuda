#ifndef Z_MUL_H
#define Z_MUL_H

#include "z_t.h"

void z_mul_uc(z_t* lhs, unsigned char rhs);
void z_mul_ui(z_t* lhs, unsigned int rhs);
void z_mul_ul(z_t* lhs, unsigned long rhs);
void z_mul_ull(z_t* lhs, unsigned long long rhs);
void z_mul_us(z_t* lhs, unsigned short rhs);
void z_mul_c(z_t* lhs, char rhs);
void z_mul_i(z_t* lhs, int rhs);
void z_mul_l(z_t* lhs, long rhs);
void z_mul_ll(z_t* lhs, long long rhs);
void z_mul_s(z_t* lhs, short rhs);
void z_mul_z(z_t* lhs, z_t rhs);

#endif // Z_MUL_H
