#ifndef Z_XOR_H
#define Z_XOR_H

#include "z_t.h"

void z_xor_uc(z_t* lhs, unsigned char rhs);
void z_xor_ui(z_t* lhs, unsigned int rhs);
void z_xor_ul(z_t* lhs, unsigned long rhs);
void z_xor_ull(z_t* lhs, unsigned long long rhs);
void z_xor_us(z_t* lhs, unsigned short rhs);
void z_xor_c(z_t* lhs, char rhs);
void z_xor_i(z_t* lhs, int rhs);
void z_xor_l(z_t* lhs, long rhs);
void z_xor_ll(z_t* lhs, long long rhs);
void z_xor_s(z_t* lhs, short rhs);
void z_xor_z(z_t* lhs, z_t rhs);

#endif // Z_XOR_H
