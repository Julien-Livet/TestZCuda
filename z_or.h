#ifndef Z_OR_H
#define Z_OR_H

#include "z_t.h"

void z_or_uc(z_t* lhs, unsigned char rhs);
void z_or_ui(z_t* lhs, unsigned int rhs);
void z_or_ul(z_t* lhs, unsigned long rhs);
void z_or_ull(z_t* lhs, unsigned long long rhs);
void z_or_us(z_t* lhs, unsigned short rhs);
void z_or_c(z_t* lhs, char rhs);
void z_or_i(z_t* lhs, int rhs);
void z_or_l(z_t* lhs, long rhs);
void z_or_ll(z_t* lhs, long long rhs);
void z_or_s(z_t* lhs, short rhs);
void z_or_z(z_t* lhs, z_t rhs);

#endif // Z_OR_H
