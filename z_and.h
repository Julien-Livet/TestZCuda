#ifndef Z_AND_H
#define Z_AND_H

#include "z_t.h"

void z_and_uc(z_t* lhs, unsigned char rhs);
void z_and_ui(z_t* lhs, unsigned int rhs);
void z_and_ul(z_t* lhs, unsigned long rhs);
void z_and_ull(z_t* lhs, unsigned long long rhs);
void z_and_us(z_t* lhs, unsigned short rhs);
void z_and_c(z_t* lhs, char rhs);
void z_and_i(z_t* lhs, int rhs);
void z_and_l(z_t* lhs, long rhs);
void z_and_ll(z_t* lhs, long long rhs);
void z_and_s(z_t* lhs, short rhs);
void z_and_z(z_t* lhs, z_t rhs);

#endif // Z_AND_H
