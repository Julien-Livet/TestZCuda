#ifndef Z_LSHIFT_H
#define Z_LSHIFT_H

#include "z_t.h"

void z_lshift_uc(z_t* lhs, unsigned char rhs);
void z_lshift_ui(z_t* lhs, unsigned int rhs);
void z_lshift_ul(z_t* lhs, unsigned long rhs);
void z_lshift_ull(z_t* lhs, unsigned long long rhs);
void z_lshift_us(z_t* lhs, unsigned short rhs);
void z_lshift_c(z_t* lhs, char rhs);
void z_lshift_i(z_t* lhs, int rhs);
void z_lshift_l(z_t* lhs, long rhs);
void z_lshift_ll(z_t* lhs, long long rhs);
void z_lshift_s(z_t* lhs, short rhs);
void z_lshift_z(z_t* lhs, z_t rhs);

#endif // Z_LSHIFT_H
