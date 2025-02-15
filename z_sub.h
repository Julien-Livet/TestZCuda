#ifndef Z_SUB_H
#define Z_SUB_H

#include "z_t.h"

void z_sub_uc(z_t* lhs, unsigned char rhs);
void z_sub_ui(z_t* lhs, unsigned int rhs);
void z_sub_ul(z_t* lhs, unsigned long rhs);
void z_sub_ull(z_t* lhs, unsigned long long rhs);
void z_sub_us(z_t* lhs, unsigned short rhs);
void z_sub_c(z_t* lhs, char rhs);
void z_sub_i(z_t* lhs, int rhs);
void z_sub_l(z_t* lhs, long rhs);
void z_sub_ll(z_t* lhs, long long rhs);
void z_sub_s(z_t* lhs, short rhs);
void z_sub_z(z_t* lhs, z_t rhs);

#endif // Z_SUB_H
