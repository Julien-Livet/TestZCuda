#ifndef Z_ADD_H
#define Z_ADD_H

#include "z_t.h"

void z_add_uc(z_t* lhs, unsigned char rhs);
void z_add_ui(z_t* lhs, unsigned int rhs);
void z_add_ul(z_t* lhs, unsigned long rhs);
void z_add_ull(z_t* lhs, unsigned long long rhs);
void z_add_us(z_t* lhs, unsigned short rhs);
void z_add_c(z_t* lhs, char rhs);
void z_add_i(z_t* lhs, int rhs);
void z_add_l(z_t* lhs, long rhs);
void z_add_ll(z_t* lhs, long long rhs);
void z_add_s(z_t* lhs, short rhs);
void z_add_z(z_t* lhs, z_t rhs);

#endif // Z_ADD_H
