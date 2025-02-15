#ifndef Z_DIV_R_H
#define Z_DIV_R_H

#include "z_t.h"

z_t z_div_r_uc(z_t lhs, unsigned char rhs);
z_t z_div_r_ui(z_t lhs, unsigned int rhs);
z_t z_div_r_ul(z_t lhs, unsigned long rhs);
z_t z_div_r_ull(z_t lhs, unsigned long long rhs);
z_t z_div_r_us(z_t lhs, unsigned short rhs);
z_t z_div_r_c(z_t lhs, char rhs);
z_t z_div_r_i(z_t lhs, int rhs);
z_t z_div_r_l(z_t lhs, long rhs);
z_t z_div_r_ll(z_t lhs, long long rhs);
z_t z_div_r_s(z_t lhs, short rhs);
z_t z_div_r_z(z_t lhs, z_t rhs);

#endif // Z_DIV_R_H
