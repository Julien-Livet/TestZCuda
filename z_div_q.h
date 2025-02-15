#ifndef Z_DIV_Q_H
#define Z_DIV_Q_H

#include "z_t.h"

z_t z_div_q_uc(z_t lhs, unsigned char rhs);
z_t z_div_q_ui(z_t lhs, unsigned int rhs);
z_t z_div_q_ul(z_t lhs, unsigned long rhs);
z_t z_div_q_ull(z_t lhs, unsigned long long rhs);
z_t z_div_q_us(z_t lhs, unsigned short rhs);
z_t z_div_q_c(z_t lhs, char rhs);
z_t z_div_q_i(z_t lhs, int rhs);
z_t z_div_q_l(z_t lhs, long rhs);
z_t z_div_q_ll(z_t lhs, long long rhs);
z_t z_div_q_s(z_t lhs, short rhs);
z_t z_div_q_z(z_t lhs, z_t rhs);

#endif // Z_DIV_Q_H
