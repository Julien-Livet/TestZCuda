#ifndef Z_CMP_H
#define Z_CMP_H

#include "z_t.h"

int z_cmp_uc(z_t lhs, unsigned char rhs);
int z_cmp_ui(z_t lhs, unsigned int rhs);
int z_cmp_ul(z_t lhs, unsigned long rhs);
int z_cmp_ull(z_t lhs, unsigned long long rhs);
int z_cmp_us(z_t lhs, unsigned short rhs);
int z_cmp_c(z_t lhs, char rhs);
int z_cmp_i(z_t lhs, int rhs);
int z_cmp_l(z_t lhs, long rhs);
int z_cmp_ll(z_t lhs, long long rhs);
int z_cmp_s(z_t lhs, short rhs);
int z_cmp_z(z_t lhs, z_t rhs);

#endif // Z_CMP_H
