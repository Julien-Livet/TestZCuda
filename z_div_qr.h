#ifndef Z_DIV_QR_H
#define Z_DIV_QR_H

#include "z_t.h"

void z_div_qr_uc(z_t lhs, unsigned char rhs, z_t* q, z_t* r);
void z_div_qr_ui(z_t lhs, unsigned int rhs, z_t* q, z_t* r);
void z_div_qr_ul(z_t lhs, unsigned long rhs, z_t* q, z_t* r);
void z_div_qr_ull(z_t lhs, unsigned long long rhs, z_t* q, z_t* r);
void z_div_qr_us(z_t lhs, unsigned short rhs, z_t* q, z_t* r);
void z_div_qr_c(z_t lhs, char rhs, z_t* q, z_t* r);
void z_div_qr_i(z_t lhs, int rhs, z_t* q, z_t* r);
void z_div_qr_l(z_t lhs, long rhs, z_t* q, z_t* r);
void z_div_qr_ll(z_t lhs, long long rhs, z_t* q, z_t* r);
void z_div_qr_s(z_t lhs, short rhs, z_t* q, z_t* r);
void z_div_qr_z(z_t lhs, z_t rhs, z_t* q, z_t* r);

#endif // Z_DIV_QR_H
