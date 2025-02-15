#ifndef Z_CU_DIV_R_CUH
#define Z_CU_DIV_R_CUH

#include "z_cu_t.cuh"

__global__ void z_cu_div_r_uc(z_cu_t const* lhs, unsigned char rhs, z_cu_t* r, bool* synchro);
__global__ void z_cu_div_r_ui(z_cu_t const* lhs, unsigned int rhs, z_cu_t* r, bool* synchro);
__global__ void z_cu_div_r_ul(z_cu_t const* lhs, unsigned long rhs, z_cu_t* r, bool* synchro);
__global__ void z_cu_div_r_ull(z_cu_t const* lhs, unsigned long long rhs, z_cu_t* r, bool* synchro);
__global__ void z_cu_div_r_us(z_cu_t const* lhs, unsigned short rhs, z_cu_t* r, bool* synchro);
__global__ void z_cu_div_r_c(z_cu_t const* lhs, char rhs, z_cu_t* r, bool* synchro);
__global__ void z_cu_div_r_i(z_cu_t const* lhs, int rhs, z_cu_t* r, bool* synchro);
__global__ void z_cu_div_r_l(z_cu_t const* lhs, long rhs, z_cu_t* r, bool* synchro);
__global__ void z_cu_div_r_ll(z_cu_t const* lhs, long long rhs, z_cu_t* r, bool* synchro);
__global__ void z_cu_div_r_s(z_cu_t const* lhs, short rhs, z_cu_t* r, bool* synchro);
__global__ void z_cu_div_r_z(z_cu_t const* lhs, z_cu_t const* rhs, z_cu_t* r, bool* synchro);

#endif // Z_CU_DIV_R_CUH
