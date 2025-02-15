#ifndef Z_CU_MUL_CUH
#define Z_CU_MUL_CUH

#include "z_cu_t.cuh"

__global__ void z_cu_mul_uc(z_cu_t* lhs, unsigned char rhs, bool* synchro);
__global__ void z_cu_mul_ui(z_cu_t* lhs, unsigned int rhs, bool* synchro);
__global__ void z_cu_mul_ul(z_cu_t* lhs, unsigned long rhs, bool* synchro);
__global__ void z_cu_mul_ull(z_cu_t* lhs, unsigned long long rhs, bool* synchro);
__global__ void z_cu_mul_us(z_cu_t* lhs, unsigned short rhs, bool* synchro);
__global__ void z_cu_mul_c(z_cu_t* lhs, char rhs, bool* synchro);
__global__ void z_cu_mul_i(z_cu_t* lhs, int rhs, bool* synchro);
__global__ void z_cu_mul_l(z_cu_t* lhs, long rhs, bool* synchro);
__global__ void z_cu_mul_ll(z_cu_t* lhs, long long rhs, bool* synchro);
__global__ void z_cu_mul_s(z_cu_t* lhs, short rhs, bool* synchro);
__global__ void z_cu_mul_z(z_cu_t* lhs, z_cu_t const* rhs, bool* synchro);

#endif // Z_CU_MUL_CUH
