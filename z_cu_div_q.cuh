#ifndef Z_CU_DIV_Q_CUH
#define Z_CU_DIV_Q_CUH

#include "z_cu_t.cuh"

__global__ void z_cu_div_q_uc(z_cu_t const* lhs, unsigned char rhs, z_cu_t* q, bool* synchro);
__global__ void z_cu_div_q_ui(z_cu_t const* lhs, unsigned int rhs, z_cu_t* q, bool* synchro);
__global__ void z_cu_div_q_ul(z_cu_t const* lhs, unsigned long rhs, z_cu_t* q, bool* synchro);
__global__ void z_cu_div_q_ull(z_cu_t const* lhs, unsigned long long rhs, z_cu_t* q, bool* synchro);
__global__ void z_cu_div_q_us(z_cu_t const* lhs, unsigned short rhs, z_cu_t* q, bool* synchro);
__global__ void z_cu_div_q_c(z_cu_t const* lhs, char rhs, z_cu_t* q, bool* synchro);
__global__ void z_cu_div_q_i(z_cu_t const* lhs, int rhs, z_cu_t* q, bool* synchro);
__global__ void z_cu_div_q_l(z_cu_t const* lhs, long rhs, z_cu_t* q, bool* synchro);
__global__ void z_cu_div_q_ll(z_cu_t const* lhs, long long rhs, z_cu_t* q, bool* synchro);
__global__ void z_cu_div_q_s(z_cu_t const* lhs, short rhs, z_cu_t* q, bool* synchro);
__global__ void z_cu_div_q_z(z_cu_t const* lhs, z_cu_t const* rhs, z_cu_t* q, bool* synchro);

#endif // Z_CU_DIV_Q_CUH
