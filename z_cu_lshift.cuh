#ifndef Z_CU_LSHIFT_CUH
#define Z_CU_LSHIFT_CUH

#include "z_cu_t.cuh"

__global__ void z_cu_lshift_uc(z_cu_t* lhs, unsigned char rhs, bool* synchro);
__global__ void z_cu_lshift_ui(z_cu_t* lhs, unsigned int rhs, bool* synchro);
__global__ void z_cu_lshift_ul(z_cu_t* lhs, unsigned long rhs, bool* synchro);
__global__ void z_cu_lshift_ull(z_cu_t* lhs, unsigned long long rhs, bool* synchro);
__global__ void z_cu_lshift_us(z_cu_t* lhs, unsigned short rhs, bool* synchro);
__global__ void z_cu_lshift_c(z_cu_t* lhs, char rhs, bool* synchro);
__global__ void z_cu_lshift_i(z_cu_t* lhs, int rhs, bool* synchro);
__global__ void z_cu_lshift_l(z_cu_t* lhs, long rhs, bool* synchro);
__global__ void z_cu_lshift_ll(z_cu_t* lhs, long long rhs, bool* synchro);
__global__ void z_cu_lshift_s(z_cu_t* lhs, short rhs, bool* synchro);
__global__ void z_cu_lshift_z(z_cu_t* lhs, z_cu_t const* rhs, bool* synchro);

#endif // Z_CU_LSHIFT_CUH
