#ifndef Z_CU_OR_CUH
#define Z_CU_OR_CUH

#include "z_cu_t.cuh"

__global__ void z_cu_or_uc(z_cu_t* lhs, unsigned char rhs, bool* synchro);
__global__ void z_cu_or_ui(z_cu_t* lhs, unsigned int rhs, bool* synchro);
__global__ void z_cu_or_ul(z_cu_t* lhs, unsigned long rhs, bool* synchro);
__global__ void z_cu_or_ull(z_cu_t* lhs, unsigned long long rhs, bool* synchro);
__global__ void z_cu_or_us(z_cu_t* lhs, unsigned short rhs, bool* synchro);
__global__ void z_cu_or_c(z_cu_t* lhs, char rhs, bool* synchro);
__global__ void z_cu_or_i(z_cu_t* lhs, int rhs, bool* synchro);
__global__ void z_cu_or_l(z_cu_t* lhs, long rhs, bool* synchro);
__global__ void z_cu_or_ll(z_cu_t* lhs, long long rhs, bool* synchro);
__global__ void z_cu_or_s(z_cu_t* lhs, short rhs, bool* synchro);
__global__ void z_cu_or_z(z_cu_t* lhs, z_cu_t const* rhs, bool* synchro);

#endif // Z_CU_OR_CUH
