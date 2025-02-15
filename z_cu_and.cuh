#ifndef Z_CU_AND_CUH
#define Z_CU_AND_CUH

#include "z_cu_t.cuh"

__global__ void z_cu_and_uc(z_cu_t* lhs, unsigned char rhs, bool* synchro);
__global__ void z_cu_and_ui(z_cu_t* lhs, unsigned int rhs, bool* synchro);
__global__ void z_cu_and_ul(z_cu_t* lhs, unsigned long rhs, bool* synchro);
__global__ void z_cu_and_ull(z_cu_t* lhs, unsigned long long rhs, bool* synchro);
__global__ void z_cu_and_us(z_cu_t* lhs, unsigned short rhs, bool* synchro);
__global__ void z_cu_and_c(z_cu_t* lhs, char rhs, bool* synchro);
__global__ void z_cu_and_i(z_cu_t* lhs, int rhs, bool* synchro);
__global__ void z_cu_and_l(z_cu_t* lhs, long rhs, bool* synchro);
__global__ void z_cu_and_ll(z_cu_t* lhs, long long rhs, bool* synchro);
__global__ void z_cu_and_s(z_cu_t* lhs, short rhs, bool* synchro);
__global__ void z_cu_and_z(z_cu_t* lhs, z_cu_t const* rhs, bool* synchro);

#endif // Z_CU_AND_CUH
