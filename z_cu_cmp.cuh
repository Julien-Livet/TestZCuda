#ifndef Z_CU_CMP_CUH
#define Z_CU_CMP_CUH

#include "z_cu_t.cuh"

__global__ void z_cu_cmp_uc(z_cu_t const* lhs, unsigned char rhs, int* cmp, bool* synchro);
__global__ void z_cu_cmp_ui(z_cu_t const* lhs, unsigned int rhs, int* cmp, bool* synchro);
__global__ void z_cu_cmp_ul(z_cu_t const* lhs, unsigned long rhs, int* cmp, bool* synchro);
__global__ void z_cu_cmp_ull(z_cu_t const* lhs, unsigned long long rhs, int* cmp, bool* synchro);
__global__ void z_cu_cmp_us(z_cu_t const* lhs, unsigned short rhs, int* cmp, bool* synchro);
__global__ void z_cu_cmp_c(z_cu_t const* lhs, char rhs, int* cmp, bool* synchro);
__global__ void z_cu_cmp_i(z_cu_t const* lhs, int rhs, int* cmp, bool* synchro);
__global__ void z_cu_cmp_l(z_cu_t const* lhs, long rhs, int* cmp, bool* synchro);
__global__ void z_cu_cmp_ll(z_cu_t const* lhs, long long rhs, int* cmp, bool* synchro);
__global__ void z_cu_cmp_s(z_cu_t const* lhs, short rhs, int* cmp, bool* synchro);
__global__ void z_cu_cmp_z(z_cu_t const* lhs, z_cu_t const* rhs, int* cmp, bool* synchro);

#endif // Z_CU_CMP_CUH
