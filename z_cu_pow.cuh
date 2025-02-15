#ifndef Z_CU_POW_CUH
#define Z_CU_POW_CUH

#include "z_cu_t.cuh"

__global__ void z_cu_pow_c(z_cu_t const* base, char exp, z_cu_t* p, bool* synchro);
__global__ void z_cu_pow_i(z_cu_t const* base, int exp, z_cu_t* p, bool* synchro);
__global__ void z_cu_pow_l(z_cu_t const* base, long exp, z_cu_t* p, bool* synchro);
__global__ void z_cu_pow_ll(z_cu_t const* base, long long exp, z_cu_t* p, bool* synchro);
__global__ void z_cu_pow_s(z_cu_t const* base, short exp, z_cu_t* p, bool* synchro);
__global__ void z_cu_pow_z(z_cu_t const* base, z_cu_t const* exp, z_cu_t* p, bool* synchro);
__global__ void z_cu_pow_uc(z_cu_t const* base, unsigned char exp, z_cu_t* p, bool* synchro);
__global__ void z_cu_pow_ui(z_cu_t const* base, unsigned int exp, z_cu_t* p, bool* synchro);
__global__ void z_cu_pow_ul(z_cu_t const* base, unsigned long exp, z_cu_t* p, bool* synchro);
__global__ void z_cu_pow_ull(z_cu_t const* base, unsigned long long exp, z_cu_t* p, bool* synchro);
__global__ void z_cu_pow_us(z_cu_t const* base, unsigned short exp, z_cu_t* p, bool* synchro);

#endif // Z_CU_POW_CUH
