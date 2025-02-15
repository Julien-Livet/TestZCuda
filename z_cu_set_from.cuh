#ifndef Z_CU_SET_FROM_CUH
#define Z_CU_SET_FROM_CUH

#include "z_cu_t.cuh"

__global__ void z_cu_set_from_c(z_cu_t* z, char n, bool* synchro);
__global__ void z_cu_set_from_data(z_cu_t* z, void const* data, size_t size, bool* synchro);
__global__ void z_cu_set_from_i(z_cu_t* z, int n, bool* synchro);
__global__ void z_cu_set_from_l(z_cu_t* z, long n, bool* synchro);
__global__ void z_cu_set_from_ll(z_cu_t* z, long long n, bool* synchro);
__global__ void z_cu_set_from_s(z_cu_t* z, short n, bool* synchro);
__global__ void z_cu_set_from_uc(z_cu_t* z, unsigned char n, bool* synchro);
__global__ void z_cu_set_from_ui(z_cu_t* z, unsigned int n, bool* synchro);
__global__ void z_cu_set_from_ul(z_cu_t* z, unsigned long n, bool* synchro);
__global__ void z_cu_set_from_ull(z_cu_t* z, unsigned long long n, bool* synchro);
__global__ void z_cu_set_from_us(z_cu_t* z, unsigned short n, bool* synchro);
__global__ void z_cu_set_from_z(z_cu_t* z, z_cu_t const* n, bool* synchro);

#endif // Z_CU_SET_FROM_CUH
