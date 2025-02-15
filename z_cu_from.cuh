#ifndef Z_CU_FROM_CUH
#define Z_CU_FROM_CUH

#include "z_cu_t.cuh"

extern "C"
{
#include "z_t.h"
}

__global__ void z_cu_from_c(z_cu_t* z, char n, bool* synchro);
__global__ void z_cu_from_data(z_cu_t* z, void const* data, size_t size, bool* synchro);
__global__ void z_cu_from_i(z_cu_t* z, int n, bool* synchro);
__global__ void z_cu_from_l(z_cu_t* z, long n, bool* synchro);
__global__ void z_cu_from_ll(z_cu_t* z, long long n, bool* synchro);
__global__ void z_cu_from_s(z_cu_t* z, short n, bool* synchro);
__global__ void z_cu_from_uc(z_cu_t* z, unsigned char n, bool* synchro);
__global__ void z_cu_from_ui(z_cu_t* z, unsigned int n, bool* synchro);
__global__ void z_cu_from_ul(z_cu_t* z, unsigned long n, bool* synchro);
__global__ void z_cu_from_ull(z_cu_t* z, unsigned long long n, bool* synchro);
__global__ void z_cu_from_us(z_cu_t* z, unsigned short n, bool* synchro);
void z_cu_from_z(z_t const* from, z_cu_t* to);
void z_from_z_cu(z_cu_t const* from, z_t* to);

#endif // Z_CU_FROM_CUH
