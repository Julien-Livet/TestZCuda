#ifndef Z_CU_TO_CUH
#define Z_CU_TO_CUH

#include "z_cu_t.cuh"

__global__ void z_cu_to_c(z_cu_t const* z, char* n, bool* synchro);
__global__ void z_cu_to_i(z_cu_t const* z, int* n, bool* synchro);
__global__ void z_cu_to_l(z_cu_t const* z, long* n, bool* synchro);
__global__ void z_cu_to_ll(z_cu_t const* z, long long* n, bool* synchro);
__global__ void z_cu_to_s(z_cu_t const* z, short* n, bool* synchro);
//__global__ void z_cu_to_str(z_cu_t const* z, size_t base, char** s);
__global__ void z_cu_to_uc(z_cu_t const* z, unsigned char* n, bool* synchro);
__global__ void z_cu_to_ui(z_cu_t const* z, unsigned int* n, bool* synchro);
__global__ void z_cu_to_ul(z_cu_t const* z, unsigned long* n, bool* synchro);
__global__ void z_cu_to_ull(z_cu_t const* z, unsigned long long* n, bool* synchro);
__global__ void z_cu_to_us(z_cu_t const* z, unsigned short* n, bool* synchro);

#endif // Z_CU_TO_CUH
