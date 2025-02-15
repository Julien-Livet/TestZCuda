#ifndef Z_CU_POWM_CUH
#define Z_CU_POWM_CUH

#include "z_cu_t.cuh"

__global__ void z_cu_powm_c(z_cu_t const* base, char exp, char mod, z_cu_t* p, bool* synchro);
__global__ void z_cu_powm_i(z_cu_t const* base, int exp, int mod, z_cu_t* p, bool* synchro);
__global__ void z_cu_powm_l(z_cu_t const* base, long exp, long mod, z_cu_t* p, bool* synchro);
__global__ void z_cu_powm_ll(z_cu_t const* base, long long exp, long long mod, z_cu_t* p, bool* synchro);
__global__ void z_cu_powm_s(z_cu_t const* base, short exp, short mod, z_cu_t* p, bool* synchro);
__global__ void z_cu_powm_z(z_cu_t const* base, z_cu_t const* exp, z_cu_t const* mod, z_cu_t* p, bool* synchro);
__global__ void z_cu_powm_uc(z_cu_t const* base, unsigned char exp, unsigned char mod, z_cu_t* p, bool* synchro);
__global__ void z_cu_powm_ui(z_cu_t const* base, unsigned int exp, unsigned int mod, z_cu_t* p, bool* synchro);
__global__ void z_cu_powm_ul(z_cu_t const* base, unsigned long exp, unsigned long mod, z_cu_t* p, bool* synchro);
__global__ void z_cu_powm_ull(z_cu_t const* base, unsigned long long exp, unsigned long long mod, z_cu_t* p, bool* synchro);
__global__ void z_cu_powm_us(z_cu_t const* base, unsigned short exp, unsigned short mod, z_cu_t* p, bool* synchro);

#endif // Z_CU_POWM_CUH
