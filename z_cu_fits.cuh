#ifndef Z_CU_FITS_CUH
#define Z_CU_FITS_CUH

#include "z_cu_t.cuh"

__global__ void z_cu_fits_c(z_cu_t const* z, bool* fits, bool* synchro);
__global__ void z_cu_fits_i(z_cu_t const* z, bool* fits, bool* synchro);
__global__ void z_cu_fits_l(z_cu_t const* z, bool* fits, bool* synchro);
__global__ void z_cu_fits_ll(z_cu_t const* z, bool* fits, bool* synchro);
__global__ void z_cu_fits_s(z_cu_t const* z, bool* fits, bool* synchro);
__global__ void z_cu_fits_uc(z_cu_t const* z, bool* fits, bool* synchro);
__global__ void z_cu_fits_ui(z_cu_t const* z, bool* fits, bool* synchro);
__global__ void z_cu_fits_ul(z_cu_t const* z, bool* fits, bool* synchro);
__global__ void z_cu_fits_ull(z_cu_t const* z, bool* fits, bool* synchro);
__global__ void z_cu_fits_us(z_cu_t const* z, bool* fits, bool* synchro);

#endif // Z_CU_FITS_CUH
