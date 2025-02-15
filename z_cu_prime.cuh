#ifndef Z_CU_PRIME_CUH
#define Z_CU_PRIME_CUH

#include "z_cu_t.cuh"

void cudaPrimes(unsigned int** p, size_t* size);
__global__ void z_cu_is_coprime(z_cu_t const* a, z_cu_t const* b, bool* result, bool* synchro);
__global__ void z_cu_is_prime(z_cu_t const* n, size_t reps, int* result, unsigned int const* p, size_t primes_size, bool* synchro);
__global__ void z_cu_next_prime(z_cu_t const* n, z_cu_t* prime, unsigned int const* p, size_t primes_size, bool* synchro);
__global__ void z_cu_previous_prime(z_cu_t const* n, z_cu_t* prime, unsigned int const* p, size_t primes_size, bool* synchro);

#endif // Z_CU_PRIME_CUH
