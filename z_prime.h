#ifndef Z_PRIME_H
#define Z_PRIME_H

#include "z_t.h"

bool z_is_coprime(z_t a, z_t b);
int z_is_prime(z_t n, size_t reps);
z_t z_next_prime(z_t n);
z_t z_previous_prime(z_t n);

#endif // Z_PRIME_H
