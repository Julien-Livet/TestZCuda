#include <stdio.h>

#include "z_cu_t.cuh"

int main()
{/*
    unsigned long long n;
    scanf("%llu", &n);
    z_t z = z_from_ull(n);
*/
    //z_t z = z_from_str("1", 10);
    //z_t z = z_from_str("2", 10);
    //z_t z = z_from_str("3", 10);
    //z_t z = z_from_str("4", 10);
    //z_t z = z_from_str("5", 10);
    //z_t z = z_from_str("6", 10);
    //z_t z = z_from_str("9", 10);
    //z_t z = z_from_str("35", 10);
    z_t z = z_from_str("56062005704198360319209", 10);
    //z_t z = z_from_str("4113101149215104800030529537915953170486139623539759933135949994882770404074832568499", 10);

    z_cu_t* z_cu;
    cudaMalloc(&z_cu, sizeof(z_cu_t));

    z_cu_from_z(&z, z_cu);

    unsigned int* p;
    cudaMalloc(&p, sizeof(unsigned int));
    size_t* primes_size;
    cudaMalloc(&primes_size, sizeof(size_t));

    cudaPrimes(&p, primes_size);

    size_t p_s;
    cudaMemcpy(&p_s, primes_size, sizeof(size_t), cudaMemcpyDeviceToHost);

    int* result;
    cudaMalloc(&result, sizeof(int));

    bool* synchro;
    cudaMalloc(&synchro, sizeof(bool));
    cudaMemset(synchro, false, sizeof(bool));

    z_cu_is_prime<<<1, 1>>>(z_cu, 25, result, p, p_s, synchro);

    cudaDeviceSynchronize();

    int r;
    cudaMemcpy(&r, result, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Is prime: %d\n", r);

    z_free(&z);
    
    z_cu_free<<<1, 1>>>(z_cu, synchro);

    cudaFree(z_cu);
    cudaFree(result);
    cudaFree(p);
    cudaFree(primes_size);
    cudaFree(synchro);

    return 0;
}
