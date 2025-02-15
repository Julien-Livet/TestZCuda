#!/bin/bash
nvcc -c z_t.c
nvcc -c z_add.c
nvcc -c z_and.c
nvcc -c z_cmp.c
nvcc -c z_div_q.c
nvcc -c z_div_qr.c
nvcc -c z_div_r.c
nvcc -c z_fits.c 
nvcc -c z_from.c
nvcc -c z_lshift.c
nvcc -c z_mul.c
nvcc -c z_or.c
nvcc -c z_pow.c
nvcc -c z_powm.c
nvcc -c z_prime.c
nvcc -c z_rshift.c
nvcc -c z_set_from.c
nvcc -c z_sub.c
nvcc -c z_to.c
nvcc -c z_xor.c 

nvcc -c main.cu
nvcc -rdc=true -c z_cu_t.cu
nvcc -rdc=true -c z_cu_add.cu
nvcc -rdc=true -c z_cu_and.cu
nvcc -rdc=true -c z_cu_cmp.cu
nvcc -rdc=true -c z_cu_div_q.cu
nvcc -rdc=true -c z_cu_div_qr.cu
nvcc -rdc=true -c z_cu_div_r.cu
nvcc -rdc=true -c z_cu_fits.cu
nvcc -rdc=true -c z_cu_from.cu
nvcc -rdc=true -c z_cu_lshift.cu
nvcc -rdc=true -c z_cu_mul.cu
nvcc -rdc=true -c z_cu_or.cu
nvcc -rdc=true -c z_cu_pow.cu
nvcc -rdc=true -c z_cu_powm.cu
nvcc -rdc=true -c z_cu_prime.cu
nvcc -rdc=true -c z_cu_rshift.cu
nvcc -rdc=true -c z_cu_set_from.cu
nvcc -rdc=true -c z_cu_sub.cu
nvcc -rdc=true -c z_cu_to.cu
nvcc -rdc=true -c z_cu_xor.cu

nvcc z_t.o z_add.o z_and.o z_cmp.o z_div_q.o z_div_qr.o z_div_r.o z_fits.o z_from.o z_lshift.o z_mul.o z_or.o z_pow.o z_powm.o z_prime.o z_rshift.o z_set_from.o z_sub.o z_to.o z_xor.o main.o z_cu_t.o z_cu_add.o z_cu_and.o z_cu_cmp.o z_cu_div_q.o z_cu_div_qr.o z_cu_div_r.o z_cu_fits.o z_cu_from.o z_cu_lshift.o z_cu_mul.o z_cu_or.o z_cu_pow.o z_cu_powm.o z_cu_prime.o z_cu_rshift.o z_cu_set_from.o z_cu_sub.o z_cu_to.o z_cu_xor.o -o TestZCuda
