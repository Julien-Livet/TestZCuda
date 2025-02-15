#ifndef Z_T_H
#define Z_T_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef uintmax_t z_type;
typedef uintmax_t longest_type;

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

struct z_t_struct
{
    bool is_positive;
    z_type* bits;
    size_t size;
    bool is_nan;
    bool is_infinity;
    bool is_auto_adjust;
};

typedef struct z_t_struct z_t;

z_t z_abs(z_t z);
void z_adjust(z_t* z);
z_t z_copy(z_t z);
z_t z_factorial(z_t n);
void z_free(z_t* z);
z_t z_gcd(z_t a, z_t b);
z_t z_gcd_extended(z_t a, z_t b, z_t* u, z_t* v);
z_t z_infinity();
void z_invert(z_t* z);
bool z_is_auto_adjust(z_t z);
bool z_is_even(z_t z);
bool z_is_nan(z_t z);
bool z_is_infinity(z_t z);
bool z_is_negative(z_t z);
bool z_is_null(z_t z);
bool z_is_odd(z_t z);
bool z_is_positive(z_t z);
z_t z_max(z_t a, z_t b);
z_t z_min(z_t a, z_t b);
z_t z_nan();
void z_neg(z_t* z);
z_t z_number(z_t z);
size_t z_precision(z_t z);
void z_printf(z_t z, size_t base);
void z_printf_bits(z_t z);
void z_printf_bytes(z_t z);
void z_set_auto_adjust(z_t* z, bool is_auto_adjust);
void z_set_infinity(z_t* z);
void z_set_nan(z_t* z);
void z_set_negative(z_t* z);
void z_set_positive(z_t* z);
void z_set_precision(z_t* z, size_t precision);
void z_set_random(z_t* z);
int z_sign(z_t z);
z_t z_sqrt(z_t n);

#include "z_add.h"
#include "z_and.h"
#include "z_cmp.h"
#include "z_div_q.h"
#include "z_div_qr.h"
#include "z_div_r.h"
#include "z_fits.h"
#include "z_from.h"
#include "z_lshift.h"
#include "z_mul.h"
#include "z_or.h"
#include "z_pow.h"
#include "z_powm.h"
#include "z_prime.h"
#include "z_rshift.h"
#include "z_set_from.h"
#include "z_sub.h"
#include "z_to.h"
#include "z_xor.h"

#endif // Z_T_H
