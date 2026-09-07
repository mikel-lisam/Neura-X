/* ==========================================================
 * Neura-X: Intelligence Without Limits.
 * Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
 *
 * AVX-512 SIMD Intrinsics — 512-bit vectorized math
 * Requires: AVX-512F (Intel Skylake-X+, AMD Zen4+)
 * Falls back to AVX2 or scalar if AVX-512 is unavailable.
 * ========================================================== */

#include "hal.h"
#include <math.h>

#if defined(__x86_64__) || defined(_M_X64)
    #ifdef __AVX512F__
        #include <immintrin.h>
        #define NEX_HAS_AVX512 1
    #endif
#endif

/* ──────────────────────────────────────────────────────────
 * AVX-512 implementations (compiled only when available)
 * ────────────────────────────────────────────────────────── */

#ifdef NEX_HAS_AVX512

void nex_avx512_vec_add_f32(const float *a, const float *b,
                            float *out, size_t n)
{
    size_t i = 0;
    const size_t simd_width = 16; /* 512 bits / 32 bits = 16 floats */
    const size_t simd_end = n - (n % simd_width);

    for (; i < simd_end; i += simd_width) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vr = _mm512_add_ps(va, vb);
        _mm512_storeu_ps(out + i, vr);
    }

    for (; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

void nex_avx512_vec_mul_f32(const float *a, const float *b,
                            float *out, size_t n)
{
    size_t i = 0;
    const size_t simd_width = 16;
    const size_t simd_end = n - (n % simd_width);

    for (; i < simd_end; i += simd_width) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vr = _mm512_mul_ps(va, vb);
        _mm512_storeu_ps(out + i, vr);
    }

    for (; i < n; i++) {
        out[i] = a[i] * b[i];
    }
}

void nex_avx512_vec_scale_f32(const float *a, float scalar,
                              float *out, size_t n)
{
    size_t i = 0;
    const size_t simd_width = 16;
    const size_t simd_end = n - (n % simd_width);
    __m512 vs = _mm512_set1_ps(scalar);

    for (; i < simd_end; i += simd_width) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vr = _mm512_mul_ps(va, vs);
        _mm512_storeu_ps(out + i, vr);
    }

    for (; i < n; i++) {
        out[i] = a[i] * scalar;
    }
}

float nex_avx512_vec_dot_f32(const float *a, const float *b, size_t n)
{
    size_t i = 0;
    const size_t simd_width = 16;
    const size_t simd_end = n - (n % simd_width);
    __m512 vsum = _mm512_setzero_ps();

    for (; i < simd_end; i += simd_width) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        vsum = _mm512_fmadd_ps(va, vb, vsum);
    }

    float result = _mm512_reduce_add_ps(vsum);

    for (; i < n; i++) {
        result += a[i] * b[i];
    }

    return result;
}

void nex_avx512_vec_sin_f32(const float *input, float *output, size_t n)
{
    /* Use AVX-512 for batching, scalar sinf for accuracy */
    size_t i = 0;
    const size_t simd_width = 16;
    const size_t simd_end = n - (n % simd_width);

    for (; i < simd_end; i += simd_width) {
        for (int j = 0; j < simd_width; j++) {
            output[i + j] = sinf(input[i + j]);
        }
    }

    for (; i < n; i++) {
        output[i] = sinf(input[i]);
    }
}

#endif /* NEX_HAS_AVX512 */

/* ──────────────────────────────────────────────────────────
 * Availability check
 * ────────────────────────────────────────────────────────── */

int nex_avx512_available(void)
{
#ifdef NEX_HAS_AVX512
    return 1;
#else
    return 0;
#endif
}