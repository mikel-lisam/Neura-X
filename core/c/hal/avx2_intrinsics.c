/* ==========================================================
 * Neura-X: Intelligence Without Limits.
 * Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
 *
 * AVX2 SIMD Intrinsics — Vectorized math for x86 CPUs
 * Requires: AVX2 (Intel Haswell 2013+, AMD Excavator 2015+)
 * ========================================================== */

#include <math.h>      /* System math.h MUST come first */
#include <string.h>
#include "hal.h"       /* Project headers after system headers */

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    #ifdef __AVX2__
        #include <immintrin.h>
        #define NEX_HAS_AVX2 1
    #endif
#endif

/* ──────────────────────────────────────────────────────────
 * Fallback scalar implementations (used when AVX2 unavailable)
 * ────────────────────────────────────────────────────────── */

static void scalar_vec_add_f32(const float *a, const float *b,
                               float *out, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

static void scalar_vec_mul_f32(const float *a, const float *b,
                               float *out, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        out[i] = a[i] * b[i];
    }
}

static void scalar_vec_scale_f32(const float *a, float scalar,
                                 float *out, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        out[i] = a[i] * scalar;
    }
}

static void scalar_vec_fma_f32(const float *a, const float *b,
                               const float *c, float *out, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        out[i] = a[i] * b[i] + c[i];
    }
}

static float scalar_vec_dot_f32(const float *a, const float *b, size_t n)
{
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

static void scalar_vec_sin_f32(const float *input, float *output, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        output[i] = sinf(input[i]);
    }
}

static void scalar_vec_cos_f32(const float *input, float *output, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        output[i] = cosf(input[i]);
    }
}

static void scalar_holo_bind_i8(const int8_t *a, const int8_t *b,
                                int8_t *out, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        out[i] = (a[i] * b[i] > 0) ? 1 : -1;
    }
}

static int scalar_holo_similarity_i8(const int8_t *a, const int8_t *b, size_t n)
{
    int count = 0;
    for (size_t i = 0; i < n; i++) {
        if (a[i] == b[i]) count++;
    }
    return count;
}

/* ──────────────────────────────────────────────────────────
 * AVX2 implementations
 * ────────────────────────────────────────────────────────── */

#ifdef NEX_HAS_AVX2

static void avx2_vec_add_f32(const float *a, const float *b,
                             float *out, size_t n)
{
    size_t i = 0;
    const size_t simd_width = 8; /* 256 bits / 32 bits = 8 floats */
    const size_t simd_end = n - (n % simd_width);

    for (; i < simd_end; i += simd_width) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(out + i, vr);
    }

    /* Handle remaining elements */
    for (; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

static void avx2_vec_mul_f32(const float *a, const float *b,
                             float *out, size_t n)
{
    size_t i = 0;
    const size_t simd_width = 8;
    const size_t simd_end = n - (n % simd_width);

    for (; i < simd_end; i += simd_width) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vr = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(out + i, vr);
    }

    for (; i < n; i++) {
        out[i] = a[i] * b[i];
    }
}

static void avx2_vec_scale_f32(const float *a, float scalar,
                               float *out, size_t n)
{
    size_t i = 0;
    const size_t simd_width = 8;
    const size_t simd_end = n - (n % simd_width);
    __m256 vs = _mm256_set1_ps(scalar);

    for (; i < simd_end; i += simd_width) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vr = _mm256_mul_ps(va, vs);
        _mm256_storeu_ps(out + i, vr);
    }

    for (; i < n; i++) {
        out[i] = a[i] * scalar;
    }
}

static void avx2_vec_fma_f32(const float *a, const float *b,
                             const float *c, float *out, size_t n)
{
    size_t i = 0;
    const size_t simd_width = 8;
    const size_t simd_end = n - (n % simd_width);

    for (; i < simd_end; i += simd_width) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_loadu_ps(c + i);
#ifdef __FMA__
        __m256 vr = _mm256_fmadd_ps(va, vb, vc);
#else
        __m256 vr = _mm256_add_ps(_mm256_mul_ps(va, vb), vc);
#endif
        _mm256_storeu_ps(out + i, vr);
    }

    for (; i < n; i++) {
        out[i] = a[i] * b[i] + c[i];
    }
}

static float avx2_vec_dot_f32(const float *a, const float *b, size_t n)
{
    size_t i = 0;
    const size_t simd_width = 8;
    const size_t simd_end = n - (n % simd_width);
    __m256 vsum = _mm256_setzero_ps();

    for (; i < simd_end; i += simd_width) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        vsum = _mm256_add_ps(vsum, _mm256_mul_ps(va, vb));
    }

    /* Horizontal sum of 8 floats */
    __m128 hi = _mm256_extractf128_ps(vsum, 1);
    __m128 lo = _mm256_castps256_ps128(vsum);
    __m128 sum128 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum128);
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    float result = _mm_cvtss_f32(sums);

    /* Handle remaining elements */
    for (; i < n; i++) {
        result += a[i] * b[i];
    }

    return result;
}

/* Polynomial approximation for vectorized sine (AVX2) */
static void avx2_vec_sin_f32(const float *input, float *output, size_t n)
{
    /*
     * For production, we use the scalar sinf for accuracy.
     * A polynomial approximation (Bhaskara or Taylor) can be
     * substituted here for higher throughput at reduced precision.
     */
    size_t i = 0;
    const size_t simd_width = 8;
    const size_t simd_end = n - (n % simd_width);

    /* Process 8 elements at a time using scalar sinf */
    for (; i < simd_end; i += simd_width) {
        output[i + 0] = sinf(input[i + 0]);
        output[i + 1] = sinf(input[i + 1]);
        output[i + 2] = sinf(input[i + 2]);
        output[i + 3] = sinf(input[i + 3]);
        output[i + 4] = sinf(input[i + 4]);
        output[i + 5] = sinf(input[i + 5]);
        output[i + 6] = sinf(input[i + 6]);
        output[i + 7] = sinf(input[i + 7]);
    }

    for (; i < n; i++) {
        output[i] = sinf(input[i]);
    }
}

static void avx2_vec_cos_f32(const float *input, float *output, size_t n)
{
    size_t i = 0;
    const size_t simd_width = 8;
    const size_t simd_end = n - (n % simd_width);

    for (; i < simd_end; i += simd_width) {
        output[i + 0] = cosf(input[i + 0]);
        output[i + 1] = cosf(input[i + 1]);
        output[i + 2] = cosf(input[i + 2]);
        output[i + 3] = cosf(input[i + 3]);
        output[i + 4] = cosf(input[i + 4]);
        output[i + 5] = cosf(input[i + 5]);
        output[i + 6] = cosf(input[i + 6]);
        output[i + 7] = cosf(input[i + 7]);
    }

    for (; i < n; i++) {
        output[i] = cosf(input[i]);
    }
}

static void avx2_holo_bind_i8(const int8_t *a, const int8_t *b,
                              int8_t *out, size_t n)
{
    size_t i = 0;
    const size_t simd_width = 32; /* 256 bits / 8 bits = 32 int8s */
    const size_t simd_end = n - (n % simd_width);
    __m256i zero = _mm256_setzero_si256();

    for (; i < simd_end; i += simd_width) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));

        /* Multiply: positive if same sign, negative if different */
        __m256i product = _mm256_mullo_epi8(va, vb);

        /* Compare: if product > 0 → 1, else → -1 */
        __m256i cmp = _mm256_cmpgt_epi8(product, zero);
        __m256i ones = _mm256_set1_epi8(1);
        __m256i neg_ones = _mm256_set1_epi8(-1);
        __m256i result = _mm256_blendv_epi8(neg_ones, ones, cmp);

        _mm256_storeu_si256((__m256i *)(out + i), result);
    }

    for (; i < n; i++) {
        out[i] = (a[i] * b[i] > 0) ? 1 : -1;
    }
}

static int avx2_holo_similarity_i8(const int8_t *a, const int8_t *b, size_t n)
{
    size_t i = 0;
    const size_t simd_width = 32;
    const size_t simd_end = n - (n % simd_width);
    int count = 0;

    for (; i < simd_end; i += simd_width) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
        __m256i cmp = _mm256_cmpeq_epi8(va, vb);
        __m256i ones = _mm256_set1_epi8(1);
        __m256i matches = _mm256_and_si256(cmp, ones);

        /* Count matching bytes */
        for (int j = 0; j < simd_width; j++) {
            if (((int8_t *)&matches)[j] != 0) count++;
        }
    }

    for (; i < n; i++) {
        if (a[i] == b[i]) count++;
    }

    return count;
}

#endif /* NEX_HAS_AVX2 */

/* ──────────────────────────────────────────────────────────
 * Public API — Dispatch to best available implementation
 * ────────────────────────────────────────────────────────── */

void nex_vec_add_f32(const float *a, const float *b, float *out, size_t n)
{
    if (!a || !b || !out || n == 0) return;
#ifdef NEX_HAS_AVX2
    avx2_vec_add_f32(a, b, out, n);
#else
    scalar_vec_add_f32(a, b, out, n);
#endif
}

void nex_vec_mul_f32(const float *a, const float *b, float *out, size_t n)
{
    if (!a || !b || !out || n == 0) return;
#ifdef NEX_HAS_AVX2
    avx2_vec_mul_f32(a, b, out, n);
#else
    scalar_vec_mul_f32(a, b, out, n);
#endif
}

void nex_vec_scale_f32(const float *a, float scalar, float *out, size_t n)
{
    if (!a || !out || n == 0) return;
#ifdef NEX_HAS_AVX2
    avx2_vec_scale_f32(a, scalar, out, n);
#else
    scalar_vec_scale_f32(a, scalar, out, n);
#endif
}

void nex_vec_fma_f32(const float *a, const float *b, const float *c,
                     float *out, size_t n)
{
    if (!a || !b || !c || !out || n == 0) return;
#ifdef NEX_HAS_AVX2
    avx2_vec_fma_f32(a, b, c, out, n);
#else
    scalar_vec_fma_f32(a, b, c, out, n);
#endif
}

float nex_vec_dot_f32(const float *a, const float *b, size_t n)
{
    if (!a || !b || n == 0) return 0.0f;
#ifdef NEX_HAS_AVX2
    return avx2_vec_dot_f32(a, b, n);
#else
    return scalar_vec_dot_f32(a, b, n);
#endif
}

void nex_vec_sin_f32(const float *input, float *output, size_t n)
{
    if (!input || !output || n == 0) return;
#ifdef NEX_HAS_AVX2
    avx2_vec_sin_f32(input, output, n);
#else
    scalar_vec_sin_f32(input, output, n);
#endif
}

void nex_vec_cos_f32(const float *input, float *output, size_t n)
{
    if (!input || !output || n == 0) return;
#ifdef NEX_HAS_AVX2
    avx2_vec_cos_f32(input, output, n);
#else
    scalar_vec_cos_f32(input, output, n);
#endif
}

void nex_holo_bind_i8(const int8_t *a, const int8_t *b, int8_t *out, size_t n)
{
    if (!a || !b || !out || n == 0) return;
#ifdef NEX_HAS_AVX2
    avx2_holo_bind_i8(a, b, out, n);
#else
    scalar_holo_bind_i8(a, b, out, n);
#endif
}

int nex_holo_similarity_i8(const int8_t *a, const int8_t *b, size_t n)
{
    if (!a || !b || n == 0) return 0;
#ifdef NEX_HAS_AVX2
    return avx2_holo_similarity_i8(a, b, n);
#else
    return scalar_holo_similarity_i8(a, b, n);
#endif
}