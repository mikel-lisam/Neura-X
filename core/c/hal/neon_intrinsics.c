/* ==========================================================
 * Neura-X: Intelligence Without Limits.
 * Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
 *
 * ARM NEON Intrinsics — Vectorized math for ARM CPUs
 * Targets: Apple M-Series, Snapdragon X, Cortex-A
 * ========================================================== */

#include "hal.h"
#include <math.h>

#if defined(__aarch64__) || defined(__ARM_NEON)
    #include <arm_neon.h>
    #define NEX_HAS_NEON 1
#endif

/* ──────────────────────────────────────────────────────────
 * NEON implementations
 * ────────────────────────────────────────────────────────── */

#ifdef NEX_HAS_NEON

void nex_neon_vec_add_f32(const float *a, const float *b,
                          float *out, size_t n)
{
    size_t i = 0;
    const size_t simd_width = 4; /* 128 bits / 32 bits = 4 floats */
    const size_t simd_end = n - (n % simd_width);

    for (; i < simd_end; i += simd_width) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vr = vaddq_f32(va, vb);
        vst1q_f32(out + i, vr);
    }

    for (; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

void nex_neon_vec_mul_f32(const float *a, const float *b,
                          float *out, size_t n)
{
    size_t i = 0;
    const size_t simd_width = 4;
    const size_t simd_end = n - (n % simd_width);

    for (; i < simd_end; i += simd_width) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vr = vmulq_f32(va, vb);
        vst1q_f32(out + i, vr);
    }

    for (; i < n; i++) {
        out[i] = a[i] * b[i];
    }
}

void nex_neon_vec_scale_f32(const float *a, float scalar,
                            float *out, size_t n)
{
    size_t i = 0;
    const size_t simd_width = 4;
    const size_t simd_end = n - (n % simd_width);
    float32x4_t vs = vdupq_n_f32(scalar);

    for (; i < simd_end; i += simd_width) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vr = vmulq_f32(va, vs);
        vst1q_f32(out + i, vr);
    }

    for (; i < n; i++) {
        out[i] = a[i] * scalar;
    }
}

void nex_neon_vec_fma_f32(const float *a, const float *b,
                          const float *c, float *out, size_t n)
{
    size_t i = 0;
    const size_t simd_width = 4;
    const size_t simd_end = n - (n % simd_width);

    for (; i < simd_end; i += simd_width) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vld1q_f32(c + i);
        float32x4_t vr = vfmaq_f32(vc, va, vb);
        vst1q_f32(out + i, vr);
    }

    for (; i < n; i++) {
        out[i] = a[i] * b[i] + c[i];
    }
}

float nex_neon_vec_dot_f32(const float *a, const float *b, size_t n)
{
    size_t i = 0;
    const size_t simd_width = 4;
    const size_t simd_end = n - (n % simd_width);
    float32x4_t vsum = vdupq_n_f32(0.0f);

    for (; i < simd_end; i += simd_width) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vsum = vmlaq_f32(vsum, va, vb);
    }

    /* Horizontal sum */
    float result = vsum[0] + vsum[1] + vsum[2] + vsum[3];

    for (; i < n; i++) {
        result += a[i] * b[i];
    }

    return result;
}

void nex_neon_vec_sin_f32(const float *input, float *output, size_t n)
{
    /* NEON does not have a native sin instruction.
     * We process in batches of 4 for cache efficiency. */
    size_t i = 0;
    const size_t simd_width = 4;
    const size_t simd_end = n - (n % simd_width);

    for (; i < simd_end; i += simd_width) {
        output[i + 0] = sinf(input[i + 0]);
        output[i + 1] = sinf(input[i + 1]);
        output[i + 2] = sinf(input[i + 2]);
        output[i + 3] = sinf(input[i + 3]);
    }

    for (; i < n; i++) {
        output[i] = sinf(input[i]);
    }
}

void nex_neon_holo_bind_i8(const int8_t *a, const int8_t *b,
                           int8_t *out, size_t n)
{
    size_t i = 0;
    const size_t simd_width = 16; /* 128 bits / 8 bits = 16 int8s */
    const size_t simd_end = n - (n % simd_width);

    for (; i < simd_end; i += simd_width) {
        int8x16_t va = vld1q_s8(a + i);
        int8x16_t vb = vld1q_s8(b + i);

        /* Multiply element-wise */
        int8x16_t product = vmulq_s8(va, vb);

        /* Compare: > 0 → 1, else → -1 */
        uint8x16_t cmp = vcgtq_s8(product, vdupq_n_s8(0));
        int8x16_t ones = vdupq_n_s8(1);
        int8x16_t neg_ones = vdupq_n_s8(-1);
        int8x16_t result = vbslq_s8(cmp, ones, neg_ones);

        vst1q_s8(out + i, result);
    }

    for (; i < n; i++) {
        out[i] = (a[i] * b[i] > 0) ? 1 : -1;
    }
}

int nex_neon_holo_similarity_i8(const int8_t *a, const int8_t *b, size_t n)
{
    size_t i = 0;
    const size_t simd_width = 16;
    const size_t simd_end = n - (n % simd_width);
    int count = 0;

    for (; i < simd_end; i += simd_width) {
        int8x16_t va = vld1q_s8(a + i);
        int8x16_t vb = vld1q_s8(b + i);
        uint8x16_t cmp = vceqq_s8(va, vb);

        /* Count true lanes */
        for (int j = 0; j < simd_width; j++) {
            if (((uint8_t *)&cmp)[j] != 0) count++;
        }
    }

    for (; i < n; i++) {
        if (a[i] == b[i]) count++;
    }

    return count;
}

#endif /* NEX_HAS_NEON */

/* ──────────────────────────────────────────────────────────
 * Availability check
 * ────────────────────────────────────────────────────────── */

int nex_neon_available(void)
{
#ifdef NEX_HAS_NEON
    return 1;
#else
    return 0;
#endif
}