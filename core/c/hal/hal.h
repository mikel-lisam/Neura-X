/* ==========================================================
 * Neura-X: Intelligence Without Limits.
 * Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
 *
 * Hardware Abstraction Layer — Main Header
 * ========================================================== */

#ifndef NEURA_X_HAL_H
#define NEURA_X_HAL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ──────────────────────────────────────────────────────────
 * CPU Vendor Enumeration
 * ────────────────────────────────────────────────────────── */

typedef enum {
    NEX_CPU_VENDOR_UNKNOWN = 0,
    NEX_CPU_VENDOR_INTEL,
    NEX_CPU_VENDOR_AMD,
    NEX_CPU_VENDOR_ARM,
    NEX_CPU_VENDOR_APPLE,
    NEX_CPU_VENDOR_RISCV
} nex_cpu_vendor_t;

/* ──────────────────────────────────────────────────────────
 * CPU Feature Flags
 * ────────────────────────────────────────────────────────── */

typedef struct {
    bool sse2;
    bool sse4_1;
    bool sse4_2;
    bool avx;
    bool avx2;
    bool avx512f;
    bool avx512bw;
    bool avx512vl;
    bool fma;
    bool neon;
    bool amx;
    bool bmi1;
    bool bmi2;
} nex_cpu_features_t;

/* ──────────────────────────────────────────────────────────
 * CPU Information Structure
 * ────────────────────────────────────────────────────────── */

typedef struct {
    nex_cpu_vendor_t vendor;
    char             vendor_string[16];
    char             model_name[64];
    int              physical_cores;
    int              logical_cores;
    int              generation;
    uint64_t         l1_cache_bytes;
    uint64_t         l2_cache_bytes;
    uint64_t         l3_cache_bytes;
    double           base_frequency_ghz;
    double           max_frequency_ghz;
    nex_cpu_features_t features;
    int              simd_width_bytes;
} nex_cpu_info_t;

/* ──────────────────────────────────────────────────────────
 * CPU Detection API
 * ────────────────────────────────────────────────────────── */

/* Detect and populate CPU information. Returns 0 on success. */
int nex_cpu_detect(nex_cpu_info_t *info);

/* Print CPU information to stdout. */
void nex_cpu_print_info(const nex_cpu_info_t *info);

/* Get the optimal SIMD width in bytes for the detected CPU. */
int nex_cpu_get_simd_width(const nex_cpu_info_t *info);

/* Check if a specific feature is available. */
bool nex_cpu_has_feature(const nex_cpu_info_t *info, const char *feature_name);

/* ──────────────────────────────────────────────────────────
 * SIMD Operations API
 * ────────────────────────────────────────────────────────── */

/* Vector add: out[i] = a[i] + b[i] */
void nex_vec_add_f32(const float *a, const float *b, float *out, size_t n);

/* Vector multiply: out[i] = a[i] * b[i] */
void nex_vec_mul_f32(const float *a, const float *b, float *out, size_t n);

/* Vector scale: out[i] = a[i] * scalar */
void nex_vec_scale_f32(const float *a, float scalar, float *out, size_t n);

/* Vector fused multiply-add: out[i] = a[i] * b[i] + c[i] */
void nex_vec_fma_f32(const float *a, const float *b, const float *c,
                     float *out, size_t n);

/* Dot product of two vectors. */
float nex_vec_dot_f32(const float *a, const float *b, size_t n);

/* Compute sine of each element (vectorized). */
void nex_vec_sin_f32(const float *input, float *output, size_t n);

/* Compute cosine of each element (vectorized). */
void nex_vec_cos_f32(const float *input, float *output, size_t n);

/* XOR operation for Holographic Memory (int8 arrays of -1/+1). */
void nex_holo_bind_i8(const int8_t *a, const int8_t *b, int8_t *out, size_t n);

/* POPCOUNT for Holographic Memory retrieval. */
int nex_holo_similarity_i8(const int8_t *a, const int8_t *b, size_t n);

/* ──────────────────────────────────────────────────────────
 * Thermal Monitoring API
 * ────────────────────────────────────────────────────────── */

typedef struct {
    double current_temp_celsius;
    double throttle_threshold_celsius;
    bool   is_throttling;
    int    fan_speed_rpm;
    bool   sensor_available;
} nex_thermal_info_t;

/* Read current thermal information. Returns 0 on success. */
int nex_thermal_read(nex_thermal_info_t *info);

/* Get the device form factor (laptop, desktop, server). */
typedef enum {
    NEX_DEVICE_UNKNOWN = 0,
    NEX_DEVICE_LAPTOP,
    NEX_DEVICE_DESKTOP,
    NEX_DEVICE_SERVER,
    NEX_DEVICE_EMBEDDED
} nex_device_type_t;

nex_device_type_t nex_detect_device_type(void);

/* ──────────────────────────────────────────────────────────
 * Version & Attribution
 * ────────────────────────────────────────────────────────── */

#define NEX_HAL_VERSION_MAJOR 1
#define NEX_HAL_VERSION_MINOR 0
#define NEX_HAL_VERSION_PATCH 0
#define NEX_HAL_VERSION_STRING "1.0.0"
#define NEX_FOUNDER "Edusei Mikel Lisamba"
#define NEX_INSTITUTION "Open University of Kenya"

/* Get HAL version string. */
const char *nex_hal_version(void);

/* Get founder attribution string. */
const char *nex_hal_founder(void);

#ifdef __cplusplus
}
#endif

#endif /* NEURA_X_HAL_H */