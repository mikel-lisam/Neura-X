/* ==========================================================
 * Neura-X: Intelligence Without Limits.
 * Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
 *
 * BLAS Bridge — Wraps system BLAS or provides fallback
 * ========================================================== */

#include "nex_math.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ──────────────────────────────────────────────────────────
 * Try to link against system BLAS. If unavailable, use
 * built-in fallback implementations.
 * ────────────────────────────────────────────────────────── */

#if defined(__linux__) || defined(__APPLE__)
    extern void sgemm_(const char *transa, const char *transb,
                       const int *m, const int *n, const int *k,
                       const float *alpha, const float *a, const int *lda,
                       const float *b, const int *ldb,
                       const float *beta, float *c, const int *ldc)
        __attribute__((weak));

    extern float sdot_(const int *n, const float *x, const int *incx,
                       const float *y, const int *incy)
        __attribute__((weak));

    extern float snrm2_(const int *n, const float *x, const int *incx)
        __attribute__((weak));

    extern void sscal_(const int *n, const float *alpha,
                       float *x, const int *incx)
        __attribute__((weak));

    extern void saxpy_(const int *n, const float *alpha,
                       const float *x, const int *incx,
                       float *y, const int *incy)
        __attribute__((weak));
#endif

/* ──────────────────────────────────────────────────────────
 * Fallback SGEMM
 * ────────────────────────────────────────────────────────── */

static void fallback_sgemm(char trans_a, char trans_b,
                           int m, int n, int k,
                           float alpha,
                           const float *a, int lda,
                           const float *b, int ldb,
                           float beta,
                           float *c, int ldc)
{
    (void)trans_a;
    (void)trans_b;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += a[i * lda + p] * b[p * ldb + j];
            }
            c[i * ldc + j] = alpha * sum + beta * c[i * ldc + j];
        }
    }
}

/* ──────────────────────────────────────────────────────────
 * Public API
 * ────────────────────────────────────────────────────────── */

void nex_sgemm(char trans_a, char trans_b,
               int m, int n, int k,
               float alpha,
               const float *a, int lda,
               const float *b, int ldb,
               float beta,
               float *c, int ldc)
{
    if (!a || !b || !c || m <= 0 || n <= 0 || k <= 0) return;

#if defined(__linux__) || defined(__APPLE__)
    if (sgemm_) {
        sgemm_(&trans_a, &trans_b, &m, &n, &k,
               &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
        return;
    }
#endif

    fallback_sgemm(trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void nex_dgemm(char trans_a, char trans_b,
               int m, int n, int k,
               double alpha,
               const double *a, int lda,
               const double *b, int ldb,
               double beta,
               double *c, int ldc)
{
    if (!a || !b || !c || m <= 0 || n <= 0 || k <= 0) return;

    /* Suppress unused parameter warnings for the fallback path */
    (void)trans_a;
    (void)trans_b;

    /* Fallback for double precision */
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int p = 0; p < k; p++) {
                sum += a[i * lda + p] * b[p * ldb + j];
            }
            c[i * ldc + j] = alpha * sum + beta * c[i * ldc + j];
        }
    }
}

void nex_sgemv(char trans, int m, int n,
               float alpha,
               const float *a, int lda,
               const float *x, int incx,
               float beta,
               float *y, int incy)
{
    if (!a || !x || !y || m <= 0 || n <= 0) return;
    (void)trans;

    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += a[i * lda + j] * x[j * incx];
        }
        y[i * incy] = alpha * sum + beta * y[i * incy];
    }
}

float nex_sdot(int n, const float *x, int incx, const float *y, int incy)
{
    if (!x || !y || n <= 0) return 0.0f;

#if defined(__linux__) || defined(__APPLE__)
    if (sdot_) {
        return sdot_(&n, x, &incx, y, &incy);
    }
#endif

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += x[i * incx] * y[i * incy];
    }
    return sum;
}

float nex_snrm2(int n, const float *x, int incx)
{
    if (!x || n <= 0) return 0.0f;

#if defined(__linux__) || defined(__APPLE__)
    if (snrm2_) {
        return snrm2_(&n, x, &incx);
    }
#endif

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += x[i * incx] * x[i * incx];
    }
    return sqrtf(sum);
}

void nex_sscal(int n, float alpha, float *x, int incx)
{
    if (!x || n <= 0) return;

#if defined(__linux__) || defined(__APPLE__)
    if (sscal_) {
        sscal_(&n, &alpha, x, &incx);
        return;
    }
#endif

    for (int i = 0; i < n; i++) {
        x[i * incx] *= alpha;
    }
}

void nex_saxpy(int n, float alpha, const float *x, int incx,
               float *y, int incy)
{
    if (!x || !y || n <= 0) return;

#if defined(__linux__) || defined(__APPLE__)
    if (saxpy_) {
        saxpy_(&n, &alpha, x, &incx, y, &incy);
        return;
    }
#endif

    for (int i = 0; i < n; i++) {
        y[i * incy] += alpha * x[i * incx];
    }
}