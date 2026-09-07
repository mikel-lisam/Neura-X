/* ==========================================================
 * Neura-X: Intelligence Without Limits.
 * Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
 *
 * Neura-X Math Bridge — Header
 * ========================================================== */

#ifndef NEURA_X_NEX_MATH_H
#define NEURA_X_NEX_MATH_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ──────────────────────────────────────────────────────────
 * BLAS Bridge — Matrix operations
 * ────────────────────────────────────────────────────────── */

/* C = alpha * A × B + beta * C  (Single precision) */
void nex_sgemm(char trans_a, char trans_b,
               int m, int n, int k,
               float alpha,
               const float *a, int lda,
               const float *b, int ldb,
               float beta,
               float *c, int ldc);

/* C = alpha * A × B + beta * C  (Double precision) */
void nex_dgemm(char trans_a, char trans_b,
               int m, int n, int k,
               double alpha,
               const double *a, int lda,
               const double *b, int ldb,
               double beta,
               double *c, int ldc);

/* y = alpha * A × x + beta * y  (Matrix-vector multiply) */
void nex_sgemv(char trans, int m, int n,
               float alpha,
               const float *a, int lda,
               const float *x, int incx,
               float beta,
               float *y, int incy);

/* Dot product */
float nex_sdot(int n, const float *x, int incx, const float *y, int incy);

/* Vector norm */
float nex_snrm2(int n, const float *x, int incx);

/* Vector scale: x = alpha * x */
void nex_sscal(int n, float alpha, float *x, int incx);

/* Vector add: y = alpha * x + y */
void nex_saxpy(int n, float alpha, const float *x, int incx,
               float *y, int incy);

/* ──────────────────────────────────────────────────────────
 * LAPACK Bridge — Decompositions
 * ────────────────────────────────────────────────────────── */

/* SVD: A = U × S × V^T (single precision, simplified) */
int nex_sgesvd_simplified(float *a, int m, int n,
                          float *u, float *s, float *vt,
                          int max_rank);

/* Solve linear system: A × x = b */
int nex_sgesv(int n, int nrhs, float *a, int lda,
              int *ipiv, float *b, int ldb);

/* ──────────────────────────────────────────────────────────
 * Neura-X Specific Math
 * ────────────────────────────────────────────────────────── */

/* Fractal bloom: generate row i of a weight matrix from seed */
void nex_fractal_bloom_row(
    int row, int cols, int K,
    const float *a, const float *w1, const float *w2, const float *phi,
    float *output);

/* Fractal bloom: generate a block of rows */
void nex_fractal_bloom_block(
    int row_start, int row_end, int cols, int K,
    const float *a, const float *w1, const float *w2, const float *phi,
    float *output);

/* Softmax (numerically stable) */
void nex_softmax_f32(const float *input, float *output, int n);

/* Gumbel-Softmax gate */
void nex_gumbel_softmax_gate(
    const float *logits, float temperature,
    float *gates, int n, bool hard);

#ifdef __cplusplus
}
#endif

#endif /* NEURA_X_NEX_MATH_H */