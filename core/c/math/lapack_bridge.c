/* ==========================================================
 * Neura-X: Intelligence Without Limits.
 * Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
 *
 * LAPACK Bridge — SVD and linear solvers
 * ========================================================== */

#include "nex_math.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ──────────────────────────────────────────────────────────
 * Simplified SVD via Power Iteration
 * ────────────────────────────────────────────────────────── */

int nex_sgesvd_simplified(float *a, int m, int n,
                          float *u, float *s, float *vt,
                          int max_rank)
{
    if (!a || !u || !s || !vt || m <= 0 || n <= 0 || max_rank <= 0)
        return -1;

    int rank = (max_rank < m && max_rank < n) ? max_rank : (m < n ? m : n);
    int power_iters = 20;

    float *work_col = (float *)calloc((size_t)m, sizeof(float));
    float *work_row = (float *)calloc((size_t)n, sizeof(float));
    float *temp = (float *)calloc((size_t)(m > n ? m : n), sizeof(float));

    if (!work_col || !work_row || !temp) {
        free(work_col); free(work_row); free(temp);
        return -1;
    }

    for (int r = 0; r < rank; r++) {
        /* Initialize with deterministic pseudo-random vector */
        for (int j = 0; j < n; j++) {
            work_row[j] = (float)((j * 7 + r * 13 + 1) % 100) / 100.0f - 0.5f;
        }

        /* Power iteration: v = A^T × A × v */
        for (int iter = 0; iter < power_iters; iter++) {
            /* temp = A × v */
            for (int i = 0; i < m; i++) {
                float sum = 0.0f;
                for (int j = 0; j < n; j++) {
                    sum += a[i * n + j] * work_row[j];
                }
                temp[i] = sum;
            }

            /* work_row = A^T × temp */
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int i = 0; i < m; i++) {
                    sum += a[i * n + j] * temp[i];
                }
                work_row[j] = sum;
            }

            /* Normalize */
            float norm = 0.0f;
            for (int j = 0; j < n; j++) {
                norm += work_row[j] * work_row[j];
            }
            norm = sqrtf(norm);
            if (norm > 1e-10f) {
                for (int j = 0; j < n; j++) {
                    work_row[j] /= norm;
                }
            }
        }

        /* Compute u = A × v / sigma */
        float sigma = 0.0f;
        for (int i = 0; i < m; i++) {
            float sum = 0.0f;
            for (int j = 0; j < n; j++) {
                sum += a[i * n + j] * work_row[j];
            }
            work_col[i] = sum;
            sigma += sum * sum;
        }
        sigma = sqrtf(sigma);

        if (sigma > 1e-10f) {
            for (int i = 0; i < m; i++) {
                u[i * rank + r] = work_col[i] / sigma;
            }
        }

        s[r] = sigma;

        for (int j = 0; j < n; j++) {
            vt[r * n + j] = work_row[j];
        }

        /* Deflate: A = A - sigma × u × v^T */
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                a[i * n + j] -= sigma * u[i * rank + r] * vt[r * n + j];
            }
        }
    }

    free(work_col);
    free(work_row);
    free(temp);

    return 0;
}

/* ──────────────────────────────────────────────────────────
 * Linear System Solver (Gaussian Elimination with Pivoting)
 * ────────────────────────────────────────────────────────── */

int nex_sgesv(int n, int nrhs, float *a, int lda,
              int *ipiv, float *b, int ldb)
{
    if (!a || !b || !ipiv || n <= 0 || nrhs <= 0) return -1;

    /* Forward elimination with partial pivoting */
    for (int col = 0; col < n; col++) {
        /* Find pivot */
        int pivot_row = col;
        float max_val = fabsf(a[col * lda + col]);
        for (int row = col + 1; row < n; row++) {
            float val = fabsf(a[row * lda + col]);
            if (val > max_val) {
                max_val = val;
                pivot_row = row;
            }
        }

        ipiv[col] = pivot_row;

        if (max_val < 1e-12f) return -1; /* Singular matrix */

        /* Swap rows */
        if (pivot_row != col) {
            for (int j = 0; j < n; j++) {
                float tmp = a[col * lda + j];
                a[col * lda + j] = a[pivot_row * lda + j];
                a[pivot_row * lda + j] = tmp;
            }
            for (int j = 0; j < nrhs; j++) {
                float tmp = b[col * ldb + j];
                b[col * ldb + j] = b[pivot_row * ldb + j];
                b[pivot_row * ldb + j] = tmp;
            }
        }

        /* Eliminate below */
        float pivot = a[col * lda + col];
        for (int row = col + 1; row < n; row++) {
            float factor = a[row * lda + col] / pivot;
            for (int j = col; j < n; j++) {
                a[row * lda + j] -= factor * a[col * lda + j];
            }
            for (int j = 0; j < nrhs; j++) {
                b[row * ldb + j] -= factor * b[col * ldb + j];
            }
        }
    }

    /* Back substitution */
    for (int col = n - 1; col >= 0; col--) {
        for (int j = 0; j < nrhs; j++) {
            float sum = b[col * ldb + j];
            for (int k = col + 1; k < n; k++) {
                sum -= a[col * lda + k] * b[k * ldb + j];
            }
            b[col * ldb + j] = sum / a[col * lda + col];
        }
    }

    return 0;
}

/* ──────────────────────────────────────────────────────────
 * Neura-X Specific: Fractal Bloom
 * ────────────────────────────────────────────────────────── */

void nex_fractal_bloom_row(
    int row, int cols, int K,
    const float *a, const float *w1, const float *w2, const float *phi,
    float *output)
{
    if (!a || !w1 || !w2 || !phi || !output || cols <= 0 || K <= 0) return;

    memset(output, 0, (size_t)cols * sizeof(float));

    float row_f = (float)row;

    for (int k = 0; k < K; k++) {
        float base = w1[k] * row_f + phi[k];
        for (int j = 0; j < cols; j++) {
            output[j] += a[k] * sinf(base + w2[k] * (float)j);
        }
    }
}

void nex_fractal_bloom_block(
    int row_start, int row_end, int cols, int K,
    const float *a, const float *w1, const float *w2, const float *phi,
    float *output)
{
    if (!output || row_end <= row_start) return;

    for (int i = row_start; i < row_end; i++) {
        nex_fractal_bloom_row(i, cols, K, a, w1, w2, phi,
                              output + (i - row_start) * cols);
    }
}

/* ──────────────────────────────────────────────────────────
 * Neura-X Specific: Softmax
 * ────────────────────────────────────────────────────────── */

void nex_softmax_f32(const float *input, float *output, int n)
{
    if (!input || !output || n <= 0) return;

    /* Find max for numerical stability */
    float max_val = input[0];
    for (int i = 1; i < n; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    /* Compute exp and sum */
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    /* Normalize */
    if (sum > 1e-10f) {
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < n; i++) {
            output[i] *= inv_sum;
        }
    }
}

/* ──────────────────────────────────────────────────────────
 * Neura-X Specific: Gumbel-Softmax Gate
 * ────────────────────────────────────────────────────────── */

void nex_gumbel_softmax_gate(
    const float *logits, float temperature,
    float *gates, int n, bool hard)
{
    if (!logits || !gates || n <= 0) return;

    if (hard || temperature < 0.01f) {
        /* Hard binary gates */
        for (int i = 0; i < n; i++) {
            gates[i] = (logits[i] > 0.0f) ? 1.0f : 0.0f;
        }
        return;
    }

    /* Soft gates with Gumbel noise */
    float *noisy = (float *)malloc((size_t)n * sizeof(float));
    if (!noisy) return;

    /* Add Gumbel noise: -log(-log(U)) where U ~ Uniform(0,1) */
    unsigned int seed = 42;
    for (int i = 0; i < n; i++) {
        seed = seed * 1103515245 + 12345;
        float u = (float)((seed >> 16) & 0x7FFF) / 32767.0f;
        u = u * 0.999f + 0.0005f; /* Clamp to avoid log(0) */
        float gumbel = -logf(-logf(u));
        noisy[i] = (logits[i] + gumbel) / temperature;
    }

    /* Softmax */
    nex_softmax_f32(noisy, gates, n);

    free(noisy);
}