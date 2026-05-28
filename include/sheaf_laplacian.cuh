#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

// ── Distance matrix ──────────────────────────────────────────────
// Computes tiled pairwise Euclidean distances.
// points: [N x D] row-major, output: [N x N] symmetric distance matrix
cudaError_t compute_distance_matrix(const float* d_points, float* d_dist,
                                     int N, int D, cudaStream_t stream = 0);

// ── Adjacency (epsilon-threshold) ────────────────────────────────
// Builds sparse adjacency in CSR format from distance matrix + threshold.
// Returns CSR arrays in device memory. Caller must free *d_row_ptr, *d_col_idx, *d_weights.
cudaError_t build_adjacency_csr(const float* d_dist, int N, float epsilon,
                                int** d_row_ptr, int** d_col_idx, float** d_weights,
                                int* nnz_out, cudaStream_t stream = 0);

// ── Sheaf Laplacian ──────────────────────────────────────────────
// Builds dense sheaf Laplacian [N x N] from distance matrix + adjacency.
// Uses stalk features = rows of the distance matrix.
// d_laplacian must be pre-allocated [N x N].
cudaError_t build_sheaf_laplacian(const float* d_dist, const int* d_row_ptr,
                                  const int* d_col_idx, const float* d_weights,
                                  int N, int nnz, float* d_laplacian,
                                  cudaStream_t stream = 0);

// ── Eigenvalues (power iteration + deflation) ────────────────────
// Computes top-k eigenvalues of a symmetric N x N matrix.
// d_eigenvalues must be pre-allocated [k].
cudaError_t compute_eigenvalues(const float* d_matrix, int N, int k,
                                float* d_eigenvalues, int max_iter = 200,
                                float tol = 1e-5f, cudaStream_t stream = 0);

// ── Spectral invariants ─────────────────────────────────────────
cudaError_t compute_spectral_invariants(const float* d_eigenvalues, int n,
    float* d_spectral_radius, float* d_spectral_gap,
    float* d_eigenvalue_spread, float* d_trace,
    float* d_min_eigenvalue, cudaStream_t stream = 0);

// ── CUDA check macro ─────────────────────────────────────────────
#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                  \
        }                                                                       \
    } while (0)
