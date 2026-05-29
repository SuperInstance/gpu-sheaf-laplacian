# gpu-sheaf-laplacian

**CUDA sheaf Laplacian computation — distance matrices, CSR adjacency, spectral invariants, and power iteration eigenvalues, all on GPU.**

Builds sheaf Laplacians from point cloud data entirely on the GPU. Computes tiled pairwise distances, epsilon-threshold CSR adjacency, Gaussian kernel weights, and the sheaf Laplacian (which encodes both geometric distance and stalk restriction maps). Extracts spectral invariants: spectral radius, gap, eigenvalue spread, trace.

## What This Gives You

- **Tiled distance matrix** — pairwise Euclidean distances via shared-memory tiling
- **CSR adjacency** — epsilon-threshold with Gaussian kernel weights
- **Sheaf Laplacian** — L_F = D - W modified by stalk features
- **Power iteration eigenvalues** — top-k eigenvalues with deflation
- **Spectral invariants** — radius, gap, spread, trace, all on GPU
- **Benchmarking suite** — GPU vs CPU scaling comparison

## Quick Start

```cuda
#include "sheaf_laplacian.cuh"

// Distance matrix
compute_distance_matrix(d_points, d_dist, N, D, stream);

// Adjacency (CSR)
build_adjacency_csr(d_dist, N, epsilon, &d_row_ptr, &d_col_idx, &d_weights, &nnz, stream);

// Sheaf Laplacian
build_sheaf_laplacian(d_dist, d_row_ptr, d_col_idx, d_weights, N, nnz, d_laplacian, stream);

// Eigenvalues
compute_eigenvalues(d_laplacian, N, k, d_eigenvalues, stream);

// Spectral invariants
compute_spectral_invariants(d_eigenvalues, k, &radius, &gap, &spread, &trace, stream);
```

## Build

```bash
nvcc -O3 -o test_correctness tests/test_correctness.cu src/*.cu
nvcc -O3 -o benchmark bench/benchmark.cu src/*.cu
./test_correctness && ./benchmark
```

## How It Fits

Part of the SuperInstance ecosystem:

- **[persistent-sheaf](https://github.com/SuperInstance/persistent-sheaf)** — Rust sheaf cohomology
- **gpu-sheaf-laplacian** — CUDA-accelerated sheaf Laplacian (this repo)

## License

MIT
