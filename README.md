# gpu-sheaf-laplacian

CUDA library for computing **sheaf Laplacians** on GPU, built for the RTX 4050 (sm_89).

Implements the full pipeline from point cloud → distance matrix → adjacency → sheaf Laplacian → eigenvalues → spectral invariants.

Based on experimental results (R²=0.993) showing continuous spectral invariants predict MoE generalization.

## Pipeline

1. **Distance Matrix** — Tiled pairwise Euclidean distances using shared memory
2. **Adjacency** — ε-threshold sparse adjacency in CSR format with Gaussian kernel weights
3. **Sheaf Laplacian** — Constructs restriction maps from stalk similarity (scalar stalks)
4. **Eigenvalues** — Power iteration with deflation for top-k eigenvalues
5. **Spectral Invariants** — Spectral radius, gap, spread, trace, minimum eigenvalue

## Requirements

- CUDA 12.6+
- RTX 4050 (sm_89) or compatible GPU
- nvcc with C++17 support

## Build & Test

```bash
make test
```

## Architecture

```
include/sheaf_laplacian.cuh     — Public API header
src/distance_matrix.cu          — Tiled pairwise distances
src/adjacency.cu                — ε-threshold CSR adjacency
src/laplacian.cu                — Sheaf Laplacian construction
src/eigenvalues.cu              — Power iteration + deflation
src/spectral_invariants.cu      — Spectral invariants
tests/test_correctness.cu       — 8 test cases
```

## Usage

```cpp
#include "sheaf_laplacian.cuh"

// 1. Distance matrix
compute_distance_matrix(d_points, d_dist, N, D);

// 2. Adjacency
build_adjacency_csr(d_dist, N, epsilon, &d_row_ptr, &d_col_idx, &d_weights, &nnz);

// 3. Sheaf Laplacian
build_sheaf_laplacian(d_dist, d_row_ptr, d_col_idx, d_weights, N, nnz, d_laplacian);

// 4. Eigenvalues (top-k)
compute_eigenvalues(d_laplacian, N, k, d_eigenvalues);

// 5. Spectral invariants
compute_spectral_invariants(d_eigenvalues, k, &sr, &sg, &spread, &trace, &min_eig);
```

## License

MIT

Part of the [SuperInstance OpenConstruct](https://github.com/SuperInstance/OpenConstruct) ecosystem.
