#include "sheaf_laplacian.cuh"

// ── Build sheaf Laplacian on host from device CSR ────────────────
// For N ≤ 256 we construct the dense Laplacian on host then copy back.
// This keeps the restriction map logic simple and correct.

// Stalk for vertex i: use the i-th row of the distance matrix (dim N).
// For a smaller stalk, we truncate to stalk_dim features.

__global__ void zero_matrix_kernel(float* mat, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) mat[idx] = 0.0f;
}

cudaError_t build_sheaf_laplacian(const float* d_dist, const int* d_row_ptr,
                                  const int* d_col_idx, const float* d_weights,
                                  int N, int nnz, float* d_laplacian,
                                  cudaStream_t stream) {
    // Zero out the Laplacian
    int total = N * N;
    zero_matrix_kernel<<<(total + 255) / 256, 256, 0, stream>>>(d_laplacian, total);
    CUDA_CHECK(cudaGetLastError());

    // Copy CSR + distance to host for the restriction map construction
    int* h_row_ptr = new int[N + 1];
    int* h_col_idx = nullptr;
    float* h_weights = nullptr;
    float* h_dist = new float[N * N];

    CUDA_CHECK(cudaMemcpyAsync(h_row_ptr, d_row_ptr, (N + 1) * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    if (nnz > 0) {
        h_col_idx = new int[nnz];
        h_weights = new float[nnz];
        CUDA_CHECK(cudaMemcpyAsync(h_col_idx, d_col_idx, nnz * sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_weights, d_weights, nnz * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(h_dist, d_dist, N * N * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Build Laplacian on host
    // Stalk for vertex i = row i of distance matrix
    // Restriction map R_ij: projection of stalk_i onto stalk_j similarity
    //   R_ij = <s_i, s_j> / |s_i|^2  (scalar restriction for 1D stalks)
    // For simplicity: R_ij = weight of edge (i,j) normalized
    // L_sheaf[i,i] += I - R_ij^T R_ij  (we use scalar stalks, so just 1 - R^2)
    // L_sheaf[i,j] = -R_ij * R_ji

    float* h_lap = new float[N * N]();

    for (int i = 0; i < N; i++) {
        for (int e = h_row_ptr[i]; e < h_row_ptr[i + 1]; e++) {
            int j = h_col_idx[e];
            float w_ij = h_weights[e];  // R_ij approximation

            // Compute R_ji: find the reverse edge weight
            float w_ji = 0.0f;
            for (int e2 = h_row_ptr[j]; e2 < h_row_ptr[j + 1]; e2++) {
                if (h_col_idx[e2] == i) {
                    w_ji = h_weights[e2];
                    break;
                }
            }

            // Sheaf Laplacian entries (scalar stalk)
            // Diagonal: sum of (1 - R_ij^2)
            h_lap[i * N + i] += 1.0f - w_ij * w_ij;
            // Off-diagonal: -R_ij * R_ji
            h_lap[i * N + j] = -w_ij * w_ji;
        }
    }

    // Copy to device
    CUDA_CHECK(cudaMemcpyAsync(d_laplacian, h_lap, N * N * sizeof(float),
                               cudaMemcpyHostToDevice, stream));

    delete[] h_row_ptr;
    delete[] h_col_idx;
    delete[] h_weights;
    delete[] h_dist;
    delete[] h_lap;

    return cudaSuccess;
}
