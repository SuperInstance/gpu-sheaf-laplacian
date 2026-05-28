#include "sheaf_laplacian.cuh"

// ── Count edges per vertex ───────────────────────────────────────
__global__ void count_edges_kernel(const float* __restrict__ dist, int N,
                                   float epsilon, int* __restrict__ row_counts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int count = 0;
    for (int j = 0; j < N; j++) {
        if (i != j && dist[i * N + j] <= epsilon) count++;
    }
    row_counts[i] = count;
}

// ── Fill CSR arrays ──────────────────────────────────────────────
__global__ void fill_csr_kernel(const float* __restrict__ dist, int N,
                                float epsilon,
                                const int* __restrict__ row_offsets,
                                int* __restrict__ col_idx,
                                float* __restrict__ weights) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int pos = row_offsets[i];
    for (int j = 0; j < N; j++) {
        if (i != j && dist[i * N + j] <= epsilon) {
            col_idx[pos] = j;
            // Weight = Gaussian kernel: exp(-d^2 / (2 * epsilon^2))
            float d = dist[i * N + j];
            weights[pos] = expf(-(d * d) / (2.0f * epsilon * epsilon));
            pos++;
        }
    }
}

cudaError_t build_adjacency_csr(const float* d_dist, int N, float epsilon,
                                int** d_row_ptr, int** d_col_idx, float** d_weights,
                                int* nnz_out, cudaStream_t stream) {
    // Allocate and count edges per row
    int* d_counts;
    CUDA_CHECK(cudaMalloc(&d_counts, N * sizeof(int)));

    int blocks = (N + 255) / 256;
    count_edges_kernel<<<blocks, 256, 0, stream>>>(d_dist, N, epsilon, d_counts);
    CUDA_CHECK(cudaGetLastError());

    // Prefix sum for row offsets (on host for simplicity)
    int* h_counts = new int[N];
    CUDA_CHECK(cudaMemcpyAsync(h_counts, d_counts, N * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int* h_offsets = new int[N + 1];
    h_offsets[0] = 0;
    int total = 0;
    for (int i = 0; i < N; i++) {
        total += h_counts[i];
        h_offsets[i + 1] = total;
    }
    *nnz_out = total;

    // Allocate CSR arrays
    CUDA_CHECK(cudaMalloc(d_row_ptr, (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpyAsync(*d_row_ptr, h_offsets, (N + 1) * sizeof(int),
                               cudaMemcpyHostToDevice, stream));

    if (total > 0) {
        CUDA_CHECK(cudaMalloc(d_col_idx, total * sizeof(int)));
        CUDA_CHECK(cudaMalloc(d_weights, total * sizeof(float)));
        fill_csr_kernel<<<blocks, 256, 0, stream>>>(d_dist, N, epsilon,
                                                     *d_row_ptr, *d_col_idx, *d_weights);
        CUDA_CHECK(cudaGetLastError());
    } else {
        *d_col_idx = nullptr;
        *d_weights = nullptr;
    }

    delete[] h_counts;
    delete[] h_offsets;
    CUDA_CHECK(cudaFree(d_counts));

    return cudaSuccess;
}
