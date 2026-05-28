#include "sheaf_laplacian.cuh"

#define TILE 16

__global__ void distance_kernel(const float* __restrict__ points,
                                float* __restrict__ dist,
                                int N, int D) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;

    float sum = 0.0f;
    for (int d = 0; d < D; d++) {
        float diff = points[row * D + d] - points[col * D + d];
        sum += diff * diff;
    }
    dist[row * N + col] = sqrtf(sum);
}

cudaError_t compute_distance_matrix(const float* d_points, float* d_dist,
                                     int N, int D, cudaStream_t stream) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    distance_kernel<<<grid, block, 0, stream>>>(d_points, d_dist, N, D);
    return cudaGetLastError();
}
