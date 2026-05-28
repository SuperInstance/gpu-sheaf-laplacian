#include "sheaf_laplacian.cuh"

// ── Power iteration for dominant eigenvector ─────────────────────
// Each thread processes a subset of the matrix-vector multiply.
__global__ void mat_vec_kernel(const float* __restrict__ mat,
                               const float* __restrict__ vec,
                               float* __restrict__ out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
        sum += mat[i * N + j] * vec[j];
    }
    out[i] = sum;
}

// Dot product (single-block reduction)
__global__ void dot_kernel(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ result, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    sdata[tid] = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        sdata[tid] += a[i] * b[i];
    }
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) *result = sdata[0];
}

// Normalize vector
__global__ void normalize_kernel(float* __restrict__ vec, float norm, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && norm > 1e-12f) vec[i] /= norm;
}

// Deflate: matrix -= eigval * eigvec * eigvec^T
__global__ void deflate_kernel(float* __restrict__ mat,
                               const float* __restrict__ eigvec,
                               float eigval, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N) {
        mat[i * N + j] -= eigval * eigvec[i] * eigvec[j];
    }
}

cudaError_t compute_eigenvalues(const float* d_matrix, int N, int k,
                                float* d_eigenvalues, int max_iter,
                                float tol, cudaStream_t stream) {
    // Work on a copy of the matrix (deflation modifies it)
    float* d_work;
    CUDA_CHECK(cudaMalloc(&d_work, N * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(d_work, d_matrix, N * N * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream));

    float* d_vec;
    float* d_tmp;
    float* d_dot_result;
    CUDA_CHECK(cudaMalloc(&d_vec, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tmp, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dot_result, sizeof(float)));

    dim3 block(256);
    dim3 grid_mat((N + 15) / 16, (N + 15) / 16);
    dim3 grid_vec((N + 255) / 256);
    int reduce_block = 256;
    int shared_bytes = reduce_block * sizeof(float);

    // Host-side seed for initial vectors
    float* h_vec = new float[N];

    for (int eig = 0; eig < k; eig++) {
        // Initialize random vector
        for (int i = 0; i < N; i++)
            h_vec[i] = (float)(rand() % 1000) / 500.0f - 1.0f;
        CUDA_CHECK(cudaMemcpyAsync(d_vec, h_vec, N * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));

        float prev_eigval = 0.0f;
        float eigval = 0.0f;

        for (int iter = 0; iter < max_iter; iter++) {
            // y = A * x
            mat_vec_kernel<<<grid_vec, block, 0, stream>>>(d_work, d_vec, d_tmp, N);
            CUDA_CHECK(cudaGetLastError());

            // eigval = x^T * y
            dot_kernel<<<1, reduce_block, shared_bytes, stream>>>(
                d_vec, d_tmp, d_dot_result, N);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpyAsync(&eigval, d_dot_result, sizeof(float),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // y_norm = ||y||
            float y_norm;
            dot_kernel<<<1, reduce_block, shared_bytes, stream>>>(
                d_tmp, d_tmp, d_dot_result, N);
            CUDA_CHECK(cudaMemcpyAsync(&y_norm, d_dot_result, sizeof(float),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            y_norm = sqrtf(fabsf(y_norm));

            // x = y / ||y||
            normalize_kernel<<<grid_vec, block, 0, stream>>>(d_tmp, y_norm, N);
            CUDA_CHECK(cudaMemcpyAsync(d_vec, d_tmp, N * sizeof(float),
                                       cudaMemcpyDeviceToDevice, stream));

            // Check convergence
            if (iter > 0 && fabsf(eigval - prev_eigval) < tol * fabsf(eigval + 1e-10f)) {
                break;
            }
            prev_eigval = eigval;
        }

        // Store eigenvalue
        CUDA_CHECK(cudaMemcpyAsync(d_eigenvalues + eig, &eigval, sizeof(float),
                                   cudaMemcpyHostToDevice, stream));

        // Deflate
        if (eig < k - 1) {
            // Need eigenvector on host for deflate kernel
            deflate_kernel<<<grid_mat, dim3(16, 16), 0, stream>>>(
                d_work, d_vec, eigval, N);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_tmp));
    CUDA_CHECK(cudaFree(d_dot_result));
    delete[] h_vec;

    return cudaSuccess;
}
