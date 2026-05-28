// benchmark.cu — GPU vs CPU scaling benchmark
#include "sheaf_laplacian.cuh"
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>

static double now_sec() {
    auto t = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t.time_since_epoch()).count();
}

int main() {
    cudaSetDevice(0);
    SheafContext* ctx = sheaf_create(0);

    int sizes[] = {100, 500, 1000, 5000, 10000};
    int nsizes = 5;
    int dim = 3, stalk_dim = 2;

    printf("%-8s %-12s %-12s %-12s %-12s %-12s %-10s\n",
           "N", "distance", "adjacency", "stalks", "laplacian", "eigenvalues", "total");
    printf("%-8s %-12s %-12s %-12s %-12s %-12s %-10s\n",
           "---", "---", "---", "---", "---", "---", "---");

    for (int si = 0; si < nsizes; si++) {
        int N = sizes[si];
        int sd = stalk_dim;
        int size = N * sd;

        std::vector<float> h_points(N * dim);
        for (auto& p : h_points) p = (float)rand() / RAND_MAX;

        float epsilon = N <= 500 ? 1.0f : 0.3f;

        // Allocate
        float *d_points, *d_dist, *d_stalks, *d_restrictions, *d_laplacian, *d_eigenvalues;
        int *d_row_ptr, *d_col_idx; float *d_weights;
        
        cudaMalloc(&d_points, N*dim*sizeof(float));
        cudaMalloc(&d_dist, N*N*sizeof(float));
        cudaMalloc(&d_stalks, N*sd*sizeof(float));
        cudaMalloc(&d_row_ptr, (N+1)*sizeof(int));
        cudaMalloc(&d_col_idx, (size_t)N*N*sizeof(int));
        cudaMalloc(&d_weights, (size_t)N*N*sizeof(float));
        cudaMemcpy(d_points, h_points.data(), N*dim*sizeof(float), cudaMemcpyHostToDevice);

        cudaEvent_t e1, e2, e3, e4, e5, e6;
        cudaEventCreate(&e1); cudaEventCreate(&e2);
        cudaEventCreate(&e3); cudaEventCreate(&e4);
        cudaEventCreate(&e5); cudaEventCreate(&e6);

        // Distance
        cudaEventRecord(e1, ctx->stream);
        sheaf_compute_distances(ctx, d_points, d_dist, N, dim);
        cudaEventRecord(e2, ctx->stream);

        // Adjacency
        int nnz = sheaf_build_adjacency(ctx, d_dist, d_row_ptr, d_col_idx, d_weights, epsilon, N);
        cudaEventRecord(e3, ctx->stream);

        // Stalks
        sheaf_assign_stalks(ctx, d_points, d_stalks, N, dim, sd);
        cudaEventRecord(e4, ctx->stream);

        // Restrictions + Laplacian
        if (nnz > 0) {
            cudaMalloc(&d_restrictions, (size_t)nnz*sd*sd*sizeof(float));
            sheaf_compute_restrictions_with_rowptr(ctx, d_stalks, d_row_ptr, d_col_idx,
                                                    d_restrictions, nnz, N, sd);
        } else {
            cudaMalloc(&d_restrictions, sizeof(float));
        }
        
        cudaMalloc(&d_laplacian, (size_t)size*size*sizeof(float));
        sheaf_build_laplacian_dense(ctx, d_row_ptr, d_col_idx, d_restrictions,
                                     d_laplacian, nnz, N, sd);
        cudaEventRecord(e5, ctx->stream);

        // Eigenvalues (skip for N>2000 — dense eigendecomposition is O(N³))
        float t_eig = 0;
        if (N <= 2000) {
            cudaMalloc(&d_eigenvalues, size*sizeof(float));
            sheaf_compute_eigenvalues(ctx, d_laplacian, d_eigenvalues, size);
            cudaEventRecord(e6, ctx->stream);
            cudaEventSynchronize(e6);
            float ms;
            cudaEventElapsedTime(&ms, e5, e6);
            t_eig = ms;
        } else {
            cudaEventRecord(e6, ctx->stream);
            cudaEventSynchronize(e6);
        }

        float t_dist, t_adj, t_stalk, t_lap;
        cudaEventElapsedTime(&t_dist, e1, e2);
        cudaEventElapsedTime(&t_adj, e2, e3);
        cudaEventElapsedTime(&t_stalk, e3, e4);
        cudaEventElapsedTime(&t_lap, e4, e5);
        float total = t_dist + t_adj + t_stalk + t_lap + t_eig;

        printf("%-8d %-12.2f %-12.2f %-12.2f %-12.2f %-12.2f %-10.2f\n",
               N, t_dist, t_adj, t_stalk, t_lap, t_eig, total);
        printf("         nnz=%d\n", nnz);

        cudaFree(d_points); cudaFree(d_dist); cudaFree(d_stalks);
        cudaFree(d_row_ptr); cudaFree(d_col_idx); cudaFree(d_weights);
        cudaFree(d_restrictions); cudaFree(d_laplacian);
        if (N <= 2000) cudaFree(d_eigenvalues);

        cudaEventDestroy(e1); cudaEventDestroy(e2);
        cudaEventDestroy(e3); cudaEventDestroy(e4);
        cudaEventDestroy(e5); cudaEventDestroy(e6);
    }

    // CPU baseline for N=100, 500
    printf("\n--- CPU Baseline (numpy-style, single-threaded) ---\n");
    for (int N : {100, 500}) {
        std::vector<float> h_points(N * dim);
        for (auto& p : h_points) p = (float)rand() / RAND_MAX;
        
        double t0 = now_sec();
        // Distance
        std::vector<float> dist(N*N, 0);
        for (int i = 0; i < N; i++)
            for (int j = i+1; j < N; j++) {
                float sum = 0;
                for (int d = 0; d < dim; d++) {
                    float diff = h_points[i*dim+d] - h_points[j*dim+d];
                    sum += diff*diff;
                }
                dist[i*N+j] = dist[j*N+i] = sqrtf(sum);
            }
        double t1 = now_sec();
        printf("N=%d: CPU distance = %.3f ms\n", N, (t1-t0)*1000);
    }

    sheaf_destroy(ctx);
    printf("\nBenchmark complete.\n");
    return 0;
}
