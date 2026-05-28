#include "sheaf_laplacian.cuh"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT_EQ(a, b, tol, msg)                                               \
    do {                                                                         \
        float _a = (a), _b = (b), _tol = (tol);                                \
        if (fabsf(_a - _b) > _tol) {                                            \
            fprintf(stderr, "FAIL %s: %.6f != %.6f (tol=%.6f)\n",              \
                    msg, _a, _b, _tol);                                         \
            tests_failed++;                                                      \
        } else {                                                                 \
            tests_passed++;                                                      \
        }                                                                        \
    } while (0)

#define ASSERT_TRUE(cond, msg)                                                   \
    do {                                                                         \
        if (!(cond)) {                                                           \
            fprintf(stderr, "FAIL %s\n", msg);                                  \
            tests_failed++;                                                      \
        } else {                                                                 \
            tests_passed++;                                                      \
        }                                                                        \
    } while (0)

// ── Helper: compute CPU distance ─────────────────────────────────
static float cpu_dist(const float* pts, int N, int D, int i, int j) {
    float sum = 0;
    for (int d = 0; d < D; d++) {
        float diff = pts[i * D + d] - pts[j * D + d];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

// ── Test 1: Distance matrix — 4 points in 3D ────────────────────
void test_distance_4pt() {
    printf("Test: Distance matrix 4 points in 3D\n");
    const int N = 4, D = 3;
    float h_pts[N * D] = {
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    };
    float *d_pts, *d_dist;
    CUDA_CHECK(cudaMalloc(&d_pts, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dist, N * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_pts, h_pts, N * D * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(compute_distance_matrix(d_pts, d_dist, N, D));
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_dist[N * N];
    CUDA_CHECK(cudaMemcpy(h_dist, d_dist, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float expected = cpu_dist(h_pts, N, D, i, j);
            ASSERT_EQ(h_dist[i * N + j], expected, 1e-4f,
                      "dist(i,j) where i,j");
        }
    }

    CUDA_CHECK(cudaFree(d_pts));
    CUDA_CHECK(cudaFree(d_dist));
}

// ── Test 2: Adjacency — correct edge count ──────────────────────
void test_adjacency_edges() {
    printf("Test: Adjacency edge count\n");
    const int N = 4, D = 3;
    float h_pts[N * D] = {0,0,0, 1,0,0, 0,1,0, 5,5,5};
    float *d_pts, *d_dist;
    CUDA_CHECK(cudaMalloc(&d_pts, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dist, N * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_pts, h_pts, N * D * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(compute_distance_matrix(d_pts, d_dist, N, D));
    CUDA_CHECK(cudaDeviceSynchronize());

    // epsilon = 1.5 → edges: (0,1), (0,2), (1,0), (1,2), (2,0), (2,1) = 6
    int* d_row_ptr; int* d_col_idx; float* d_weights; int nnz;
    CUDA_CHECK(build_adjacency_csr(d_dist, N, 1.5f, &d_row_ptr, &d_col_idx, &d_weights, &nnz));
    ASSERT_EQ((float)nnz, 6.0f, 0.5f, "adjacency nnz");

    CUDA_CHECK(cudaFree(d_pts));
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_row_ptr));
    if (nnz > 0) {
        CUDA_CHECK(cudaFree(d_col_idx));
        CUDA_CHECK(cudaFree(d_weights));
    }
}

// ── Test 3: Laplacian — 3-node chain ────────────────────────────
void test_laplacian_chain() {
    printf("Test: Laplacian 3-node chain\n");
    const int N = 3, D = 1;
    // Chain: 0-1-2 with points at 0, 1, 2
    float h_pts[N * D] = {0, 1, 2};
    float *d_pts, *d_dist;
    CUDA_CHECK(cudaMalloc(&d_pts, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dist, N * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_pts, h_pts, N * D * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(compute_distance_matrix(d_pts, d_dist, N, D));

    int* d_row_ptr; int* d_col_idx; float* d_weights; int nnz;
    // epsilon = 1.1 → edges: (0,1), (1,0), (1,2), (2,1)
    CUDA_CHECK(build_adjacency_csr(d_dist, N, 1.1f, &d_row_ptr, &d_col_idx, &d_weights, &nnz));
    ASSERT_EQ((float)nnz, 4.0f, 0.5f, "chain nnz");

    float* d_lap;
    CUDA_CHECK(cudaMalloc(&d_lap, N * N * sizeof(float)));
    CUDA_CHECK(build_sheaf_laplacian(d_dist, d_row_ptr, d_col_idx, d_weights,
                                     N, nnz, d_lap));
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_lap[N * N];
    CUDA_CHECK(cudaMemcpy(h_lap, d_lap, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Diagonal should be positive, off-diagonal negative or zero
    // L[0,2] and L[2,0] should be 0 (no edge)
    ASSERT_EQ(h_lap[0 * N + 2], 0.0f, 1e-5f, "L[0,2]==0");
    ASSERT_EQ(h_lap[2 * N + 0], 0.0f, 1e-5f, "L[2,0]==0");
    // Diagonal entries should be > 0
    ASSERT_TRUE(h_lap[0 * N + 0] > 0.0f, "L[0,0] > 0");
    ASSERT_TRUE(h_lap[1 * N + 1] > 0.0f, "L[1,1] > 0");
    ASSERT_TRUE(h_lap[2 * N + 2] > 0.0f, "L[2,2] > 0");

    CUDA_CHECK(cudaFree(d_pts)); CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_row_ptr)); CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_weights)); CUDA_CHECK(cudaFree(d_lap));
}

// ── Test 4: Laplacian — fully connected ─────────────────────────
void test_laplacian_fully_connected() {
    printf("Test: Laplacian fully connected\n");
    const int N = 3, D = 2;
    float h_pts[N * D] = {0,0, 1,0, 0,1};
    float *d_pts, *d_dist;
    CUDA_CHECK(cudaMalloc(&d_pts, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dist, N * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_pts, h_pts, N * D * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(compute_distance_matrix(d_pts, d_dist, N, D));

    int* d_row_ptr; int* d_col_idx; float* d_weights; int nnz;
    // epsilon = 2.0 → fully connected: 6 edges
    CUDA_CHECK(build_adjacency_csr(d_dist, N, 2.0f, &d_row_ptr, &d_col_idx, &d_weights, &nnz));
    ASSERT_EQ((float)nnz, 6.0f, 0.5f, "fc nnz");

    float* d_lap;
    CUDA_CHECK(cudaMalloc(&d_lap, N * N * sizeof(float)));
    CUDA_CHECK(build_sheaf_laplacian(d_dist, d_row_ptr, d_col_idx, d_weights,
                                     N, nnz, d_lap));
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_lap[N * N];
    CUDA_CHECK(cudaMemcpy(h_lap, d_lap, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    // All off-diagonals should be non-zero
    ASSERT_TRUE(h_lap[0 * N + 2] != 0.0f, "L[0,2] != 0");
    ASSERT_TRUE(h_lap[2 * N + 0] != 0.0f, "L[2,0] != 0");

    CUDA_CHECK(cudaFree(d_pts)); CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_row_ptr)); CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_weights)); CUDA_CHECK(cudaFree(d_lap));
}

// ── Test 5: Eigenvalues of identity matrix ──────────────────────
void test_eigenvalues_identity() {
    printf("Test: Eigenvalues of identity matrix\n");
    const int N = 5, K = 5;
    float h_mat[N * N] = {};
    for (int i = 0; i < N; i++) h_mat[i * N + i] = 1.0f;

    float *d_mat, *d_eig;
    CUDA_CHECK(cudaMalloc(&d_mat, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_eig, K * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_mat, h_mat, N * N * sizeof(float), cudaMemcpyHostToDevice));

    srand(42);
    CUDA_CHECK(compute_eigenvalues(d_mat, N, K, d_eig, 200, 1e-5f));
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_eig[K];
    CUDA_CHECK(cudaMemcpy(h_eig, d_eig, K * sizeof(float), cudaMemcpyDeviceToHost));

    // All eigenvalues should be ~1.0
    for (int i = 0; i < K; i++) {
        ASSERT_EQ(h_eig[i], 1.0f, 0.05f, "identity eigenvalue");
    }

    CUDA_CHECK(cudaFree(d_mat)); CUDA_CHECK(cudaFree(d_eig));
}

// ── Test 6: Spectral invariants ─────────────────────────────────
void test_spectral_invariants() {
    printf("Test: Spectral invariants\n");
    const int n = 5;
    float h_eig[n] = {5.0f, 3.0f, 2.0f, 1.0f, 0.5f};

    float *d_eig, *d_sr, *d_sg, *d_spread, *d_trace, *d_min;
    CUDA_CHECK(cudaMalloc(&d_eig, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sr, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sg, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_spread, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_trace, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_min, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_eig, h_eig, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(compute_spectral_invariants(d_eig, n, d_sr, d_sg, d_spread, d_trace, d_min));
    CUDA_CHECK(cudaDeviceSynchronize());

    float sr, sg, spread, trace, mn;
    CUDA_CHECK(cudaMemcpy(&sr, d_sr, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sg, d_sg, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&spread, d_spread, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&trace, d_trace, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&mn, d_min, sizeof(float), cudaMemcpyDeviceToHost));

    ASSERT_EQ(sr, 5.0f, 1e-4f, "spectral_radius");
    ASSERT_EQ(sg, 2.0f, 1e-4f, "spectral_gap");  // 5 - 3
    ASSERT_EQ(spread, 4.5f, 1e-4f, "spread");     // 5.0 - 0.5
    ASSERT_EQ(trace, 11.5f, 1e-4f, "trace");       // sum
    ASSERT_EQ(mn, 0.5f, 1e-4f, "min_eigenvalue");

    CUDA_CHECK(cudaFree(d_eig)); CUDA_CHECK(cudaFree(d_sr));
    CUDA_CHECK(cudaFree(d_sg)); CUDA_CHECK(cudaFree(d_spread));
    CUDA_CHECK(cudaFree(d_trace)); CUDA_CHECK(cudaFree(d_min));
}

// ── Test 7: Scaling N=100 ──────────────────────────────────────
void test_scaling_100() {
    printf("Test: Scaling N=100\n");
    const int N = 100, D = 5;
    std::vector<float> h_pts(N * D);
    srand(123);
    for (auto& v : h_pts) v = (float)(rand() % 100) / 10.0f;

    float *d_pts, *d_dist;
    CUDA_CHECK(cudaMalloc(&d_pts, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dist, N * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_pts, h_pts.data(), N * D * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(compute_distance_matrix(d_pts, d_dist, N, D));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify a few distances against CPU
    float h_dist_01, expected;
    CUDA_CHECK(cudaMemcpy(&h_dist_01, d_dist + 1, sizeof(float), cudaMemcpyDeviceToHost));
    expected = cpu_dist(h_pts.data(), N, D, 0, 1);
    ASSERT_EQ(h_dist_01, expected, 1e-3f, "N=100 dist(0,1)");

    // Build adjacency + Laplacian + eigenvalues (top 5)
    int* d_row_ptr; int* d_col_idx; float* d_weights; int nnz;
    CUDA_CHECK(build_adjacency_csr(d_dist, N, 3.0f, &d_row_ptr, &d_col_idx, &d_weights, &nnz));
    ASSERT_TRUE(nnz > 0, "N=100 has edges");

    float* d_lap;
    CUDA_CHECK(cudaMalloc(&d_lap, N * N * sizeof(float)));
    CUDA_CHECK(build_sheaf_laplacian(d_dist, d_row_ptr, d_col_idx, d_weights, N, nnz, d_lap));

    const int K = 5;
    float* d_eig;
    CUDA_CHECK(cudaMalloc(&d_eig, K * sizeof(float)));
    srand(456);
    CUDA_CHECK(compute_eigenvalues(d_lap, N, K, d_eig));
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_eig[K];
    CUDA_CHECK(cudaMemcpy(h_eig, d_eig, K * sizeof(float), cudaMemcpyDeviceToHost));
    // Eigenvalues should be real and finite
    for (int i = 0; i < K; i++) {
        ASSERT_TRUE(isfinite(h_eig[i]), "N=100 eigenvalue finite");
    }

    CUDA_CHECK(cudaFree(d_pts)); CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_row_ptr)); CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_weights)); CUDA_CHECK(cudaFree(d_lap));
    CUDA_CHECK(cudaFree(d_eig));
}

// ── Test 8: Scaling N=500 ──────────────────────────────────────
void test_scaling_500() {
    printf("Test: Scaling N=500\n");
    const int N = 500, D = 8;
    std::vector<float> h_pts(N * D);
    srand(789);
    for (auto& v : h_pts) v = (float)(rand() % 100) / 10.0f;

    float *d_pts, *d_dist;
    CUDA_CHECK(cudaMalloc(&d_pts, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dist, N * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_pts, h_pts.data(), N * D * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(compute_distance_matrix(d_pts, d_dist, N, D));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Just verify diagonal is 0
    float h_diag;
    CUDA_CHECK(cudaMemcpy(&h_diag, d_dist, sizeof(float), cudaMemcpyDeviceToHost));
    ASSERT_EQ(h_diag, 0.0f, 1e-4f, "N=500 diagonal(0)");

    // Adjacency
    int* d_row_ptr; int* d_col_idx; float* d_weights; int nnz;
    CUDA_CHECK(build_adjacency_csr(d_dist, N, 2.0f, &d_row_ptr, &d_col_idx, &d_weights, &nnz));
    ASSERT_TRUE(nnz > 0, "N=500 has edges");

    // Spectral invariants on a small eigenvalue set
    const int K = 3;
    float* d_lap;
    CUDA_CHECK(cudaMalloc(&d_lap, N * N * sizeof(float)));
    CUDA_CHECK(build_sheaf_laplacian(d_dist, d_row_ptr, d_col_idx, d_weights, N, nnz, d_lap));

    float* d_eig;
    CUDA_CHECK(cudaMalloc(&d_eig, K * sizeof(float)));
    srand(999);
    CUDA_CHECK(compute_eigenvalues(d_lap, N, K, d_eig));
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_eig[K];
    CUDA_CHECK(cudaMemcpy(h_eig, d_eig, K * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < K; i++) {
        ASSERT_TRUE(isfinite(h_eig[i]), "N=500 eigenvalue finite");
    }

    // Spectral invariants
    float *d_sr, *d_sg, *d_spread, *d_trace, *d_min;
    CUDA_CHECK(cudaMalloc(&d_sr, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sg, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_spread, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_trace, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_min, sizeof(float)));

    CUDA_CHECK(compute_spectral_invariants(d_eig, K, d_sr, d_sg, d_spread, d_trace, d_min));
    CUDA_CHECK(cudaDeviceSynchronize());

    float sr;
    CUDA_CHECK(cudaMemcpy(&sr, d_sr, sizeof(float), cudaMemcpyDeviceToHost));
    ASSERT_TRUE(sr >= 0.0f, "N=500 spectral_radius >= 0");

    CUDA_CHECK(cudaFree(d_pts)); CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_row_ptr)); CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_weights)); CUDA_CHECK(cudaFree(d_lap));
    CUDA_CHECK(cudaFree(d_eig));
    CUDA_CHECK(cudaFree(d_sr)); CUDA_CHECK(cudaFree(d_sg));
    CUDA_CHECK(cudaFree(d_spread)); CUDA_CHECK(cudaFree(d_trace));
    CUDA_CHECK(cudaFree(d_min));
}

int main() {
    CUDA_CHECK(cudaSetDevice(0));
    printf("═══ gpu-sheaf-laplacian test suite ═══\n\n");

    test_distance_4pt();
    test_adjacency_edges();
    test_laplacian_chain();
    test_laplacian_fully_connected();
    test_eigenvalues_identity();
    test_spectral_invariants();
    test_scaling_100();
    test_scaling_500();

    printf("\n═══ Results: %d passed, %d failed ═══\n",
           tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
