#include "sheaf_laplacian.cuh"

// Sort eigenvalues descending using simple bitonic-like approach on host
// then compute invariants on device.

__global__ void compute_invariants_kernel(const float* __restrict__ eigenvalues, int n,
    float* spectral_radius, float* spectral_gap,
    float* eigenvalue_spread, float* trace,
    float* min_eigenvalue) {
    // Single-thread kernel — eigenvalues array is small (k << N)
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    float sr = 0.0f, mn = eigenvalues[0], mx = eigenvalues[0], tr = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = fabsf(eigenvalues[i]);
        if (v > sr) sr = v;
        if (eigenvalues[i] < mn) mn = eigenvalues[i];
        if (eigenvalues[i] > mx) mx = eigenvalues[i];
        tr += eigenvalues[i];
    }

    // Spectral gap: difference between largest and second-largest |eigenvalue|
    float gap = 0.0f;
    if (n >= 2) {
        // Find top-2 absolute eigenvalues
        float first = 0.0f, second = 0.0f;
        for (int i = 0; i < n; i++) {
            float v = fabsf(eigenvalues[i]);
            if (v > first) { second = first; first = v; }
            else if (v > second) { second = v; }
        }
        gap = first - second;
    }

    *spectral_radius = sr;
    *spectral_gap = gap;
    *eigenvalue_spread = mx - mn;
    *trace = tr;
    *min_eigenvalue = mn;
}

cudaError_t compute_spectral_invariants(const float* d_eigenvalues, int n,
    float* d_spectral_radius, float* d_spectral_gap,
    float* d_eigenvalue_spread, float* d_trace,
    float* d_min_eigenvalue, cudaStream_t stream) {
    compute_invariants_kernel<<<1, 1, 0, stream>>>(
        d_eigenvalues, n, d_spectral_radius, d_spectral_gap,
        d_eigenvalue_spread, d_trace, d_min_eigenvalue);
    return cudaGetLastError();
}
