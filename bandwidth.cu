#include "bandwidth.h"
#include "cuda_utils.cuh"

#include <cstdlib>

// Measure bandwidth for a single direction and memory pairing
static double measure_bandwidth_single_direction(
    void* dst,
    const void* src,
    size_t bytes,
    cudaMemcpyKind kind,
    int num_iters)
{
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warm up to avoid first-use overhead
    CHECK_CUDA(cudaMemcpy(dst, src, bytes, kind));

    CHECK_CUDA(cudaEventRecord(start, 0));
    for (int i = 0; i < num_iters; ++i) {
        CHECK_CUDA(cudaMemcpy(dst, src, bytes, kind));
    }
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    double seconds = ms / 1000.0;
    double total_bytes = static_cast<double>(bytes) * num_iters;

    // Bandwidth in GB/s
    double gb_per_s = total_bytes / (seconds * 1.0e9);
    return gb_per_s;
}

// Run test for one size, filling in the 4 bandwidth numbers
static void run_test_for_size(size_t bytes,
                              int num_iters,
                              double& h2d_pageable,
                              double& d2h_pageable,
                              double& h2d_pinned,
                              double& d2h_pinned)
{
    // -----------------------------
    // Pageable host memory
    // -----------------------------
    {
        // Host allocation (pageable)
        float* h_buf = static_cast<float*>(std::malloc(bytes));
        if (!h_buf) {
            std::fprintf(stderr,
                         "Failed to allocate pageable host memory (%zu bytes)\n",
                         bytes);
            std::exit(EXIT_FAILURE);
        }

        // Initialize host data
        size_t n = bytes / sizeof(float);
        for (size_t i = 0; i < n; ++i) {
            h_buf[i] = static_cast<float>(i);
        }

        // Device allocation
        float* d_buf = nullptr;
        CHECK_CUDA(cudaMalloc(&d_buf, bytes));

        // Measure H2D pageable
        h2d_pageable = measure_bandwidth_single_direction(
            d_buf, h_buf, bytes, cudaMemcpyHostToDevice, num_iters);

        // Measure D2H pageable
        d2h_pageable = measure_bandwidth_single_direction(
            h_buf, d_buf, bytes, cudaMemcpyDeviceToHost, num_iters);

        CHECK_CUDA(cudaFree(d_buf));
        std::free(h_buf);
    }

    // -----------------------------
    // Pinned host memory
    // -----------------------------
    {
        // Host allocation (pinned)
        float* h_buf = nullptr;
        CHECK_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&h_buf),
                                 bytes,
                                 cudaHostAllocDefault));

        // Initialize host data
        size_t n = bytes / sizeof(float);
        for (size_t i = 0; i < n; ++i) {
            h_buf[i] = static_cast<float>(i);
        }

        // Device allocation
        float* d_buf = nullptr;
        CHECK_CUDA(cudaMalloc(&d_buf, bytes));

        // Measure H2D pinned
        h2d_pinned = measure_bandwidth_single_direction(
            d_buf, h_buf, bytes, cudaMemcpyHostToDevice, num_iters);

        // Measure D2H pinned
        d2h_pinned = measure_bandwidth_single_direction(
            h_buf, d_buf, bytes, cudaMemcpyDeviceToHost, num_iters);

        CHECK_CUDA(cudaFree(d_buf));
        CHECK_CUDA(cudaFreeHost(h_buf));
    }
}

void run_bandwidth_tests(const std::vector<size_t>& sizes,
                         int num_iters,
                         std::vector<BandwidthResult>& results)
{
    results.clear();
    results.reserve(sizes.size());

    for (size_t bytes : sizes) {
        double h2d_pageable = 0.0;
        double d2h_pageable = 0.0;
        double h2d_pinned = 0.0;
        double d2h_pinned = 0.0;

        run_test_for_size(bytes, num_iters,
                          h2d_pageable, d2h_pageable,
                          h2d_pinned,  d2h_pinned);

        BandwidthResult r;
        r.sizeMB       = static_cast<double>(bytes) / (1024.0 * 1024.0);
        r.h2d_pageable = h2d_pageable;
        r.d2h_pageable = d2h_pageable;
        r.h2d_pinned   = h2d_pinned;
        r.d2h_pinned   = d2h_pinned;

        results.push_back(r);
    }
}
