#include <cstdio>
#include <vector>

#include "cuda_utils.cuh"
#include "bandwidth.h"

int main(int argc, char** argv)
{
    // Use device 0 by default
    CHECK_CUDA(cudaSetDevice(0));

    // Sizes from 1 MB to 1 GB in powers of 2:
    // 1, 2, 4, ..., 1024 MB
    const int num_sizes = 11;
    std::vector<size_t> sizes;
    sizes.reserve(num_sizes);

    for (int i = 0; i < num_sizes; ++i) {
        size_t bytes = static_cast<size_t>(1) << (20 + i); // 2^(20+i)
        sizes.push_back(bytes);
    }

    // Number of repetitions per measurement
    int num_iters = 10;

    std::vector<BandwidthResult> results;
    run_bandwidth_tests(sizes, num_iters, results);

    // CSV header
    std::printf("size_MB,h2d_pageable_GBs,d2h_pageable_GBs,h2d_pinned_GBs,d2h_pinned_GBs\n");

    for (const auto& r : results) {
        std::printf("%.0f,%.3f,%.3f,%.3f,%.3f\n",
                    r.sizeMB,
                    r.h2d_pageable,
                    r.d2h_pageable,
                    r.h2d_pinned,
                    r.d2h_pinned);
    }

    CHECK_CUDA(cudaDeviceReset());
    return 0;
}
