#ifndef BANDWIDTH_H
#define BANDWIDTH_H

#include <cstddef>
#include <vector>

struct BandwidthResult {
    double sizeMB;         // Array size in MB
    double h2d_pageable;   // GB/s
    double d2h_pageable;   // GB/s
    double h2d_pinned;     // GB/s
    double d2h_pinned;     // GB/s
};

/**
 * Run bandwidth tests for a list of sizes.
 *
 * @param sizes      Array sizes in bytes.
 * @param num_iters  Number of repetitions per measurement.
 * @param results    Output vector of BandwidthResult, one per size.
 */
void run_bandwidth_tests(const std::vector<size_t>& sizes,
                         int num_iters,
                         std::vector<BandwidthResult>& results);

#endif // BANDWIDTH_H
