#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#include <cinttypes>
#include <cstddef>

struct ModelConfig {
    double sample_rate, alpha, rp_sample;
    size_t layer_size;
    uint64_t iterations;
    int window_size, negative_sample, min_count;
    bool hierarchical_softmax, cbow, binary;
};

#endif
