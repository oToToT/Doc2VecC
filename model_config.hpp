#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#include <cinttypes>

struct ModelConfig {
    double sample_rate, alpha, rp_sample;
    bool hierarchical_softmax, cbow, binary;
    uint64_t layer_size, iterations;
    int window_size, negative_sample, min_count;
};

#endif
