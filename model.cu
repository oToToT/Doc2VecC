#include "model.hpp"
#include "huffman.hpp"
#include "unigram.hpp"
#include "sigmoid.hpp"
#include <limits>
#include <random>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cinttypes>
#include <queue>
#include <cuda_runtime.h>
#include <curand_kernel.h>
using llu = uint64_t;
using llf = double;

extern int debug;

#define INF 2147483647

// could be parallelized
std::vector<size_t> file_to_docs(const Vocab& vocab, std::string train_file, size_t *&docs) {
    if (debug > 1) {
        std::cout << "Converting train file to indices." << std::endl;
    }
    size_t pv = 0;
    std::vector<size_t> tmp, pivot;
    std::fstream fs(train_file);
    std::string ln;
    while (std::getline(fs, ln)) {
        std::stringstream ss(ln);
        std::string w;
        while (ss >> w) {
            if (vocab.contain(w)) {
                tmp.push_back(vocab.get_id(w));
            }
        }
        pivot.push_back(tmp.size() - pv);
        pv = tmp.size();
    }
    if (debug > 1) {
        std::cout << "Total Size: " << tmp.size() * sizeof(size_t) / 1024 << " KiB" << std::endl;
    }
    cudaMallocManaged(&docs, tmp.size() * sizeof(size_t));
    memcpy(docs, tmp.data(), tmp.size() * sizeof(size_t));
    return pivot;
}

void init_net(llf *&syn0, llf *&syn1, llf *&syn1neg, const ModelConfig& conf, size_t vocab_size) {
    cudaMallocManaged(&syn0, vocab_size * conf.layer_size * sizeof(llf));
    llu rnd = 1;
    for (size_t i = 0; i < vocab_size; ++i) {
        for (size_t j = 0; j < conf.layer_size; ++j) {
            rnd = rnd * 25214903917LLU + 11;
            syn0[i * conf.layer_size + j] = (static_cast<llf>(rnd & 0xffff) / 0xffff - 0.5) / conf.layer_size;
        }
    }

    if (conf.hierarchical_softmax) {
        cudaMallocManaged(&syn1, vocab_size * conf.layer_size * sizeof(llf));
        cudaMemset(syn1, 0, vocab_size * conf.layer_size * sizeof(llf));
    }

    if (conf.negative_sample > 0) {
        cudaMallocManaged(&syn1neg, vocab_size * conf.layer_size * sizeof(llf));
        cudaMemset(syn1neg, 0, vocab_size * conf.layer_size * sizeof(llf));
    }
}

__constant__ llf sigmoid[SIGMOID_TABLE_SIZE + 1];

void init_sigmoid() {
    llf sigTbl[SIGMOID_TABLE_SIZE + 1];
    for (size_t i = 0; i <= SIGMOID_TABLE_SIZE; ++i) {
        sigTbl[i] = exp((static_cast<llf>(i) / SIGMOID_TABLE_SIZE * 2 - 1) * MAX_EXP);
        sigTbl[i] = sigTbl[i] / (sigTbl[i] + 1);
    }
    cudaMemcpyToSymbol(sigmoid, sigTbl, (SIGMOID_TABLE_SIZE + 1) * sizeof(llf));
}

__global__ void sample_doc(const size_t *doc, const size_t n, const llf s, const llf rp_sample, const VocabWord *w, const size_t total_count, const int seed, size_t *sen, size_t *sen_sample) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed, i, 0, &state);
    if (i >= n) return;
    sen[i] = INF;
    llf r = 1;
    if (s > 0) {
        r = (sqrt(w[i].cnt / (s * total_count)) + 1) * (s * total_count) / w[i].cnt;
    }
    if (r >= curand_uniform_double(&state)) {
        sen[i] = doc[i];
    }
    sen_sample[i] = INF;
    if (rp_sample >= curand_uniform_double(&state)) {
        sen_sample[i] = sen[i];
    }
}


__global__ void cbow(const size_t *sen, const size_t n, const size_t w, const int b, const int ws, llf *syn0, llf *neu1) {
    const int c = (int)w + (int)blockIdx.x + b - ws;
    if (c < 0 or c >= n or sen[c] == INF or c == w) return;
    const int i = threadIdx.x;
    atomicAdd(&neu1[i], syn0[sen[c] * blockDim.x + i]);
}

__global__ void doc2vecc(const size_t *sen_sample, const size_t n, const llf w, llf *syn0, llf *neu1) {
    const int c = blockIdx.x;
    if (c >= n or sen_sample[c] == INF) return;
    const int i = threadIdx.x;
    atomicAdd(&neu1[c], w * syn0[i + sen_sample[c] * blockDim.x]);
}

__global__ void negative_sampling() {

}

static inline llu rnd() {
    static llu r = 0;
    r = r * 25214903917LLU + 11LLU;
    return r;
}

void train_model(const Vocab& vocab, const ModelConfig& conf) {
    VocabWord *words;
    build_binary_tree(vocab, words);

    size_t *docs;
    std::vector<size_t> pvt = file_to_docs(vocab, conf.train_file, docs);
    size_t mx_len = *std::max_element(pvt.begin(), pvt.end());

    size_t *unigram;
    if (conf.negative_sample > 0) {
        init_unigram_table(vocab, unigram);
    }

    llf *syn0, *syn1, *syn1neg;
    init_net(syn0, syn1, syn1neg, conf, vocab.size());

    init_sigmoid();

    size_t *sen, *sen_sample;
    cudaMalloc(&sen, mx_len * sizeof(size_t));
    cudaMalloc(&sen_sample, mx_len * sizeof(size_t));
    llf *neu1, *neu1e;
    cudaMalloc(&neu1, conf.layer_size * sizeof(llf));
    cudaMalloc(&neu1e, conf.layer_size * sizeof(llf));
    
    for (llu it = 0; it < conf.iterations; ++it) {
        if (debug > 0) {
            std::cout << "training epoch " << it << std::endl;
        }
        size_t *doc = docs;
        for (auto pt: pvt) {
            sample_doc<<<256, (pt + 255) / 256>>>(doc, pt, conf.sample_rate, conf.rp_sample, words, vocab.get_total_count(), rnd(), sen, sen_sample);
            doc2vecc<<<conf.layer_size, pt>>>(sen_sample, pt, 1. / conf.rp_sample / pt, syn0, neu1);
            for (size_t i = 0; i < pt; ++i) {
                cudaMemset(&neu1, 0, conf.layer_size * sizeof(llf));
                if (conf.cbow) {
                    int b = rnd() % conf.window_size;
                    // maybe conf.window_size * 1
                    int blocks = conf.window_size * 2 - 2 * b + 1;
                    cbow<<<conf.layer_size, blocks>>>(doc, pt, i, b, conf.window_size, syn0, neu1);
                    int cw = std::max((int)i + conf.window_size - b + 1, (int)pt) - std::max((int)i - conf.window_size + b, 0);
                    if (conf.hierarchical_softmax) {

                    } else {
                    }
                } else {
                    if (conf.hierarchical_softmax) {
                    } else {
                    }
                }
            }
            doc += pt;
        }
    }
}
