#include "model.hpp"
#include "huffman.hpp"
#include "unigram.hpp"
#include "sigmoid.hpp"
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cinttypes>
#include <queue>
#include <cuda_runtime.h>
using llu = uint64_t;
using llf = double;

extern int debug;

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

__global__ void cbow_hs() {
}

__global__ void cbow_ns() {

}

__global__ void skip_hs() {
}

__global__ void skip_ns() {
}

void train_model(const Vocab& vocab, const ModelConfig& conf) {
    VocabWord *words;
    build_binary_tree(vocab, words);

    size_t *docs;
    std::vector<size_t> pvt = file_to_docs(vocab, conf.train_file, docs);
    std::cout << *std::max_element(pvt.begin(), pvt.end()) << std::endl;

    size_t *unigram;
    if (conf.negative_sample > 0) {
        init_unigram_table(vocab, unigram);
    }

    llf *syn0, *syn1, *syn1neg;
    init_net(syn0, syn1, syn1neg, conf, vocab.size());

    init_sigmoid();
    
    for (llu it = 0; it < conf.iterations; ++it) {
        size_t *doc = docs;
        for (auto pt: pvt) {
            if (conf.cbow) {
                if (conf.hierarchical_softmax) {

                } else {
                }
            } else {
                if (conf.hierarchical_softmax) {
                } else {
                }
            }
            doc += pt;
        }
    }
}
