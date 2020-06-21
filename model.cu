#include "model.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cinttypes>
#include <queue>
#include <cuda_runtime.h>
using llu = uint64_t;
using llf = double;

extern int debug;

struct VocabWord {
    size_t *nodes;
    uint8_t *code;
    size_t codelen;
};

void build_binary_tree(const Vocab& vocab, VocabWord *&words) {
    if (debug > 1) {
        std::cout << "Building Huffman Tree." << std::endl;
    }
    cudaMallocManaged(&words, vocab.size() * sizeof(VocabWord));
    auto vocab_cnt = vocab.get_count();
    std::vector<size_t> pa(vocab.size() * 2 - 1);
    std::vector<uint8_t> b_code(vocab.size() * 2 - 1);
    size_t ctr = vocab_cnt.size();
    std::queue<std::pair<llu, size_t>> qu;
    auto top = [&qu, &vocab_cnt]()->std::pair<llu, size_t>{
        if (vocab_cnt.empty()) return qu.front();
        if (qu.empty()) return {vocab_cnt.back(), vocab_cnt.size() - 1};
        if (qu.front().first < vocab_cnt.back())
            return qu.front();
        return {vocab_cnt.back(), vocab_cnt.size() - 1};
    };
    auto pop = [&qu, &vocab_cnt]() {
        if (qu.empty()) vocab_cnt.pop_back();
        else if (vocab_cnt.empty()) qu.pop();
        else if (qu.front().first < vocab_cnt.back())
            qu.pop();
        else vocab_cnt.pop_back();
    };
    while (qu.size() + vocab_cnt.size() > 1) {
        auto m1 = top(); pop();
        auto m2 = top(); pop();
        pa[m1.second] = ctr;
        pa[m2.second] = ctr;
        b_code[m2.second] = 1;
        qu.push({m1.first + m2.first, ctr++});
    }

    for (size_t i = 0; i < vocab.size(); ++i) {
        words[i].codelen = 0;
        for (size_t pt = i; pt + 1 < ctr; pt = pa[pt]) {
            words[i].codelen += 1;
        }
        cudaMallocManaged(&words[i].code, words[i].codelen * sizeof(uint8_t));
        cudaMallocManaged(&words[i].nodes, (words[i].codelen + 1) * sizeof(size_t));
        words[i].nodes[0] = ctr - 1 - vocab.size();
        for (size_t pt = i, t = 0; pt + 1 < ctr; pt = pa[pt], t += 1) {
            words[i].code[words[i].codelen - t - 1] = b_code[pt];
            words[i].nodes[words[i].codelen - t] = pt - vocab.size();
        }
    }
}

// could be parallelized
void file_to_docs(const Vocab& vocab, std::string train_file, size_t *&docs) {
    if (debug > 1) {
        std::cout << "Converting train file to indices." << std::endl;
    }
    std::vector<size_t> tmp;
    std::fstream fs(train_file);
    std::string ln;
    const size_t eol = vocab.get_id("</s>");
    while (std::getline(fs, ln)) {
        std::stringstream ss(ln);
        std::string w;
        while (ss >> w) {
            if (vocab.contain(w)) {
                tmp.push_back(vocab.get_id(w));
            }
        }
        tmp.push_back(eol);
    }
    if (debug > 1) {
        std::cout << "Total Size: " << tmp.size() * sizeof(size_t) / 1024 << " KiB" << std::endl;
    }
    cudaMallocManaged(&docs, tmp.size() * sizeof(size_t));
    memcpy(docs, tmp.data(), tmp.size() * sizeof(size_t));
}


const size_t UNIGRAM_SIZE = 100'000'000;
const llf UNIGRAM_POWER = 0.75;

void init_unigram_table(const Vocab& vocab, size_t *tbl) {
    auto vocab_cnt = vocab.get_count();
    llf words_tot = 0;
    for (size_t i = 0; i < vocab.size(); ++i)
        words_tot += pow(vocab_cnt[i], UNIGRAM_POWER);

    size_t pos = 0;
    llf cur_tot = pow(vocab_cnt[pos], UNIGRAM_POWER) / words_tot;
    for (size_t i = 0; i < UNIGRAM_SIZE; ++i) {
        tbl[i] = pos;
        if (pos + 1 < vocab.size() and i > cur_tot * UNIGRAM_SIZE) {
            pos += 1;
            cur_tot += pow(vocab_cnt[pos], UNIGRAM_POWER) / words_tot;
        }
    }
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

void train_model(const Vocab& vocab, const ModelConfig& conf) {
    VocabWord *words;
    build_binary_tree(vocab, words);

    size_t *docs;
    file_to_docs(vocab, conf.train_file, docs);

    size_t *unigram;
    if (conf.negative_sample > 0) {
        cudaMallocManaged(&unigram, UNIGRAM_SIZE * sizeof(size_t));
        init_unigram_table(vocab, unigram);
    }

    llf *syn0, *syn1, *syn1neg;
    init_net(syn0, syn1, syn1neg, conf, vocab.size());


}
