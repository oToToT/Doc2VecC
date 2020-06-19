#include "model.hpp"
#include <cinttypes>
#include <queue>
#include <cuda_runtime.h>
using llu = uint64_t;
using llf = double;

struct VocabWord {
    size_t *nodes;
    uint8_t *code;
    size_t codelen;
};

void build_binary_tree(const Vocab& vocab, VocabWord *words) {
    // need optimized to O(N) time
    auto vocab_cnt = vocab.get_count();
    std::vector<size_t> pa(vocab.size() * 2 - 1);
    std::vector<uint8_t> b_code(vocab.size() * 2 - 1);
    size_t ctr = vocab_cnt.size();
    std::priority_queue<
        std::pair<llu, size_t>,
        std::vector<std::pair<llu, size_t>>,
        std::greater<std::pair<llu, size_t>>
    > pq;
    for (size_t i = 0; i < vocab.size(); ++i) {
        pq.push({vocab_cnt[i], i});
    }
    while (pq.size() > 1) {
        auto m1 = pq.top(); pq.pop();
        auto m2 = pq.top(); pq.pop();
        pa[m1.second] = ctr;
        pa[m2.second] = ctr;
        b_code[m2.second] = 1;
        pq.push({m1.first + m2.first, ctr++});
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


void init_net(llf *syn0, llf *syn1, llf *syn1neg, const ModelConfig& conf, size_t vocab_size) {
    llu rnd = 1;
    for (size_t i = 0; i < vocab_size; ++i) {
        for (size_t j = 0; j < conf.layer_size; ++j) {
            rnd = rnd * 25214903917LLU + 11;
            syn0[i * conf.layer_size + j] = (static_cast<llf>(rnd & 0xffff) / 0xffff - 0.5) / conf.layer_size;
        }
    }

    if (conf.hierarchical_softmax) {
        cudaMemset(syn1, 0, vocab_size * conf.layer_size * sizeof(llf));
    }

    if (conf.negative_sample > 0) {
        cudaMemset(syn1neg, 0, vocab_size * conf.layer_size * sizeof(llf));
    }
}

void train_model(const Vocab& vocab, const ModelConfig& conf) {
    VocabWord *words;
    cudaMallocManaged(&words, vocab.size() * sizeof(VocabWord));
    build_binary_tree(vocab, words);

    llf *syn0, *syn1, *syn1neg;
    cudaMallocManaged(&syn0, vocab.size() * conf.layer_size * sizeof(llf));
    if (conf.hierarchical_softmax) {
        cudaMallocManaged(&syn1, vocab.size() * conf.layer_size * sizeof(llf));
    }
    size_t *unigram;
    if (conf.negative_sample > 0) {
        cudaMallocManaged(&syn1neg, vocab.size() * conf.layer_size * sizeof(llf));
        cudaMallocManaged(&unigram, UNIGRAM_SIZE * sizeof(size_t));
        init_unigram_table(vocab, unigram);
    }
    
    init_net(syn0, syn1, syn1neg, conf, vocab.size());
}
