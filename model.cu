#include "model.hpp"
#include <cinttypes>
#include <queue>
#include <cuda_runtime.h>
using llu = uint64_t;

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


void init_unigram_table() {

}

void init_net() {
}

void train_model(const Vocab& vocab, const ModelConfig& conf) {
    VocabWord *words;
    cudaMallocManaged(&words, vocab.size() * sizeof(VocabWord));
    build_binary_tree(vocab, words);

    if (conf.negative_sample > 0) init_unigram_table();

    init_net();
}
