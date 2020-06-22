#include "unigram.hpp"
#include <cuda_runtime.h>
#include <cinttypes>
#include <cstddef>

using llf = double;

void init_unigram_table(const Vocab& vocab, size_t *&tbl) {
    cudaMallocManaged(&tbl, UNIGRAM_SIZE * sizeof(size_t));
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