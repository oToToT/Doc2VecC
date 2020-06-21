#ifndef HUFFMAN_H
#define HUFFMAN_H

#include "vocab.hpp"
#include <cinttypes>
#include <cstddef>

struct VocabWord {
    size_t *nodes;
    uint8_t *code;
    size_t codelen;
};

void build_binary_tree(const Vocab &, VocabWord *&);

#endif
