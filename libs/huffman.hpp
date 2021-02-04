#ifndef HUFFMAN_H
#define HUFFMAN_H

#include <cinttypes>
#include <cstddef>

#include "../components/vocab.hpp"

struct VocabWord {
  size_t *nodes;
  uint8_t *code;
  size_t codelen;
  uint64_t cnt;
};

void build_binary_tree(const Vocab &, VocabWord *&);

#endif
