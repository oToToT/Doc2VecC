#include <cuda_runtime.h>

#include <iostream>
#include <queue>
#include <utility>
#include <vector>

#include "huffman.hpp"

using llu = uint64_t;

extern int debug;

void BuildBinaryTree(const Vocab& vocab, VocabWord** words_ptr) {
  VocabWord*& words = *words_ptr;
  if (debug > 1) {
    std::cout << "Building Huffman Tree." << std::endl;
  }
  cudaMallocManaged(&words, vocab.size() * sizeof(VocabWord));
  auto vocab_cnt = vocab.GetCounts();
  std::vector<size_t> pa(vocab.size() * 2 - 1);
  std::vector<uint8_t> b_code(vocab.size() * 2 - 1);
  size_t ctr = vocab_cnt.size();
  std::queue<std::pair<llu, size_t>> qu;
  auto top = [&qu, &vocab_cnt]() -> std::pair<llu, size_t> {
    if (vocab_cnt.empty()) {
      return qu.front();
    }
    if (qu.empty()) {
      return {vocab_cnt.back(), vocab_cnt.size() - 1};
    }
    if (qu.front().first < vocab_cnt.back()) {
      return qu.front();
    }
    return {vocab_cnt.back(), vocab_cnt.size() - 1};
  };
  auto pop = [&qu, &vocab_cnt]() {
    if (qu.empty()) {
      vocab_cnt.pop_back();
    } else if (vocab_cnt.empty()) {
      qu.pop();
    } else if (qu.front().first < vocab_cnt.back()) {
      qu.pop();
    } else {
      vocab_cnt.pop_back();
    }
  };
  while (qu.size() + vocab_cnt.size() > 1) {
    auto m1 = top();
    pop();
    auto m2 = top();
    pop();
    pa[m1.second] = ctr;
    pa[m2.second] = ctr;
    b_code[m2.second] = 1;
    qu.emplace(m1.first + m2.first, ctr++);
  }

  for (size_t i = 0; i < vocab.size(); ++i) {
    words[i].cnt = vocab.GetCount(i);
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
