#include "vocab.hpp"

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
using llu = uint64_t;

extern int debug;

size_t Vocab::size() const noexcept { return words_.size(); }

bool Vocab::contain(const std::string& s) const {
  return words_index_.find(s) != words_index_.end();
}

size_t Vocab::get_id(const std::string& s) const {
  return words_index_.find(s)->second;
}

std::string Vocab::get_word(size_t i) const { return words_[i]; }

void Vocab::add(const std::string& s) {
  if (words_index_.find(s) == words_index_.end()) {
    words_index_[s] = words_.size();
    words_count_.push_back(0);
    words_.push_back(s);
  }
  words_count_[words_index_[s]] += 1;
}

void Vocab::conclude() {
  std::vector<std::pair<llu, std::string>> c;
  c.reserve(size());
  for (const auto& it : words_index_) {
    c.emplace_back(words_count_[it.second], it.first);
  }
  std::sort(
      c.begin(), c.end(),
      [](const std::pair<llu, std::string>& a,
         const std::pair<llu, std::string>& b) { return a.first > b.first; });
  std::vector<llu> new_count;
  std::vector<std::string> new_words;
  new_count.reserve(size());
  new_words.reserve(size());
  for (const auto& it : c) {
    new_count.push_back(words_count_[words_index_[it.second]]);
    new_words.push_back(it.second);
    words_index_[it.second] = new_count.size() - 1;
  }
  words_count_ = new_count;
  words_ = new_words;
}

void Vocab::reduce(llu reduce_cnt) {
  conclude();
  size_t reduce_size;
  for (reduce_size = 0; reduce_size < words_.size(); ++reduce_size) {
    if (words_count_[reduce_size] < reduce_cnt) break;
  }
  for (size_t i = reduce_size; i < words_.size(); ++i) {
    words_index_.erase(words_[i]);
  }
  words_count_.resize(reduce_size);
  words_.resize(reduce_size);
  vocab_count_ = std::accumulate(words_count_.begin(), words_count_.end(), 0);
}

llu Vocab::build_from_file(const std::string& filename) {
  std::fstream fs(filename);
  if (fs.fail()) {
    std::cerr << "Error while reading " << filename << std::endl;
    exit(-1);
  }
  vocab_count_ = 0;
  std::string w;
  while (fs >> w) {
    add(w);
    vocab_count_++;
    if (debug > 1 && vocab_count_ % 10000 == 0) {
      std::cout << "Reading " << vocab_count_ / 1000 << "K\r";
    }
  }

  return vocab_count_;
}

llu Vocab::read_from_file(const std::string& filename) {
  std::fstream fs(filename, std::ios_base::binary | std::ios_base::in);
  if (fs.fail()) {
    std::cerr << "Error while reading " << filename << std::endl;
    exit(-1);
  }
  vocab_count_ = 0;
  std::string s;
  llu c;
  while (fs >> s >> c) {
    words_index_[s] = words_.size();
    words_count_.push_back(c);
    words_.push_back(s);
    vocab_count_ += c;
  }
  return vocab_count_;
}

void Vocab::save_to_file(const std::string& filename) const {
  std::fstream fs(filename, std::ios_base::binary | std::ios_base::out |
                                std::ios_base::trunc);
  if (fs.fail()) {
    std::cerr << "Error while writing " << filename << std::endl;
    exit(-1);
  }
  for (size_t i = 0; i < words_.size(); ++i) {
    fs << words_[i] << ' ' << words_count_[i] << '\n';
  }
}

std::vector<llu> Vocab::get_count() const noexcept { return words_count_; }

llu Vocab::get_count(size_t i) const { return words_count_[i]; }

llu Vocab::get_total_count() const noexcept { return vocab_count_; }
