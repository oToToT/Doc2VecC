#ifndef VOCAB_H
#define VOCAB_H

#include <cinttypes>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class Vocab {
  std::vector<std::string> words_;
  std::vector<uint64_t> words_count_;
  std::unordered_map<std::string, size_t> words_index_;
  uint64_t vocab_count_;

 public:
  void add(const std::string&);
  bool contain(const std::string&) const;
  std::string get_word(size_t) const;
  size_t get_id(const std::string&) const;
  size_t size() const noexcept;
  void conclude();
  void reduce(uint64_t);
  uint64_t build_from_file(const std::string&);
  uint64_t read_from_file(const std::string&);
  void save_to_file(const std::string&) const;
  std::vector<uint64_t> get_count() const noexcept;
  uint64_t get_count(size_t) const;
  uint64_t get_total_count() const noexcept;
};

#endif
