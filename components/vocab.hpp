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
  size_t size() const noexcept;
  void Add(const std::string&);
  bool Contain(const std::string&) const;
  std::string GetWord(size_t) const;
  size_t GetId(const std::string&) const;
  void Sort();
  void Reduce(uint64_t);
  uint64_t BuildFromFile(const std::string&);
  uint64_t RestoreFromSavedFile(const std::string&);
  void SaveToFile(const std::string&) const;
  std::vector<uint64_t> GetCounts() const noexcept;
  uint64_t GetCount(size_t) const;
  uint64_t GetTotalCount() const noexcept;
};

#endif
