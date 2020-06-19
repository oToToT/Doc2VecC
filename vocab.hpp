#ifndef VOCAB_H
#define VOCAB_H

#include <cinttypes>
#include <unordered_map>
#include <vector>
#include <string>
#include <utility>

class Vocab {
    std::vector<std::string> words;
    std::vector<uint64_t> words_count;
    std::unordered_map<std::string, size_t> words_index;
    public:
        void add(std::string);
        size_t size() const noexcept;
        void reduce(uint64_t);
        uint64_t build_from_file(std::string);
        uint64_t read_from_file(std::string);
        void save_to_file(std::string) const;
        std::vector<uint64_t> get_count() const noexcept;
};

#endif
