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
        void add(const std::string&);
        bool contain(const std::string&) const;
        size_t get_id(const std::string&) const;
        size_t size() const noexcept;
        void conclude();
        void reduce(uint64_t);
        uint64_t build_from_file(const std::string&);
        uint64_t read_from_file(const std::string&);
        void save_to_file(const std::string&) const;
        std::vector<uint64_t> get_count() const noexcept;
};

#endif
