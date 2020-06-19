#ifndef VOCAB_H
#define VOCAB_H

#include <cinttypes>
#include <unordered_map>
#include <vector>
#include <string>

class Vocab {
    std::unordered_map<std::string, uint64_t> words;
    public:
        void add(std::string);
        inline size_t size() const {return words.size();}
        void reduce(uint64_t);
        uint64_t build_from_file(std::string);
        uint64_t read_from_file(std::string);
        void save_to_file(std::string);
};

#endif
