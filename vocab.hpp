#include <cinttypes>
#include <unordered_map>
#include <vector>
#include <string>

struct VocabWord {
    VocabWord();
    uint64_t count;
};
class Vocab {
    std::unordered_map<std::string, VocabWord> words;
    public:
        void add(std::string);
        inline size_t size() const {return words.size();}
        void reduce(uint64_t);
        uint64_t build_from_file(std::string);
        uint64_t read_from_file(std::string);
        void save_to_file(std::string);
};
