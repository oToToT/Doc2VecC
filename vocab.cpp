#include <fstream>
#include <cinttypes>
#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <iostream>
#include "vocab.hpp"
using llu = uint64_t;

extern int debug;

size_t Vocab::size() const noexcept {
    return words.size();
}

void Vocab::add(std::string s) {
    if (words_index.find(s) == words_index.end()) {
        words_index[s] = words.size();
        words_count.push_back(0);
        words.push_back(s);
    }
    words_count[words_index[s]] += 1;
}

void Vocab::conclude() {
    std::vector<std::pair<llu, std::string>> c;
    c.reserve(size());
    for (const auto& it: words_index) {
        c.emplace_back(words_count[it.second], it.first);
    }
    std::sort(
        c.begin(),
        c.end(),
        [] (const std::pair<llu, std::string> a,
                const std::pair<llu, std::string> b) {
            return a.first > b.first;
        }
    );
    std::vector<llu> new_count;
    std::vector<std::string> new_words;
    new_count.reserve(size());
    new_words.reserve(size());
    for (auto& it: words_index) {
        new_count.push_back(words_count[it.second]);
        new_words.push_back(it.first);
        it.second = new_count.size() - 1;
    }
    words_count = new_count;
    words = new_words;
}

void Vocab::reduce(llu reduce_size) {
    if (words_index.size() < reduce_size) return;
    conclude();
    for (size_t i = reduce_size; i < words.size(); ++i) {
        words_index.erase(words[i]);
    }
    words_count.resize(reduce_size);
    words.resize(reduce_size);
}

llu Vocab::build_from_file(std::string filename) {
    std::fstream fs(filename);
    if (fs.fail()) {
        std::cerr << "Error while reading " << filename << std::endl;
        exit(-1);
    }
    llu count = 1;
    add("</s>");
    std::string ln;
    while (std::getline(fs, ln)) {
        std::stringstream ss(ln);
        std::string w;
        while (ss >> w) {
            add(w); count++;
            if (debug > 1 && count % 10000 == 0) {
                std::cout << "Reading " << count / 1000 << "K\r";
            }
        }
        add("</s>"); count++;
    }
    
    return count;
}

llu Vocab::read_from_file(std::string filename) {
    std::fstream fs(filename, std::ios_base::binary | std::ios_base::in);
    if (fs.fail()) {
        std::cerr << "Error while reading " << filename << std::endl;
        exit(-1);
    }
    llu count = 0;
    std::string s; llu c;
    while (fs >> s >> c) {
        words_index[s] = words.size();
        words_count.push_back(c);
        words.push_back(s);
        count += c;
    }
    return count;
}

void Vocab::save_to_file(std::string filename) const {
    std::fstream fs(filename, std::ios_base::binary | std::ios_base::out | std::ios_base::trunc);
    if (fs.fail()) {
        std::cerr << "Error while writing " << filename << std::endl;
        exit(-1);
    }
    for (size_t i = 0; i < words.size(); ++i) {
        fs << words[i] << ' ' << words_count[i] << '\n';
    }
}

std::vector<llu> Vocab::get_count() const noexcept {
    return words_count;
}
