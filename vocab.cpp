#include <fstream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <string>
#include <utility>
#include <sstream>
#include <iostream>
#include "vocab.hpp"
using llu = uint64_t;

extern int debug;

VocabWord::VocabWord(): count(0) {}

void Vocab::add(std::string s) {
    words[s].count += 1;
}
void Vocab::reduce(llu reduce_size) {
    if (words.size() < reduce_size) return;
    std::vector<std::pair<llu, std::string>> c;
    for (const auto& it: words) {
        c.emplace_back(it.second.count, it.first);
    }
    std::sort(c.begin(), c.end(),
            [](const std::pair<llu, std::string> a,
                const std::pair<llu, std::string> b) {
            return a.first > b.first;
            });
    for (llu i = reduce_size; i < c.size(); ++i) {
        words.erase(c[i].second);
    }
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
        words[s].count = c;
        count += c;
    }
    return count;
}
void Vocab::save_to_file(std::string filename) {
    std::fstream fs(filename, std::ios_base::binary | std::ios_base::out | std::ios_base::trunc);
    if (fs.fail()) {
        std::cerr << "Error while writing " << filename << std::endl;
        exit(-1);
    }
    for (const auto& it: words) {
        fs << it.first << " " << it.second.count << '\n';
    }
}

