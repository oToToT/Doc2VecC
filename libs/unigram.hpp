#ifndef UNIGRAM_H
#define UNIGRAM_H

#include "../components/vocab.hpp"

#define UNIGRAM_SIZE 100000000
#define UNIGRAM_POWER 0.75

void init_unigram_table(const Vocab &, size_t **);

#endif
