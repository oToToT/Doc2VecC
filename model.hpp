#ifndef MODEL_H
#define MODEL_H

#include "vocab.hpp"
#include "huffman.hpp"
#include "model_config.hpp"

void train_model(const Vocab&, const ModelConfig&, const VocabWord *, double *&);
void predict_model(double *, double, size_t, const Vocab&, const VocabWord *, std::string, std::string);

#endif
