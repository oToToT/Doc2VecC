#ifndef MODEL_H
#define MODEL_H

#include "components/model_config.hpp"
#include "components/vocab.hpp"
#include "libs/huffman.hpp"

void TrainModel(const Vocab &, const ModelConfig &, const VocabWord *,
                 double *&);
void PredictModel(double *, double, size_t, const Vocab &, const VocabWord *,
                   std::string, std::string);

#endif
