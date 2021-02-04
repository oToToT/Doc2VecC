#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_functions.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <sstream>

#include "libs/sigmoid.hpp"
#include "libs/unigram.hpp"
#include "model.hpp"
using llu = uint64_t;
using llf = double;

extern int debug;

#define INF 2147483647
#define SAMPLE_DOC_THREAD 1024

static inline llu rnd() {
  static llu r = 0;
  r = r * 25214903917LLU + 11LLU;
  return r;
}

// could be parallelized?
static std::vector<size_t> FileToDocs(const Vocab &vocab, std::string f,
                                      size_t *&docs) {
  if (debug > 1) {
    std::cout << "Converting train file to indices." << std::endl;
  }
  size_t pv = 0;
  std::vector<size_t> tmp;
  std::vector<size_t> pivot;
  std::fstream fs(f);
  std::string ln;
  while (std::getline(fs, ln)) {
    std::stringstream ss(ln);
    std::string w;
    while (ss >> w) {
      if (vocab.contain(w)) {
        tmp.push_back(vocab.get_id(w));
      }
    }
    pivot.push_back(tmp.size() - pv);
    pv = tmp.size();
  }
  if (debug > 1) {
    std::cout << "Total Size: " << tmp.size() * sizeof(size_t) / 1024 << " KiB"
              << std::endl;
  }
  cudaMallocManaged(&docs, tmp.size() * sizeof(size_t));
  memcpy(docs, tmp.data(), tmp.size() * sizeof(size_t));
  return pivot;
}

static void InitNet(llf *&syn0, llf *&syn1, const ModelConfig &conf,
                    size_t vocab_size) {
  cudaMallocManaged(&syn0, vocab_size * conf.layer_size * sizeof(llf));
  llu rnd = 1;
  for (size_t i = 0; i < vocab_size; ++i) {
    for (size_t j = 0; j < conf.layer_size; ++j) {
      rnd = rnd * 25214903917LLU + 11;
      syn0[i * conf.layer_size + j] =
          (static_cast<llf>(rnd & 0xffff) / 0xffff - 0.5) / conf.layer_size;
    }
  }

  cudaMalloc(&syn1, vocab_size * conf.layer_size * sizeof(llf));
  cudaMemset(syn1, 0, vocab_size * conf.layer_size * sizeof(llf));
}

__constant__ llf sigmoid[SIGMOID_TABLE_SIZE + 1];

static void InitSigmoid() {
  std::array<llf, SIGMOID_TABLE_SIZE + 1> sigmoid_table;
  for (size_t i = 0; i <= SIGMOID_TABLE_SIZE; ++i) {
    sigmoid_table[i] =
        std::exp((static_cast<llf>(i) / SIGMOID_TABLE_SIZE * 2 - 1) * MAX_EXP);
    sigmoid_table[i] = sigmoid_table[i] / (sigmoid_table[i] + 1);
  }
  cudaMemcpyToSymbol(sigmoid, sigmoid_table.data(),
                     (SIGMOID_TABLE_SIZE + 1) * sizeof(llf));
}

static __global__ void SampleDoc(const size_t *doc, const size_t n, const llf s,
                                 const llf rp_sample, const VocabWord *w,
                                 const size_t total_count, const llu seed,
                                 size_t *sen, size_t *sen_sample,
                                 int *sen_len) {
  extern __shared__ int shm[];
  shm[threadIdx.x] = 0;

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    curandState state;
    curand_init(seed, i, 0, &state);
    sen[i] = INF;
    llf r = 1;
    if (s > 0) {
      r = (sqrt(w[i].cnt / (s * total_count)) + 1) * (s * total_count) /
          w[i].cnt;
      shm[threadIdx.x] += 1;
    }
    if (r >= curand_uniform_double(&state)) {
      sen[i] = doc[i];
    }
    sen_sample[i] = INF;
    if (rp_sample >= curand_uniform_double(&state)) {
      sen_sample[i] = sen[i];
    }
  }
  for (int step = blockDim.x >> 1; step; step >>= 1) {
    if (threadIdx.x < step) {
      shm[threadIdx.x] += shm[threadIdx.x + step];
    }
  }
  if (threadIdx.x == 0) {
    sen_len[blockIdx.x] = shm[0];
  }
}

static __global__ void CBOW(const size_t *sen, const size_t n, const size_t w,
                            const int b, const int ws, llf *syn0, llf *neu1) {
  const int c = (int)w + (int)blockIdx.x + b - ws;
  if (c < 0 or c >= n or sen[c] == INF or c == w) return;
  const int i = threadIdx.x;
  neu1[i] += syn0[sen[c] * ws + i];
  // atomicAdd(&neu1[i], syn0[sen[c] * ws + i]);
}

static __global__ void CBOWBack(const size_t *sen, const size_t n,
                                const size_t w, const int b, const int ws,
                                const int cw, llf *syn0, llf *neu1e) {
  const int c = (int)w + (int)blockIdx.x + b - ws;
  if (c < 0 or c >= n or sen[c] == INF or c == w) return;
  const int i = threadIdx.x;
  syn0[i + sen[c] * ws] += neu1e[i] / cw;
  // atomicAdd(&syn0[i + sen[c] * ws], neu1e[i] / cw);
}

static __global__ void Doc2vecc(const size_t *sen_sample, const size_t n,
                                const llf w, llf *syn0, llf *neu1) {
  const int c = blockIdx.x;
  if (sen_sample[c] == INF) return;
  const int i = threadIdx.x;
  neu1[i] += w * syn0[i + sen_sample[c] * blockDim.x];
  // atomicAdd(&neu1[i], w * syn0[i + sen_sample[c] * blockDim.x]);
}

__global__ void Doc2veccBack(const size_t *sen_sample, const size_t n,
                             const llf w, const int cw, llf *syn0, llf *neu1e) {
  const int c = blockIdx.x;
  if (sen_sample[c] == INF) return;
  const int i = threadIdx.x;
  syn0[i + sen_sample[c] * blockDim.x] += neu1e[i] * w / cw;
  // atomicAdd(&syn0[i + sen_sample[c] * blockDim.x], neu1e[i] * w / cw);
}

static __global__ void NegativeSampling(const size_t w, const llf alpha,
                                        const size_t lsize, const llu seed,
                                        size_t *uni, llf *syn1, llf *neu1,
                                        llf *neu1e) {
  const int c = threadIdx.x;

  extern __shared__ llf sm[];
  sm[c] = 0;

  size_t ta;
  int label;
  if (blockIdx.x == 0) {
    ta = w;
    label = 1;
  } else {
    curandState state;
    curand_init(seed, blockIdx.x, 0, &state);
    ta = uni[(int)(curand_uniform(&state) * UNIGRAM_SIZE)];
    label = 0;
  }
  const size_t tpos = ta * lsize;
  if (c < lsize) {
    sm[c] = neu1[c] * syn1[c + tpos];
  }
  for (int step = blockDim.x >> 1; step; step >>= 1) {
    __syncthreads();
    if (c < step) {
      sm[c] += sm[c + step];
    }
  }
  if (c == 0) {
    sm[lsize] = sm[0];
  }
  __syncthreads();
  llf f = sm[lsize], g;
  if (f > MAX_EXP)
    g = (label - 1) * alpha;
  else if (f < -MAX_EXP)
    g = (label - 0) * alpha;
  else
    g = (label -
         sigmoid[(int)((f + MAX_EXP) * (SIGMOID_TABLE_SIZE / MAX_EXP / 2))]) *
        alpha;
  if (c < lsize) {
    neu1e[c] += g * syn1[c + tpos];
    syn1[c + tpos] += g * neu1[c];
    // atomicAdd(&neu1e[c], g * syn1[c + tpos]);
    // atomicAdd(&syn1[c + tpos], g * neu1[c]);
  }
}

static __global__ void averaging(const int cw, llf *neu1) {
  neu1[threadIdx.x] /= cw;
}

static void SaveEmbedding(std::string f, llf *syn0, const Vocab &vocab,
                          size_t layer_size) {
  std::fstream fs(f);
  fs << vocab.size() << layer_size << '\n';
  for (size_t i = 0; i < vocab.size(); ++i) {
    fs << vocab.get_word(i);
    for (size_t j = 0; j < layer_size; ++j)
      fs << ' ' << syn0[i * layer_size + j];
    fs << '\n';
  }
}

void TrainModel(const Vocab &vocab, const ModelConfig &conf,
                const VocabWord *words, llf *&syn0) {
  size_t layer_size_2 = 1;
  while (layer_size_2 < conf.layer_size) layer_size_2 <<= 1;

  size_t *docs;
  std::vector<size_t> pvt = FileToDocs(vocab, conf.train_file, docs);
  size_t mx_len = *std::max_element(pvt.begin(), pvt.end());

  size_t *unigram;
  if (conf.negative_sample > 0) {
    init_unigram_table(vocab, &unigram);
  }

  llf *syn1;
  InitNet(syn0, syn1, conf, vocab.size());

  InitSigmoid();

  size_t *sen, *sen_sample;
  cudaMalloc(&sen, mx_len * sizeof(size_t));
  cudaMalloc(&sen_sample, mx_len * sizeof(size_t));
  llf *neu1, *neu1e;
  cudaMalloc(&neu1, conf.layer_size * sizeof(llf));
  cudaMalloc(&neu1e, conf.layer_size * sizeof(llf));

  int *sen_lens;
  cudaMallocManaged(&sen_lens, (mx_len + SAMPLE_DOC_THREAD - 1) /
                                   SAMPLE_DOC_THREAD * sizeof(int));

  const llu total_train_word = conf.iterations * vocab.get_total_count();
  llu cur_train_word = 0;
  for (llu it = 0; it < conf.iterations; ++it) {
    if (debug > 0) {
      std::cout << "training epoch " << it << "                 " << std::endl;
    }
    size_t *doc = docs;
    for (auto pt : pvt) {
      cudaMemset(
          sen_lens, 0,
          (pt + SAMPLE_DOC_THREAD - 1) / SAMPLE_DOC_THREAD * sizeof(int));
      SampleDoc<<<(pt + SAMPLE_DOC_THREAD - 1) / SAMPLE_DOC_THREAD,
                  SAMPLE_DOC_THREAD, SAMPLE_DOC_THREAD * sizeof(int)>>>(
          doc, pt, conf.sample_rate, conf.rp_sample, words,
          vocab.get_total_count(), rnd(), sen, sen_sample, sen_lens);
      int sen_len = 0;
      for (int i = 0; i < (pt + SAMPLE_DOC_THREAD - 1) / SAMPLE_DOC_THREAD; ++i)
        sen_len += sen_lens[i];
      for (size_t i = 0; i < pt; ++i) {
        llf alpha = conf.alpha *
                    (1 - static_cast<llf>(cur_train_word) / total_train_word);
        alpha = std::max(alpha, conf.alpha * 0.0001);
        if (debug > 0 and cur_train_word % 100 == 0) {
          std::cout << "Alpha: " << alpha << ", Progress: "
                    << static_cast<llf>(cur_train_word) / total_train_word * 100
                    << "%                             \r";
        }
        cudaMemset(neu1, 0, conf.layer_size * sizeof(llf));
        cudaMemset(neu1e, 0, conf.layer_size * sizeof(llf));
        Doc2vecc<<<pt, conf.layer_size>>>(
            sen_sample, pt, 1. / conf.rp_sample / sen_len, syn0, neu1);
        int cw = 2;
        if (conf.cbow) {
          int b = rnd() % conf.window_size;
          // this part is quite wrong...
          int blocks = conf.window_size * 2 - 2 * b + 1;
          CBOW<<<blocks, conf.layer_size>>>(doc, pt, i, b, conf.window_size,
                                            syn0, neu1);
          cw = std::max((int)i + conf.window_size - b + 1, (int)pt) -
               std::max((int)i - conf.window_size + b, 0);
          averaging<<<1, conf.layer_size>>>(cw, neu1);
          if (conf.hierarchical_softmax) {
          } else {
            NegativeSampling<<<conf.negative_sample, layer_size_2,
                               (layer_size_2 + 1) * sizeof(llf)>>>(
                i, alpha, conf.layer_size, rnd(), unigram, syn1, neu1, neu1e);
          }
          CBOWBack<<<blocks, conf.layer_size>>>(doc, pt, i, b, conf.window_size,
                                                cw, syn0, neu1e);
        } else {
          if (conf.hierarchical_softmax) {
          } else {
          }
        }
        Doc2veccBack<<<pt, conf.layer_size>>>(
            sen_sample, pt, 1. / conf.rp_sample / sen_len, cw, syn0, neu1e);
        cur_train_word++;
      }
      doc += pt;
    }
  }
  SaveEmbedding(conf.wordembedding_file, syn0, vocab, conf.layer_size);
}

void PredictModel(llf *syn0, llf sr, const size_t layer_size,
                  const Vocab &vocab, const VocabWord *words, std::string inp,
                  std::string oup) {
  std::fstream fs(oup);
  size_t *docs, *sen_sample, *sen;
  std::vector<size_t> pvt = FileToDocs(vocab, inp, docs);
  size_t mx_len = *std::max_element(pvt.begin(), pvt.end());
  cudaMallocManaged(&sen, mx_len * sizeof(size_t));
  cudaMalloc(&sen_sample, mx_len * sizeof(size_t));

  llf *neu1;
  cudaMallocManaged(&neu1, layer_size * sizeof(llf));

  int *sen_lens;
  cudaMallocManaged(&sen_lens, (mx_len + SAMPLE_DOC_THREAD - 1) /
                                   SAMPLE_DOC_THREAD * sizeof(int));
  for (auto pt : pvt) {
    SampleDoc<<<(pt + SAMPLE_DOC_THREAD - 1) / SAMPLE_DOC_THREAD,
                SAMPLE_DOC_THREAD, SAMPLE_DOC_THREAD * sizeof(int)>>>(
        docs, pt, sr, 0, words, vocab.get_total_count(), rnd(), sen, sen_sample,
        sen_lens);
    int sen_len = 0;
    for (int i = 0; i < (pt + SAMPLE_DOC_THREAD - 1) / SAMPLE_DOC_THREAD; ++i)
      sen_len += sen_lens[i];
    Doc2Vecc<<<pt, layer_size>>>(sen_sample, pt, 1. / sen_len, syn0, neu1);
    for (size_t i = 0; i < layer_size; ++i) {
      fs << neu1[i] << "　\n"[i + 1 == layer_size];
    }
    docs += pt;
  }
}
