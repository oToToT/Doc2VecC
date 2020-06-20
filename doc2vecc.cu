#include "arg_parser.hpp"
#include "vocab.hpp"
#include "model.hpp"
#include "model_config.hpp"
#include <iostream>
#include <cstdlib>
#include <cinttypes>
#include <cuda_runtime.h>

using llf = double;
using llu = uint64_t;

int gpu_id, debug;
llu max_vocab_size;
std::string vocab_output, vocab_source;

void print_usage() {
    std::cout << R"(GPU accelerated Doc2VecC implementation

Options:
Parameters for training:
	-train <file>
		Use text data from <file> to train the model; default is data.txt
	-word <file>
		Use <file> to save the resulting word vectors; default is wordvec.txt
	-output <file>
		Use <file> to save the resulting document vectors; default is docvec.txt
	-size <int>
		Set size of word vectors; default is 100
	-window <int>
		Set max skip length between words; default is 5
	-sample <float>
		Set threshold for occurrence of words. Those that appear with higher frequency in the training data
		will be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)
	-hs <int>
		Use Hierarchical Softmax; default is 0 (not used)
	-negative <int>
		Number of negative examples; default is 5, common values are 3 - 10 (0 = not used)
	-gpu <int>
		Use <int>-th gpu (default 0)
	-iter <int>
		Run more training iterations (default 10)
	-min-count <int>
		This will discard words that appear less than <int> times; default is 5
	-alpha <float>
		Set the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW
	-debug <int>
		Set the debug mode (default = 2 = more info during training)
	-binary <int>
		Save the resulting vectors in binary moded; default is 0 (off)
	-save-vocab <file>
		The vocabulary will be saved to <file>
	-read-vocab <file>
		The vocabulary will be read from <file>, not constructed from the training data
	-cbow <int>
		Use the continuous bag of words model; default is 1 (use 0 for skip-gram model)
	-sentence-sample <float>
		The rate to sample words out of a document for documenet representation; default is 0.1
    -vocab-limit <int>
        The size limit of vocabulary dictionary; default is 21000000

Examples:
./doc2vecc -train data.txt -output docvec.txt -word wordvec.txt -sentence-sample 0.1 -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3)" << std::endl;
}
ArgParser get_parser() {
    ArgParser parser;
    parser.add_argument("-train", "data.txt");
    parser.add_argument("-word", "wordvec.txt");
    parser.add_argument("-output", "docvec.txt");
    parser.add_argument("-size", "100");
    parser.add_argument("-window", "5");
    parser.add_argument("-sample", "1e-3");
    parser.add_argument("-hs", "0");
    parser.add_argument("-negative", "5");
    parser.add_argument("-gpu", "0");
    parser.add_argument("-iter", "10");
    parser.add_argument("-min-count", "5");
    parser.add_argument("-alpha");
    parser.add_argument("-debug", "2");
    parser.add_argument("-binary", "0");
    parser.add_argument("-save-vocab");
    parser.add_argument("-read-vocab");
    parser.add_argument("-cbow", "1");
    parser.add_argument("-sentence-sample", "0.1");
    parser.add_argument("-vocab-limit", "21000000");
    return parser;
}

ModelConfig parse_args(ArgParser& arg_parser) {
    vocab_output = arg_parser.getopt("-save-vocab");
    vocab_source = arg_parser.getopt("-read-vocab");
    gpu_id = stoi(arg_parser.getopt("-gpu"));
    debug = stoi(arg_parser.getopt("-debug"));
    max_vocab_size = stoull(arg_parser.getopt("-vocab-limit"));
    
    ModelConfig conf;
    conf.train_file = arg_parser.getopt("-train");
    conf.wordembedding_file = arg_parser.getopt("-word");
    conf.output_file = arg_parser.getopt("-output");
    conf.layer_size = stoi(arg_parser.getopt("-size"));
    conf.window_size = stoi(arg_parser.getopt("-window"));
    conf.sample_rate = stod(arg_parser.getopt("-sample"));
    conf.hierarchical_softmax = stoi(arg_parser.getopt("-hs"));
    conf.negative_sample = stoi(arg_parser.getopt("-negative"));
    conf.iterations = stoull(arg_parser.getopt("-iter"));
    conf.min_count = stoi(arg_parser.getopt("-min-count"));
    conf.cbow = stoi(arg_parser.getopt("-cbow"));
    if (conf.cbow) conf.alpha = 0.05;
    else conf.alpha = 0.025;
    if (arg_parser.getopt("-alpha") != "") {
        conf.alpha = stod(arg_parser.getopt("-alpha"));
    }
    conf.binary = stoi(arg_parser.getopt("-binary"));
    conf.rp_sample = stod(arg_parser.getopt("-sentence-sample"));
    return conf;
}

int main(int argc, const char *argv[]) {
    if (argc == 1) {
        print_usage();
        return 0;
    }
    ArgParser arg_parser = get_parser();
    arg_parser.parse_arg(argc, argv);
    ModelConfig conf = parse_args(arg_parser);

    Vocab vocab;
    
    llu words_count = 0;
    if (vocab_source == "") words_count = vocab.build_from_file(conf.train_file);
    else words_count = vocab.read_from_file(vocab_source);
    vocab.reduce(max_vocab_size);
    vocab.conclude();
    std::cout << "Vocab size: " << vocab.size() << '\n';
    std::cout << "Words in train file: " << words_count << std::endl;

    if (vocab_output != "") vocab.save_to_file(vocab_output);
    if (conf.output_file == "") return 0;

    if (cudaSetDevice(gpu_id) != cudaSuccess) {
        std::cerr << "Error using device " << gpu_id << std::endl;
        exit(-1);
    }

    train_model(vocab, conf);
    return 0;
}
