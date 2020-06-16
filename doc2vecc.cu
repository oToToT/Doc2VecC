#include "utils.hpp"
#include "vocab.hpp"
#include <iostream>
#include <cstdlib>
#include <cinttypes>
using llf = double;
using llu = uint64_t;

int gpu_id;

llf sample_rate, alpha, rp_sample;
bool hierarchical_softmax, cbow, binary;
llu layer_size, iterations, max_vocab_size;
int window_size, negative_sample, min_count, debug;
std::string train_file, wordembedding_file, output_file, vocab_output, vocab_source;

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

void parse_args(ArgParser& arg_parser) {
    train_file = arg_parser.getopt("-train");
    wordembedding_file = arg_parser.getopt("-word");
    output_file = arg_parser.getopt("-output");
    layer_size = atoi(arg_parser.getopt("-size").c_str());
    window_size = atoi(arg_parser.getopt("-window").c_str());
    sample_rate = atof(arg_parser.getopt("-sample").c_str());
    hierarchical_softmax = atoi(arg_parser.getopt("-hs").c_str());
    negative_sample = atoi(arg_parser.getopt("-negative").c_str());
    gpu_id = atoi(arg_parser.getopt("-gpu").c_str());
    iterations = atoll(arg_parser.getopt("-iter").c_str());
    min_count = atoi(arg_parser.getopt("-min-count").c_str());
    cbow = atoi(arg_parser.getopt("-cbow").c_str());
    if (cbow) alpha = 0.05;
    else alpha = 0.025;
    alpha = atof(arg_parser.getopt("-alpha").c_str());
    debug = atoi(arg_parser.getopt("-debug").c_str());
    binary = atoi(arg_parser.getopt("-binary").c_str());
    vocab_output = arg_parser.getopt("-save-vocab");
    vocab_source = arg_parser.getopt("-read-vocab");
    rp_sample = atof(arg_parser.getopt("-sentence-sample").c_str());
    max_vocab_size = atoll(arg_parser.getopt("-vocab-limit").c_str());
}

int main(int argc, const char *argv[]) {
    if (argc == 1) {
        print_usage();
        return 0;
    }
    ArgParser arg_parser = get_parser();
    arg_parser.parse_arg(argc, argv);
    parse_args(arg_parser);

    Vocab vocab;
    
    llu words_count = 0;
    if (vocab_source == "") words_count = vocab.build_from_file(train_file);
    else words_count = vocab.read_from_file(vocab_source);
    vocab.reduce(max_vocab_size);
    std::cout << "Vocab size: " << vocab.size() << '\n';
    std::cout << "Words in train file: " << words_count << std::endl;

    if (vocab_output != "") vocab.save_to_file(vocab_output);

    if (output_file == "") return 0;


    return 0;
}
