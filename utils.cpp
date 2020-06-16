#include <iostream>
#include <stdexcept>
#include "utils.hpp"

void ArgParser::add_argument(std::string arg_key, std::string default_value) {
    result[arg_key] = default_value;
}

void ArgParser::parse_arg(int argc, const char *argv[]) {
    if (argc % 2 == 0) {
        throw std::invalid_argument("Invalid arguments");
    }
    for (int i = 1; i < argc; i += 2) {
        if (result.find(argv[i]) == result.end()) {
            throw std::invalid_argument("Unexpected argument '" + std::string(argv[i]) + "'");
        }
        result[argv[i]] = argv[i + 1];
    }
}

std::string ArgParser::getopt(std::string arg_key) {
    if (result.find(arg_key) == result.end()) {
        throw std::invalid_argument("Unknown argument '" + arg_key + "'");
    }
    return result[arg_key];
}

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
