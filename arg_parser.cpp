#include <iostream>
#include <stdexcept>
#include "arg_parser.hpp"

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

