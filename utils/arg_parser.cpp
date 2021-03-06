#include "arg_parser.hpp"

#include <iostream>
#include <stdexcept>

void ArgParser::add_argument(std::string arg_key, std::string default_value) {
  result_[arg_key] = default_value;
}

void ArgParser::parse_arg(int argc, const char **argv) {
  if (argc % 2 == 0) {
    throw std::invalid_argument("Invalid arguments");
  }
  for (int i = 1; i < argc; i += 2) {
    if (result_.find(argv[i]) == result_.end()) {
      throw std::invalid_argument("Unexpected argument '" +
                                  std::string(argv[i]) + "'");
    }
    result_[argv[i]] = argv[i + 1];
  }
}

std::string ArgParser::getopt(std::string arg_key) const {
  auto result_iterator = result_.find(arg_key);
  if (result_iterator == result_.end()) {
    throw std::invalid_argument("Unknown argument '" + arg_key + "'");
  }
  return result_iterator->second;
}
