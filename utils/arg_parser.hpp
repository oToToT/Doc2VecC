#ifndef ARG_PARSER_H
#define ARG_PARSER_H

#include <string>
#include <unordered_map>
#include <unordered_set>

class ArgParser {
 private:
  std::unordered_map<std::string, std::string> result_;

 public:
  void add_argument(std::string, std::string = "");
  void parse_arg(int, const char **);
  std::string getopt(std::string) const;
};

#endif
