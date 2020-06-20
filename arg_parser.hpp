#ifndef ARG_PARSER_H
#define ARG_PARSER_H

#include <string>
#include <unordered_set>
#include <unordered_map>

class ArgParser {
    private:
        std::unordered_map<std::string, std::string> result;
    public:
        void add_argument(std::string, std::string = "");
        void parse_arg(int, const char *[]);
        std::string getopt(std::string);
};

#endif