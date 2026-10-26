/// Command parsing for the terminal, and the text each command returns.

#include "ui/shell.h"

#include <sstream>
#include <stdexcept>

#include "isa/asm.h"

namespace {

std::vector<std::string> split_words(const std::string& line) {
    std::istringstream stream(line);
    std::vector<std::string> words;
    std::string word;
    while (stream >> word) {
        words.push_back(word);
    }
    return words;
}

}  // namespace

Shell::Shell(Tpu& tpu) : tpu_(tpu) {}

std::string Shell::execute(const std::string& line) {
    const std::vector<std::string> words = split_words(line);
    if (words.empty()) {
        return "";
    }

    try {
        const std::string& command = words[0];
        if (command == "quit") {
            quit_ = true;
            return "";
        }
        if (command == "load") {
            return load(words);
        }
        throw std::invalid_argument("unknown command '" + command + "': the terminal takes load FILE and quit");
    } catch (const AsmError& e) {
        return e.what();   // already reads "file:line:col: error: ..."
    } catch (const std::exception& e) {
        return std::string("error: ") + e.what();
    }
}

bool Shell::quit_requested() const {
    return quit_;
}

std::string Shell::load(const std::vector<std::string>& words) {
    if (words.size() != 2) {
        throw std::invalid_argument("usage: load FILE");
    }
    const std::string& path = words[1];
    const Program program = assemble_file(path);
    tpu_.load(program);

    u64 host_bytes = 0;
    for (const DataBlock& block : program.host) {
        host_bytes += block.bytes.size();
    }
    u64 weight_bytes = 0;
    for (const DataBlock& block : program.weights) {
        weight_bytes += block.bytes.size();
    }

    return "loaded " + path + ": " + std::to_string(program.code.size()) + " instructions, " +
           std::to_string(host_bytes) + " host bytes, " + std::to_string(weight_bytes) + " weight bytes";
}
