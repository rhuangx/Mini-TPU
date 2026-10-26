/// The assembler: splits each line into words, builds an Instr or a data block, and encodes with isa.cpp's encode().

#include "isa/asm.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <utility>
#include <vector>

#include "common/config.h"

namespace {

struct Token {
    std::string text;
    u32         col = 0;   // 1-based column where the word starts
};

// A plain char can be negative, which std::isspace and std::isdigit do not accept.
bool is_space(char c) {
    return std::isspace(static_cast<unsigned char>(c)) != 0;
}

bool is_digit(char c) {
    return std::isdigit(static_cast<unsigned char>(c)) != 0;
}

bool is_hex(const std::string& text) {
    if (text.size() <= 2) {
        return false;
    }
    return text[0] == '0' && (text[1] == 'x' || text[1] == 'X');
}

bool contains(const std::vector<std::string>& list, const std::string& item) {
    return std::find(list.begin(), list.end(), item) != list.end();
}

std::vector<Token> tokenize(const std::string& line) {
    std::vector<Token> tokens;
    std::size_t i = 0;

    while (i < line.size()) {
        if (line[i] == '#') {
            break;
        }
        if (is_space(line[i])) {
            ++i;
            continue;
        }

        const std::size_t start = i;
        while (i < line.size() && line[i] != '#' && !is_space(line[i])) {
            ++i;
        }

        Token token;
        token.text = line.substr(start, i - start);
        token.col  = static_cast<u32>(start + 1);
        tokens.push_back(token);
    }
    return tokens;
}

// Decimal (optionally negative) or 0x hex; the whole text must be the number.
bool parse_int(const std::string& text, std::int64_t& value) {
    const char* first = text.data();
    const char* last  = text.data() + text.size();
    int base = 10;

    if (is_hex(text)) {
        first += 2;
        base = 16;
        if (*first == '-') {
            return false;
        }
    }

    const std::from_chars_result result = std::from_chars(first, last, value, base);
    const bool parsed       = result.ec == std::errc();
    const bool used_it_all  = result.ptr == last;
    return parsed && used_it_all;
}

// The keys each instruction takes, in the order error messages list them.
std::vector<std::string> operand_keys(Op op) {
    if (op == Op::ReadHostMemory || op == Op::WriteHostMemory) {
        return {"host", "ub", "rows"};
    }
    if (op == Op::ReadWeights) {
        return {"tile"};
    }
    if (op == Op::MatrixMultiply) {
        return {"ub", "acc", "rows", "accumulate", "new_weights"};
    }
    if (op == Op::Activate) {
        return {"acc", "ub", "rows", "shift", "function"};
    }
    return {};
}

// Keys an instruction also accepts but does not require.
std::vector<std::string> optional_keys(Op op) {
    if (op == Op::Activate) {
        return {"pool", "size", "width", "stride"};
    }
    return {};
}

void set_operand(Instr& in, const std::string& key, u32 value) {
    if (key == "host") {
        in.host_row = value;
    } else if (key == "ub") {
        in.ub_row = value;
    } else if (key == "rows") {
        in.rows = value;
    } else if (key == "tile") {
        in.tile = value;
    } else if (key == "acc") {
        in.acc_row = value;
    } else if (key == "accumulate") {
        in.accumulate = value;
    } else if (key == "new_weights") {
        in.new_weights = value;
    } else if (key == "shift") {
        in.shift = value;
    } else if (key == "size") {
        in.pool_size = value;
    } else if (key == "width") {
        in.pool_width = value;
    } else if (key == "stride") {
        in.pool_stride = value;
    }
}

class Assembler {
public:
    explicit Assembler(std::string name) : name_(std::move(name)) {}

    Program run(const std::string& source) {
        std::istringstream lines(source);
        std::string line;

        while (std::getline(lines, line)) {
            ++line_;
            const std::vector<Token> tokens = tokenize(line);
            if (tokens.empty()) {
                continue;
            }

            const char first = tokens[0].text[0];
            if (first == '.') {
                directive(tokens);
            } else if (first == '-' || is_digit(first)) {
                data(tokens);
            } else {
                instruction(tokens);
            }
        }
        return std::move(prog_);
    }

private:
    std::string name_;
    u32         line_  = 0;
    DataBlock*  block_ = nullptr;   // the open data block, or none
    Program     prog_;

    [[noreturn]] void fail(u32 col, const std::string& msg) const {
        const std::string where = name_ + ":" + std::to_string(line_) + ":" + std::to_string(col);
        throw AsmError(where + ": error: " + msg);
    }

    std::int64_t number(const Token& token) const {
        std::int64_t value = 0;
        if (!parse_int(token.text, value)) {
            fail(token.col, "'" + token.text + "' is not a number");
        }
        return value;
    }

    // `.host ROW` or `.weights TILE [ROW]`: later data lines fill a block starting there.
    void directive(const std::vector<Token>& tokens) {
        const Token& name = tokens[0];
        const bool is_host = name.text == ".host";
        if (!is_host && name.text != ".weights") {
            fail(name.col, "unknown directive '" + name.text + "' (expects .host or .weights)");
        }
        if (is_host && tokens.size() != 2) {
            fail(name.col, ".host takes one address");
        }
        if (!is_host && (tokens.size() < 2 || tokens.size() > 3)) {
            fail(name.col, ".weights takes a tile and an optional row");
        }

        const std::int64_t address = number(tokens[1]);
        if (address < 0 || address > 0xFFFFFFFF) {
            fail(tokens[1].col, name.text + " address " + tokens[1].text + " is out of range");
        }

        DataBlock block;
        if (name.text == ".host") {
            block.addr = static_cast<u64>(address) * v1::kMxuDim;       // host memory is addressed in rows
            prog_.host.push_back(block);
            block_ = &prog_.host.back();
        } else {
            // Weight Memory is addressed in tiles; the optional row picks one of the tile's 256 rows of 256 bytes.
            std::int64_t row = 0;
            if (tokens.size() == 3) {
                row = number(tokens[2]);
                if (row < 0 || row > 255) {
                    fail(tokens[2].col, "row " + tokens[2].text + " is past the tile's last row 255");
                }
            }
            block.addr = static_cast<u64>(address) * v1::kTileBytes + static_cast<u64>(row) * v1::kMxuDim;
            prog_.weights.push_back(block);
            block_ = &prog_.weights.back();
        }
    }

    // A line of int8 values, appended to the open block.
    void data(const std::vector<Token>& tokens) {
        if (block_ == nullptr) {
            fail(tokens[0].col, "data outside a .host or .weights block");
        }

        for (const Token& token : tokens) {
            const std::int64_t value = number(token);

            bool fits = false;
            if (is_hex(token.text)) {
                fits = value <= 0xFF;
            } else {
                fits = value >= -128 && value <= 127;
            }
            if (!fits) {
                fail(token.col, "value " + token.text + " is not an int8 (-128..127 or 0x00..0xFF)");
            }

            // Hex is a bit pattern: 0xFF is the byte 11111111, which as an int8 is -1.
            const u8 bits = static_cast<u8>(value);
            block_->bytes.push_back(static_cast<i8>(bits));
        }
    }

    Op find_op(const Token& mnemonic) const {
        for (Op op : kAllOps) {
            if (mnemonic.text == op_name(op)) {
                return op;
            }
        }
        fail(mnemonic.col, "unknown instruction '" + mnemonic.text + "'");
    }

    // function= takes a name, not a number.
    ActivationFunction find_function(const Token& name) const {
        for (ActivationFunction function : kAllFunctions) {
            if (name.text == function_name(function)) {
                return function;
            }
        }
        fail(name.col, "unknown function '" + name.text + "' (expects identity, relu, sigmoid, tanh)");
    }

    // pool= takes a name, not a number.
    Pooling find_pooling(const Token& name) const {
        for (Pooling pool : kAllPoolings) {
            if (name.text == pooling_name(pool)) {
                return pool;
            }
        }
        fail(name.col, "unknown pooling '" + name.text + "' (expects none, max, avg)");
    }

    // `Op key=value key=value ...`
    void instruction(const std::vector<Token>& tokens) {
        block_ = nullptr;   // an instruction ends any open data block

        const Token& mnemonic = tokens[0];
        Instr in;
        in.op = find_op(mnemonic);

        const std::vector<std::string> keys     = operand_keys(in.op);
        const std::vector<std::string> optional = optional_keys(in.op);
        std::vector<std::string> given;

        for (std::size_t i = 1; i < tokens.size(); ++i) {
            const Token& token = tokens[i];

            const std::size_t equals = token.text.find('=');
            if (equals == std::string::npos) {
                fail(token.col, "expected key=value, got '" + token.text + "'");
            }
            const std::string key = token.text.substr(0, equals);

            if (!contains(keys, key) && !contains(optional, key)) {
                fail(token.col, mnemonic.text + " has no operand '" + key + "'" + expects_hint(keys, optional));
            }
            if (contains(given, key)) {
                fail(token.col, "operand '" + key + "' given twice");
            }
            given.push_back(key);

            Token value_token;
            value_token.text = token.text.substr(equals + 1);
            value_token.col  = token.col + static_cast<u32>(equals) + 1;   // errors point at the value, not the key

            if (key == "function") {
                in.function = find_function(value_token);
                continue;
            }
            if (key == "pool") {
                in.pool = find_pooling(value_token);
                continue;
            }

            const std::int64_t value = number(value_token);
            if (value < 0 || value > 0xFFFFFFFF) {
                fail(token.col, key + "=" + value_token.text + " is out of range");
            }
            set_operand(in, key, static_cast<u32>(value));
        }

        for (const std::string& key : keys) {
            if (!contains(given, key)) {
                fail(mnemonic.col, mnemonic.text + " is missing operand '" + key + "'");
            }
        }

        // encode() knows each field's width and throws if a value does not fit.
        try {
            prog_.code.push_back(encode(in));
        } catch (const std::out_of_range& e) {
            fail(mnemonic.col, e.what());
        }
    }

    static std::string expects_hint(const std::vector<std::string>& keys, const std::vector<std::string>& optional) {
        if (keys.empty()) {
            return " (it takes none)";
        }
        std::string hint = " (expects " + join(keys);
        if (!optional.empty()) {
            hint += "; optional " + join(optional);
        }
        return hint + ")";
    }

    static std::string join(const std::vector<std::string>& words) {
        std::string text;
        for (std::size_t k = 0; k < words.size(); ++k) {
            if (k > 0) {
                text += ", ";
            }
            text += words[k];
        }
        return text;
    }
};

}  // namespace

namespace {

// One 256-byte row of data: the directive that places it, then its values 16 per line, trailing zeros left out.
std::string data_row(const std::string& directive, const i8* bytes, u32 count) {
    u32 used = count;
    while (used > 0 && bytes[used - 1] == 0) {
        used = used - 1;
    }
    if (used == 0) {
        return "";   // memory starts zeroed, so an all-zero row needs no text
    }

    std::string text = directive + "\n";
    for (u32 i = 0; i < used; ++i) {
        text += std::to_string(static_cast<int>(bytes[i]));
        const bool end_of_line = (i % 16 == 15) || (i + 1 == used);
        if (end_of_line) {
            text += "\n";
        } else {
            text += " ";
        }
    }
    return text;
}

// Every row a data block covers; blocks from the compiler always start on a row.
std::string data_text(const std::vector<DataBlock>& blocks, bool weights) {
    const u64 row_bytes = v1::kMxuDim;
    std::string text;
    for (const DataBlock& block : blocks) {
        if (block.addr % row_bytes != 0) {
            throw std::invalid_argument("a data block at byte " + std::to_string(block.addr) + " does not start on a row");
        }
        for (u64 offset = 0; offset < block.bytes.size(); offset += row_bytes) {
            const u64 row = (block.addr + offset) / row_bytes;
            std::string directive = ".host " + std::to_string(row);
            if (weights) {
                const u64 rows_per_tile = v1::kTileBytes / row_bytes;
                directive = ".weights " + std::to_string(row / rows_per_tile) + " " + std::to_string(row % rows_per_tile);
            }
            const u64 left  = block.bytes.size() - offset;
            const u32 count = static_cast<u32>(std::min<u64>(left, row_bytes));
            text += data_row(directive, block.bytes.data() + offset, count);
        }
    }
    return text;
}

}  // namespace

std::string program_text(const Program& program, const std::string& header) {
    std::string text = "# " + header + "\n\n";
    text += data_text(program.host, false);
    text += data_text(program.weights, true);
    text += "\n";
    for (const InstrBytes& word : program.code) {
        text += disasm(decode(word)) + "\n";
    }
    return text;
}

Program assemble(const std::string& source, const std::string& name) {
    Assembler assembler(name);
    return assembler.run(source);
}

Program assemble_file(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw AsmError(path + ": error: cannot open file");
    }
    std::ostringstream text;
    text << file.rdbuf();
    return assemble(text.str(), path);
}
