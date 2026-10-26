/// Encoding, decoding and disassembly of the TPUv1 instructions: one switch case per instruction, fields inline.

#include "isa/isa.h"

#include <cstdio>
#include <stdexcept>

namespace {

// Reads `count` bytes starting at byte `first`, lowest byte first (little-endian).
u32 read_bytes(const InstrBytes& word, u32 first, u32 count) {
    u32 value = 0;
    for (u32 i = 0; i < count; ++i) {
        const u32 byte = word[first + i];
        value = value | (byte << (8 * i));
    }
    return value;
}

// Writes `value` into `count` bytes starting at byte `first`, lowest byte first; throws if it does not fit.
void write_bytes(InstrBytes& word, u32 first, u32 count, u64 value, const char* name) {
    const u32 bits = 8 * count;
    const u64 largest = (u64{1} << bits) - 1;
    if (value > largest) {
        const std::string field = std::string(name) + "=" + std::to_string(value);
        throw std::out_of_range(field + " does not fit in " + std::to_string(bits) + " bits");
    }

    for (u32 i = 0; i < count; ++i) {
        word[first + i] = static_cast<u8>((value >> (8 * i)) & 0xFF);
    }
}

void check_flag(u32 value, const char* name) {
    if (value > 1) {
        throw std::out_of_range(std::string(name) + "=" + std::to_string(value) + " must be 0 or 1");
    }
}

void check_shift(u32 value) {
    if (value > 31) {
        throw std::out_of_range("shift=" + std::to_string(value) + " must be 0 to 31");
    }
}

// With pooling, size is 1-7, width 1-255 and stride 0-7; without it all three stay 0, so the encoding is unchanged.
void check_pooling(const Instr& in) {
    if (in.pool == Pooling::None) {
        if (in.pool_size != 0 || in.pool_width != 0) {
            throw std::out_of_range("size= and width= only go with pool=max or pool=avg");
        }
        if (in.pool_stride != 0) {
            throw std::out_of_range("stride= only goes with pool=max or pool=avg");
        }
        return;
    }
    if (in.pool_size < 1 || in.pool_size > 7) {
        throw std::out_of_range("size=" + std::to_string(in.pool_size) + " must be 1 to 7");
    }
    if (in.pool_width < 1) {
        throw std::out_of_range("pooling needs width=, the feature map's width in rows");
    }
    if (in.pool_stride > 7) {
        throw std::out_of_range("stride=" + std::to_string(in.pool_stride) + " must be 0 to 7");
    }
}

std::string hex(u32 value) {
    char text[16];
    std::snprintf(text, sizeof text, "0x%X", value);
    return text;
}

}  // namespace

const char* op_name(Op op) {
    switch (op) {
        case Op::Nop:
            return "Nop";
        case Op::Halt:
            return "Halt";
        case Op::Sync:
            return "Sync";
        case Op::ReadHostMemory:
            return "Read_Host_Memory";
        case Op::WriteHostMemory:
            return "Write_Host_Memory";
        case Op::ReadWeights:
            return "Read_Weights";
        case Op::MatrixMultiply:
            return "MatrixMultiply";
        case Op::Activate:
            return "Activate";
    }
    return "?";
}

const char* function_name(ActivationFunction function) {
    switch (function) {
        case ActivationFunction::Identity:
            return "identity";
        case ActivationFunction::Relu:
            return "relu";
        case ActivationFunction::Sigmoid:
            return "sigmoid";
        case ActivationFunction::Tanh:
            return "tanh";
    }
    return "?";
}

const char* pooling_name(Pooling pool) {
    switch (pool) {
        case Pooling::None:
            return "none";
        case Pooling::Max:
            return "max";
        case Pooling::Average:
            return "avg";
    }
    return "?";
}

InstrBytes encode(const Instr& in) {
    InstrBytes word{};   // all twelve bytes start as zero
    word[0] = static_cast<u8>(in.op);

    switch (in.op) {
        case Op::Nop:
        case Op::Halt:
        case Op::Sync:
            break;

        case Op::ReadHostMemory:
        case Op::WriteHostMemory: {
            write_bytes(word, 3, 3, in.ub_row, "ub");       // bytes 3-5
            write_bytes(word, 6, 4, in.host_row, "host");   // bytes 6-9
            write_bytes(word, 10, 2, in.rows, "rows");      // bytes 10-11
            break;
        }

        case Op::ReadWeights: {
            write_bytes(word, 3, 4, in.tile, "tile");       // bytes 3-6
            break;
        }

        case Op::MatrixMultiply: {
            write_bytes(word, 3, 3, in.ub_row, "ub");       // bytes 3-5
            write_bytes(word, 6, 2, in.acc_row, "acc");     // bytes 6-7
            write_bytes(word, 8, 4, in.rows, "rows");       // bytes 8-11
            check_flag(in.accumulate, "accumulate");
            check_flag(in.new_weights, "new_weights");
            word[1] = static_cast<u8>(in.accumulate | (in.new_weights << 1));   // flag byte: bit 0, bit 1
            break;
        }

        case Op::Activate: {
            write_bytes(word, 3, 3, in.ub_row, "ub");       // bytes 3-5
            write_bytes(word, 6, 2, in.acc_row, "acc");     // bytes 6-7
            write_bytes(word, 8, 2, in.rows, "rows");       // bytes 8-9
            check_shift(in.shift);
            check_pooling(in);
            write_bytes(word, 11, 1, in.pool_width, "width");   // byte 11
            word[10] = static_cast<u8>(in.shift);               // byte 10
            word[2]  = static_cast<u8>(in.pool_stride);         // byte 2, bits 0-2
            const u32 flags = static_cast<u32>(in.function) | (static_cast<u32>(in.pool) << 2) | (in.pool_size << 4);
            word[1]  = static_cast<u8>(flags);                  // flag byte: bits 0-1 function, 2-3 pool, 4-6 size
            break;
        }
    }
    return word;
}

Instr decode(const InstrBytes& word) {
    Instr in;
    in.op = static_cast<Op>(word[0]);

    switch (in.op) {
        case Op::Nop:
        case Op::Halt:
        case Op::Sync:
            break;

        case Op::ReadHostMemory:
        case Op::WriteHostMemory: {
            in.ub_row   = read_bytes(word, 3, 3);    // bytes 3-5
            in.host_row = read_bytes(word, 6, 4);    // bytes 6-9
            in.rows     = read_bytes(word, 10, 2);   // bytes 10-11
            break;
        }

        case Op::ReadWeights: {
            in.tile = read_bytes(word, 3, 4);        // bytes 3-6
            break;
        }

        case Op::MatrixMultiply: {
            in.accumulate  = word[1] & 0x1;          // flag byte, bit 0
            in.new_weights = (word[1] >> 1) & 0x1;   // flag byte, bit 1
            in.ub_row      = read_bytes(word, 3, 3);
            in.acc_row     = read_bytes(word, 6, 2);
            in.rows        = read_bytes(word, 8, 4);
            break;
        }

        case Op::Activate: {
            in.function = static_cast<ActivationFunction>(word[1] & 0x3);   // flag byte, bits 0-1
            in.ub_row   = read_bytes(word, 3, 3);
            in.acc_row  = read_bytes(word, 6, 2);
            in.rows     = read_bytes(word, 8, 2);
            in.shift    = word[10] & 0x1F;                                  // byte 10, bits 0-4
            in.pool       = static_cast<Pooling>((word[1] >> 2) & 0x3);    // flag byte, bits 2-3
            in.pool_size  = (word[1] >> 4) & 0x7;                          // flag byte, bits 4-6
            in.pool_width = word[11];                                      // byte 11
            in.pool_stride = word[2] & 0x7;                                // byte 2, bits 0-2
            if (static_cast<u8>(in.pool) > 2) {
                throw std::invalid_argument("unknown pooling " + std::to_string(static_cast<u32>(in.pool)));
            }
            break;
        }

        default:
            throw std::invalid_argument("unknown opcode " + hex(word[0]));
    }
    return in;
}

std::string disasm(const Instr& in) {
    std::string name = op_name(in.op);

    switch (in.op) {
        case Op::Nop:
        case Op::Halt:
        case Op::Sync:
            return name;

        case Op::ReadHostMemory: {
            return name + " host=" + hex(in.host_row) + " ub=" + hex(in.ub_row) + " rows=" + std::to_string(in.rows);
        }

        case Op::WriteHostMemory: {
            return name + " ub=" + hex(in.ub_row) + " host=" + hex(in.host_row) + " rows=" + std::to_string(in.rows);
        }

        case Op::ReadWeights: {
            return name + " tile=" + hex(in.tile);
        }

        case Op::MatrixMultiply: {
            const std::string operands = " ub=" + hex(in.ub_row) + " acc=" + hex(in.acc_row) + " rows=" + std::to_string(in.rows);
            const std::string flags    = " accumulate=" + std::to_string(in.accumulate) + " new_weights=" + std::to_string(in.new_weights);
            return name + operands + flags;
        }

        case Op::Activate: {
            const std::string operands = " acc=" + hex(in.acc_row) + " ub=" + hex(in.ub_row) + " rows=" + std::to_string(in.rows);
            const std::string options  = " shift=" + std::to_string(in.shift) + " function=" + function_name(in.function);
            if (in.pool == Pooling::None) {
                return name + operands + options;
            }
            std::string pooling = " pool=" + std::string(pooling_name(in.pool)) + " size=" + std::to_string(in.pool_size) +
                                  " width=" + std::to_string(in.pool_width);
            if (in.pool_stride != 0) {
                pooling += " stride=" + std::to_string(in.pool_stride);
            }
            return name + operands + options + pooling;
        }
    }
    return name;
}
