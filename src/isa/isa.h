/// TPUv1's CISC instructions: opcodes, the decoded form, and 12-byte encode, decode and disassembly.

#pragma once

#include <array>
#include <string>
#include <vector>

#include "common/types.h"

// Every instruction is 12 bytes: byte 0 opcode, bytes 1-2 flags, bytes 3-11 operands.
inline constexpr u32 kInstrBytes = 12;
using InstrBytes = std::array<u8, kInstrBytes>;

enum class Op : u8 {
    Nop             = 0x00,
    Halt            = 0x01,
    Sync            = 0x02,   // wait until every unit is idle
    ReadHostMemory  = 0x10,   // host memory -> Unified Buffer
    WriteHostMemory = 0x11,   // Unified Buffer -> host memory
    ReadWeights     = 0x20,   // Weight Memory tile -> Weight FIFO
    MatrixMultiply  = 0x30,   // UB rows x weights -> accumulator rows
    Activate        = 0x40,   // accumulator rows -> activation function -> UB rows
};

// Every opcode, so the assembler can look names up instead of keeping its own list.
inline constexpr Op kAllOps[] = {
    Op::Nop,
    Op::Halt,
    Op::Sync,
    Op::ReadHostMemory,
    Op::WriteHostMemory,
    Op::ReadWeights,
    Op::MatrixMultiply,
    Op::Activate,
};

// Activate's nonlinear function, held in bits 0-1 of its flag byte.
enum class ActivationFunction : u8 {
    Identity = 0,
    Relu     = 1,
    Sigmoid  = 2,
    Tanh     = 3,
};

// Every function, so the assembler can look names up.
inline constexpr ActivationFunction kAllFunctions[] = {
    ActivationFunction::Identity,
    ActivationFunction::Relu,
    ActivationFunction::Sigmoid,
    ActivationFunction::Tanh,
};

// Activate's optional pooling, held in bits 2-3 of its flag byte.
enum class Pooling : u8 {
    None    = 0,
    Max     = 1,
    Average = 2,
};

// Every pooling kind, so the assembler can look names up.
inline constexpr Pooling kAllPoolings[] = {
    Pooling::None,
    Pooling::Max,
    Pooling::Average,
};

// One instruction with its fields unpacked into plain numbers.
struct Instr {
    Op  op          = Op::Nop;
    u32 ub_row      = 0;   // 24 bits: Unified Buffer row
    u32 host_row    = 0;   // 32 bits: host memory is addressed in 256-byte rows
    u32 rows        = 0;   // rows moved (16 bits) or multiplied (32 bits)
    u32 tile        = 0;   // 32 bits: Weight Memory tile (64 KiB each)
    u32 acc_row     = 0;   // 16 bits: accumulator row
    u32 accumulate  = 0;   // flag, 0 or 1: add into the accumulators instead of overwriting
    u32 new_weights = 0;   // flag, 0 or 1: switch to the shadow weight plane first
    u32 shift       = 0;   // 0-31: Activate reads each accumulator as value / 2^shift
    ActivationFunction function = ActivationFunction::Identity;   // Activate: identity, relu, sigmoid or tanh
    Pooling pool     = Pooling::None;   // Activate: none, max or avg over size x size windows of UB rows
    u32 pool_size    = 0;               // 1-7 with pooling, else 0: the window is pool_size x pool_size pixels
    u32 pool_width   = 0;               // 1-255 with pooling, else 0: the feature map's width, one UB row per pixel
    u32 pool_stride  = 0;               // 0-7 with pooling, else 0: pixels between windows; 0 means pool_size, windows side by side
};

// Bytes to place in a DRAM before the program runs.
struct DataBlock {
    u64             addr = 0;   // byte address
    std::vector<i8> bytes;
};

// A loadable program: what the host sends to the TPU, plus the data it expects in host memory and Weight Memory.
struct Program {
    std::vector<InstrBytes> code;
    std::vector<DataBlock>  host;
    std::vector<DataBlock>  weights;
};

// The instruction's name as the assembler and disassembler spell it, e.g. "Read_Host_Memory".
const char* op_name(Op op);

// The function's name as the assembler and disassembler spell it, e.g. "relu".
const char* function_name(ActivationFunction function);

// The pooling's name as the assembler and disassembler spell it: "none", "max" or "avg".
const char* pooling_name(Pooling pool);

// Instr to 12 bytes; throws std::out_of_range when a field does not fit its width.
InstrBytes encode(const Instr& in);

// 12 bytes to Instr; throws std::invalid_argument on an unknown opcode.
Instr decode(const InstrBytes& bytes);

// Instr to text, e.g. "Read_Host_Memory host=0x200 ub=0x10 rows=4".
std::string disasm(const Instr& in);
