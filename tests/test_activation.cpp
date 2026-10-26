/// Activation: rounding, saturation and each function's values, one row per cycle, the row hazards, and a whole layer.

#include <limits>
#include <string>
#include <vector>

#include "core/tpu.h"
#include "isa/asm.h"
#include "test_framework.h"

namespace {

constexpr u32 kDim = v1::kMxuDim;
constexpr i32 kMostPositive = std::numeric_limits<i32>::max();
constexpr i32 kMostNegative = std::numeric_limits<i32>::min();

// As an int, so a failed check prints a number rather than a character.
int activated(i32 value, u32 shift, ActivationFunction function) {
    return activate(value, shift, function);
}

// A small fixed pseudo-random sequence of int8 values, so every run uses the same data.
std::vector<i8> random_bytes(std::size_t count, u32 seed) {
    std::vector<i8> bytes(count);
    u32 state = seed;
    for (std::size_t i = 0; i < count; ++i) {
        state = state * 1664525u + 1013904223u;
        bytes[i] = static_cast<i8>(state >> 24);
    }
    return bytes;
}

// out[r][n] = sum over k of x[r][k] * w[k][n], computed directly.
std::vector<i32> plain_matmul(const std::vector<i8>& x, const std::vector<i8>& w, u32 rows) {
    std::vector<i32> out(static_cast<std::size_t>(rows) * kDim, 0);
    for (u32 r = 0; r < rows; ++r) {
        for (u32 n = 0; n < kDim; ++n) {
            i32 sum = 0;
            for (u32 k = 0; k < kDim; ++k) {
                sum = sum + static_cast<i32>(x[r * kDim + k]) * static_cast<i32>(w[k * kDim + n]);
            }
            out[r * kDim + n] = sum;
        }
    }
    return out;
}

// Runs `source` to Halt and returns the cycles charged to `stall`.
u64 stalled(const std::string& source, Stall stall) {
    Tpu tpu;
    tpu.load(assemble(source));
    tpu.run_to_halt();
    return tpu.stats().stalled(stall);
}

std::string error_of_run(const std::string& source) {
    Tpu tpu;
    tpu.load(assemble(source));
    try {
        tpu.run_to_halt();
    } catch (const std::runtime_error& e) {
        return e.what();
    }
    return "no error";
}

}  // namespace

TEST(act_shift_rounds_halves_away_from_zero) {
    const ActivationFunction identity = ActivationFunction::Identity;
    CHECK_EQ(activated(7, 0, identity), 7);
    CHECK_EQ(activated(3, 1, identity), 2);     // 1.5
    CHECK_EQ(activated(-3, 1, identity), -2);   // -1.5
    CHECK_EQ(activated(5, 2, identity), 1);     // 1.25
    CHECK_EQ(activated(-5, 2, identity), -1);   // -1.25
    CHECK_EQ(activated(6, 2, identity), 2);     // 1.5
    CHECK_EQ(activated(-6, 2, identity), -2);   // -1.5
    CHECK_EQ(activated(kMostNegative, 31, identity), -1);   // exactly -2^31 / 2^31
    CHECK_EQ(activated(kMostPositive, 31, identity), 1);    // just under 1
}

TEST(act_saturates_to_int8) {
    const ActivationFunction identity = ActivationFunction::Identity;
    CHECK_EQ(activated(127, 0, identity), 127);
    CHECK_EQ(activated(128, 0, identity), 127);
    CHECK_EQ(activated(-128, 0, identity), -128);
    CHECK_EQ(activated(-129, 0, identity), -128);
    CHECK_EQ(activated(255, 1, identity), 127);   // 127.5 rounds to 128, which saturates
    CHECK_EQ(activated(kMostPositive, 0, identity), 127);
    CHECK_EQ(activated(kMostNegative, 0, identity), -128);
}

TEST(act_relu_zeroes_negatives) {
    const ActivationFunction relu = ActivationFunction::Relu;
    CHECK_EQ(activated(5, 0, relu), 5);
    CHECK_EQ(activated(-5, 0, relu), 0);
    CHECK_EQ(activated(1000, 0, relu), 127);
    CHECK_EQ(activated(1, 1, relu), 1);    // 0.5 rounds up to 1
    CHECK_EQ(activated(-1, 1, relu), 0);   // -0.5 rounds to -1, then relu
}

TEST(act_sigmoid_and_tanh_give_q0_7) {
    const ActivationFunction sigmoid = ActivationFunction::Sigmoid;
    const ActivationFunction tanh    = ActivationFunction::Tanh;

    // shift=8 makes 256 stand for 1.0; outputs count in steps of 1/128.
    CHECK_EQ(activated(0, 0, sigmoid), 64);                // 0.5
    CHECK_EQ(activated(256, 8, sigmoid), 94);              // sigmoid(1) = 0.731, x128 = 93.6
    CHECK_EQ(activated(-256, 8, sigmoid), 34);             // sigmoid(-1) = 0.269, x128 = 34.4
    CHECK_EQ(activated(kMostPositive, 0, sigmoid), 127);   // 1.0 would be 128, which saturates
    CHECK_EQ(activated(kMostNegative, 0, sigmoid), 0);

    CHECK_EQ(activated(0, 0, tanh), 0);
    CHECK_EQ(activated(256, 8, tanh), 97);                 // tanh(1) = 0.762, x128 = 97.5
    CHECK_EQ(activated(-256, 8, tanh), -97);
    CHECK_EQ(activated(kMostPositive, 0, tanh), 127);
    CHECK_EQ(activated(kMostNegative, 0, tanh), -128);     // -1.0 is exactly -128
}

TEST(act_one_row_per_cycle) {
    Accumulators acc;
    UnifiedBuffer ub;
    ActivationUnit unit(acc, ub);

    acc.row(10)[0]   = 300;
    acc.row(10)[255] = -7;
    acc.row(11)[0]   = -300;
    unit.start(10, 20, 2, 1, ActivationFunction::Identity, Pooling::None, 0, 0, 0);
    CHECK(unit.busy());
    CHECK(unit.reads_acc_rows(11, 1));
    CHECK(unit.writes_ub_rows(21, 5));
    CHECK(!unit.writes_ub_rows(22, 5));

    unit.tick();   // acc row 10 -> UB row 20; row 21 is untouched
    CHECK_EQ(static_cast<int>(ub.row(20)[0]), 127);     // 150 saturates
    CHECK_EQ(static_cast<int>(ub.row(20)[255]), -4);    // -3.5
    CHECK_EQ(static_cast<int>(ub.row(21)[0]), 0);
    CHECK_EQ(unit.rows_done(), u32{1});

    unit.tick();
    CHECK_EQ(static_cast<int>(ub.row(21)[0]), -128);    // -150 saturates
    CHECK(!unit.busy());
    CHECK(!unit.reads_acc_rows(10, 2));
}

TEST(act_matches_a_plain_loop) {
    const u32 rows = 8;
    const std::vector<i8> x = random_bytes(rows * kDim, 3);
    const std::vector<i8> w = random_bytes(kDim * kDim, 4);

    // Each function into its own UB rows; shift 10 for identity and relu, 16 for sigmoid and tanh, so few values saturate.
    Program program = assemble(
        "Read_Host_Memory host=0 ub=0 rows=8\n"
        "Read_Weights tile=0\n"
        "MatrixMultiply ub=0 acc=0 rows=8 accumulate=0 new_weights=1\n"
        "Activate acc=0 ub=100 rows=8 shift=10 function=identity\n"
        "Activate acc=0 ub=200 rows=8 shift=10 function=relu\n"
        "Activate acc=0 ub=300 rows=8 shift=16 function=sigmoid\n"
        "Activate acc=0 ub=400 rows=8 shift=16 function=tanh\n"
        "Halt\n");
    program.host.push_back({0, x});
    program.weights.push_back({0, w});

    Tpu tpu;
    tpu.load(program);
    tpu.run_to_halt();

    const std::vector<i32> sums = plain_matmul(x, w, rows);
    u32 mismatches = 0;
    for (u32 r = 0; r < rows; ++r) {
        for (u32 n = 0; n < kDim; ++n) {
            const i32 sum = sums[r * kDim + n];
            const bool identity_ok = tpu.ub().row(100 + r)[n] == activate(sum, 10, ActivationFunction::Identity);
            const bool relu_ok     = tpu.ub().row(200 + r)[n] == activate(sum, 10, ActivationFunction::Relu);
            const bool sigmoid_ok  = tpu.ub().row(300 + r)[n] == activate(sum, 16, ActivationFunction::Sigmoid);
            const bool tanh_ok     = tpu.ub().row(400 + r)[n] == activate(sum, 16, ActivationFunction::Tanh);
            if (!identity_ok || !relu_ok || !sigmoid_ok || !tanh_ok) {
                mismatches = mismatches + 1;
            }
        }
    }
    CHECK_EQ(mismatches, u32{0});
}

TEST(act_waits_for_the_activate_before_it) {
    const char* source =
        "Activate acc=0 ub=0 rows=10 shift=0 function=identity\n"
        "Activate acc=10 ub=10 rows=10 shift=0 function=identity\n"
        "Halt\n";
    Tpu tpu;
    tpu.load(assemble(source));
    tpu.run_to_halt();

    // Rows on cycles 0-9, then 10-19; Halt issues on cycle 20.
    CHECK_EQ(tpu.stats().stalled(Stall::ActivationBusy), u64{9});
    CHECK_EQ(tpu.cycle(), Cycle{21});
}

TEST(act_waits_for_the_multiply_writing_its_rows) {
    const char* source =
        ".host 0\n"
        "1 2\n"
        ".weights 0 0\n"
        "3 -4\n"
        "Read_Host_Memory host=0 ub=0 rows=1\n"
        "Read_Weights tile=0\n"
        "MatrixMultiply ub=0 acc=0 rows=1 accumulate=0 new_weights=1\n"
        "Activate acc=0 ub=1 rows=1 shift=0 function=identity\n"
        "Halt\n";
    Tpu tpu;
    tpu.load(assemble(source));
    tpu.run_to_halt();

    // The multiply issues on cycle 1622 and runs 512 cycles; Activate waits through the 511 after it.
    CHECK_EQ(tpu.stats().stalled(Stall::AccNotReady), u64{511});
    CHECK_EQ(tpu.cycle(), Cycle{2136});
    CHECK_EQ(static_cast<int>(tpu.ub().row(1)[0]), 3);
    CHECK_EQ(static_cast<int>(tpu.ub().row(1)[1]), -4);
}

TEST(act_and_mxu_wait_for_each_others_rows) {
    const std::string weights_ready = "Read_Weights tile=0\nSync\n";

    // A multiply would overwrite accumulator rows 50-99 before the Activate reads them.
    const std::string acc_write_after_read =
        weights_ready +
        "Activate acc=0 ub=0 rows=100 shift=0 function=identity\n"
        "MatrixMultiply ub=200 acc=50 rows=1 accumulate=0 new_weights=1\n"
        "Halt\n";
    CHECK_EQ(stalled(acc_write_after_read, Stall::AccNotReady), u64{99});

    // A multiply would read UB row 50 before the Activate writes it.
    const std::string ub_read_after_write =
        weights_ready +
        "Activate acc=0 ub=0 rows=100 shift=0 function=identity\n"
        "MatrixMultiply ub=50 acc=200 rows=1 accumulate=0 new_weights=1\n"
        "Halt\n";
    CHECK_EQ(stalled(ub_read_after_write, Stall::UbNotReady), u64{99});

    // An Activate would overwrite UB row 0 while the multiply is still reading it.
    const std::string ub_write_after_read =
        weights_ready +
        "MatrixMultiply ub=0 acc=0 rows=1 accumulate=0 new_weights=1\n"
        "Activate acc=100 ub=0 rows=1 shift=0 function=identity\n"
        "Halt\n";
    CHECK_EQ(stalled(ub_write_after_read, Stall::UbNotReady), u64{511});
}

TEST(act_and_host_transfers_wait_for_each_others_rows) {
    // A one-row host transfer takes 12 cycles, so an Activate behind it waits 11.
    const char* write_then_activate =
        "Write_Host_Memory ub=0 host=0 rows=1\n"
        "Activate acc=0 ub=0 rows=1 shift=0 function=identity\n"
        "Halt\n";
    CHECK_EQ(stalled(write_then_activate, Stall::UbNotReady), u64{11});

    const char* read_then_activate =
        "Read_Host_Memory host=0 ub=0 rows=1\n"
        "Activate acc=0 ub=0 rows=1 shift=0 function=identity\n"
        "Halt\n";
    CHECK_EQ(stalled(read_then_activate, Stall::UbNotReady), u64{11});

    // A 50-row Activate takes 50 cycles, so a transfer of any of its rows waits 49.
    const char* activate_then_write =
        "Activate acc=0 ub=0 rows=50 shift=0 function=identity\n"
        "Write_Host_Memory ub=10 host=0 rows=1\n"
        "Halt\n";
    CHECK_EQ(stalled(activate_then_write, Stall::UbNotReady), u64{49});

    const char* activate_then_read =
        "Activate acc=0 ub=0 rows=50 shift=0 function=identity\n"
        "Read_Host_Memory host=0 ub=49 rows=1\n"
        "Halt\n";
    CHECK_EQ(stalled(activate_then_read, Stall::UbNotReady), u64{49});
}

TEST(act_bad_ranges_name_the_pc) {
    CHECK_EQ(error_of_run("Activate acc=4090 ub=0 rows=8 shift=0 function=identity\nHalt\n"),
             std::string("pc 0: Activate acc=0xFFA ub=0x0 rows=8 shift=0 function=identity: "
                         "acc rows [4090, +8) run past the last row 4095"));
    CHECK_EQ(error_of_run("Activate acc=0 ub=98300 rows=8 shift=0 function=relu\nHalt\n"),
             std::string("pc 0: Activate acc=0x0 ub=0x17FFC rows=8 shift=0 function=relu: "
                         "ub rows [98300, +8) run past the last row 98303"));
}
