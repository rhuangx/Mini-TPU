/// Pooling: 2x2 windows over a 4x4 map, band timing, rounding, strided and overlapping windows, bad shapes, encoding.

#include <string>

#include "isa/asm.h"
#include "ref/ref_ops.h"
#include "test_framework.h"
#include "units/activation.h"

namespace {

// A 4x4 feature map in accumulator rows 0-15, pixel (y, x) in row y * 4 + x; column n holds 3 * row - 20 + n.
void fill_map(Accumulators& acc) {
    for (u32 row = 0; row < 16; ++row) {
        for (u32 n = 0; n < Accumulators::kCols; ++n) {
            acc.row(row)[n] = static_cast<i32>(3 * row) - 20 + static_cast<i32>(n);
        }
    }
}

int ub_value(const UnifiedBuffer& ub, u32 row, u32 col) {
    return ub.row(row)[col];
}

void tick_times(ActivationUnit& unit, int cycles) {
    for (int i = 0; i < cycles; ++i) {
        unit.tick();
    }
}

std::string error_of(const std::string& source) {
    try {
        assemble(source, "t.s");
    } catch (const AsmError& e) {
        return e.what();
    }
    return "no error";
}

}  // namespace

TEST(pool_average_rounds_halves_away_from_zero) {
    CHECK_EQ(static_cast<int>(average(6, 4)), 2);     // 1.5
    CHECK_EQ(static_cast<int>(average(5, 4)), 1);     // 1.25
    CHECK_EQ(static_cast<int>(average(-2, 4)), -1);   // -0.5
    CHECK_EQ(static_cast<int>(average(-6, 4)), -2);   // -1.5
    CHECK_EQ(static_cast<int>(average(4 * 127, 4)), 127);
}

TEST(pool_max_2x2_over_a_4x4_map) {
    Accumulators acc;
    UnifiedBuffer ub;
    fill_map(acc);
    ActivationUnit unit(acc, ub);

    unit.start(0, 20, 16, 0, ActivationFunction::Identity, Pooling::Max, 2, 4, 0);
    tick_times(unit, 7);
    CHECK_EQ(ub_value(ub, 20, 0), 0);   // band 0 (image rows 0-1) is not complete until its 8th pixel
    unit.tick();
    CHECK_EQ(ub_value(ub, 20, 0), 3 * 5 - 20);   // window {0, 1, 4, 5}: the largest is row 5
    CHECK_EQ(ub_value(ub, 21, 0), 3 * 7 - 20);   // window {2, 3, 6, 7}

    tick_times(unit, 8);
    CHECK(!unit.busy());
    CHECK_EQ(ub_value(ub, 22, 0), 3 * 13 - 20);
    CHECK_EQ(ub_value(ub, 23, 0), 3 * 15 - 20);
    CHECK_EQ(ub_value(ub, 23, 10), 3 * 15 - 20 + 10);
    CHECK_EQ(ub_value(ub, 24, 0), 0);   // 16 input rows make exactly 4 pooled rows
}

TEST(pool_average_2x2_over_a_4x4_map) {
    Accumulators acc;
    UnifiedBuffer ub;
    fill_map(acc);
    ActivationUnit unit(acc, ub);

    unit.start(0, 0, 16, 0, ActivationFunction::Relu, Pooling::Average, 2, 4, 0);
    tick_times(unit, 16);

    // Window {0, 1, 4, 5} at column 0 is -20, -17, -8, -5; relu makes them all 0.
    CHECK_EQ(ub_value(ub, 0, 0), 0);
    // Window {10, 11, 14, 15} at column 0 is 10, 13, 22, 25: sum 70, average 17.5, rounds to 18.
    CHECK_EQ(ub_value(ub, 3, 0), 18);
    // Window {2, 3, 6, 7} at column 20 is 6, 9, 18, 21: sum 54, average 13.5, rounds to 14.
    CHECK_EQ(ub_value(ub, 1, 20), 14);
}

TEST(pool_needs_whole_windows) {
    Accumulators acc;
    UnifiedBuffer ub;
    ActivationUnit unit(acc, ub);

    CHECK_THROWS(unit.start(0, 0, 20, 0, ActivationFunction::Identity, Pooling::Max, 2, 5, 0));    // width 5 is odd
    CHECK_THROWS(unit.start(0, 0, 12, 0, ActivationFunction::Identity, Pooling::Max, 2, 4, 0));    // 12 rows is 1.5 bands
    CHECK(!unit.busy());
    CHECK_EQ(activate_output_rows(64, Pooling::Average, 2, 8, 0), u32{16});
    CHECK_EQ(activate_output_rows(64, Pooling::None, 0, 0, 0), u32{64});
}

TEST(pool_encoding_and_syntax) {
    Instr in;
    in.op         = Op::Activate;
    in.function   = ActivationFunction::Tanh;
    in.pool       = Pooling::Max;
    in.pool_size  = 2;
    in.pool_width = 8;

    const InstrBytes word = encode(in);
    CHECK_EQ(static_cast<int>(word[1]), 0x27);   // bits 0-1 tanh (3), bits 2-3 max (1), bits 4-6 size (2)
    CHECK_EQ(static_cast<int>(word[11]), 8);

    const Instr back = decode(word);
    CHECK(back.pool == Pooling::Max);
    CHECK_EQ(back.pool_size, u32{2});
    CHECK_EQ(back.pool_width, u32{8});
    CHECK_EQ(disasm(in), std::string("Activate acc=0x0 ub=0x0 rows=0 shift=0 function=tanh pool=max size=2 width=8"));

    const Program p = assemble(disasm(in));
    CHECK(p.code.size() == 1 && p.code[0] == word);

    CHECK_EQ(error_of("Activate acc=0 ub=0 rows=4 shift=0 function=relu pool=min size=2 width=2"),
             std::string("t.s:1:55: error: unknown pooling 'min' (expects none, max, avg)"));
    CHECK_EQ(error_of("Activate acc=0 ub=0 rows=4 shift=0 function=relu pool=max size=8 width=2"),
             std::string("t.s:1:1: error: size=8 must be 1 to 7"));
    CHECK_EQ(error_of("Activate acc=0 ub=0 rows=4 shift=0 function=relu size=2"),
             std::string("t.s:1:1: error: size= and width= only go with pool=max or pool=avg"));
    CHECK_EQ(error_of("Activate acc=0 ub=0 rows=4 shift=0 function=relu pool=avg size=2"),
             std::string("t.s:1:1: error: pooling needs width=, the feature map's width in rows"));
    CHECK_EQ(error_of("Activate acc=0 ub=0 rows=4 shift=0 function=relu pad=2"),
             std::string("t.s:1:50: error: Activate has no operand 'pad' (expects acc, ub, rows, shift, function; "
                         "optional pool, size, width, stride)"));
}

TEST(pool_overlapping_3x3_windows_2_apart) {
    // A 5x5 map in accumulator rows 0-24: windows at rows and columns {0, 1, 2} and {2, 3, 4} share their middle line.
    Accumulators acc;
    UnifiedBuffer ub;
    Matrix8 map(25, Accumulators::kCols);
    for (u32 row = 0; row < 25; ++row) {
        for (u32 n = 0; n < Accumulators::kCols; ++n) {
            const i32 value = static_cast<i32>((row * 37 + n * 11) % 101) - 50;
            acc.row(row)[n]  = value;
            map.at(row, n)   = static_cast<i8>(value);
        }
    }
    ActivationUnit unit(acc, ub);

    unit.start(0, 40, 25, 0, ActivationFunction::Identity, Pooling::Max, 3, 5, 2);
    tick_times(unit, 14);
    CHECK_EQ(ub_value(ub, 40, 0), 0);   // pooled row 0 needs image rows 0-2, which end with the 15th pixel
    unit.tick();
    CHECK(ub.row(40)[0] != 0 || ub.row(41)[0] != 0);
    tick_times(unit, 10);
    CHECK(!unit.busy());

    const Matrix8 expected = ref_pool(map, 5, 3, Pooling::Max, 2);
    CHECK_EQ(expected.rows, u32{4});
    for (u32 r = 0; r < 4; ++r) {
        for (u32 n = 0; n < Accumulators::kCols; ++n) {
            CHECK_EQ(static_cast<int>(ub.row(40 + r)[n]), static_cast<int>(expected.at(r, n)));
        }
    }
    CHECK_EQ(activate_output_rows(25, Pooling::Max, 3, 5, 2), u32{4});
}

TEST(pool_strided_window_skips_pixels_past_it) {
    // A 7x7 average over a map 9 pixels wide and 7 high reads all 63 rows but averages only the 49 in its window.
    Accumulators acc;
    UnifiedBuffer ub;
    long long sum = 0;
    for (u32 row = 0; row < 63; ++row) {
        const u32 x = row % 9;
        const i32 value = x < 7 ? static_cast<i32>(row % 23) : 100;   // the two columns past the window hold 100
        for (u32 n = 0; n < Accumulators::kCols; ++n) {
            acc.row(row)[n] = value;
        }
        if (x < 7) {
            sum = sum + value;
        }
    }
    ActivationUnit unit(acc, ub);

    unit.start(0, 0, 63, 0, ActivationFunction::Relu, Pooling::Average, 7, 9, 7);
    tick_times(unit, 63);
    CHECK_EQ(static_cast<int>(ub.row(0)[5]), static_cast<int>(average(static_cast<i32>(sum), 49)));
    CHECK_EQ(static_cast<int>(ub.row(1)[5]), 0);   // one window fits, so one row is written
    CHECK_EQ(activate_output_rows(63, Pooling::Average, 7, 9, 7), u32{1});
}

TEST(pool_strided_needs_whole_image_rows) {
    Accumulators acc;
    UnifiedBuffer ub;
    ActivationUnit unit(acc, ub);

    CHECK_THROWS(unit.start(0, 0, 26, 0, ActivationFunction::Identity, Pooling::Max, 3, 5, 2));   // 5.2 image rows
    CHECK_THROWS(unit.start(0, 0, 10, 0, ActivationFunction::Identity, Pooling::Max, 3, 5, 2));   // 2 rows: no window fits
    CHECK(!unit.busy());
}

TEST(pool_stride_encoding_and_syntax) {
    Instr in;
    in.op          = Op::Activate;
    in.function    = ActivationFunction::Relu;
    in.pool        = Pooling::Max;
    in.pool_size   = 3;
    in.pool_width  = 113;
    in.pool_stride = 2;

    const InstrBytes word = encode(in);
    CHECK_EQ(static_cast<int>(word[2]), 2);   // byte 2, bits 0-2
    CHECK_EQ(decode(word).pool_stride, u32{2});
    CHECK_EQ(disasm(in), std::string("Activate acc=0x0 ub=0x0 rows=0 shift=0 function=relu pool=max size=3 width=113 stride=2"));

    const Program p = assemble(disasm(in));
    CHECK(p.code.size() == 1 && p.code[0] == word);

    CHECK_EQ(error_of("Activate acc=0 ub=0 rows=4 shift=0 function=relu stride=2"),
             std::string("t.s:1:1: error: stride= only goes with pool=max or pool=avg"));
    CHECK_EQ(error_of("Activate acc=0 ub=0 rows=4 shift=0 function=relu pool=max size=2 width=2 stride=8"),
             std::string("t.s:1:1: error: stride=8 must be 0 to 7"));
}
