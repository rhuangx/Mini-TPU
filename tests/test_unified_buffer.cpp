/// Unified Buffer: size, zero start, row and byte views agree, and out-of-range access throws.

#include <limits>
#include <vector>

#include "mem/unified_buffer.h"
#include "test_framework.h"

TEST(ub_is_24_mib_of_zeros) {
    const UnifiedBuffer ub;
    const u32 last_row = UnifiedBuffer::kRows - 1;

    CHECK_EQ(UnifiedBuffer::kBytes, 24 * kMiB);
    CHECK_EQ(ub.row(0)[0], 0);
    CHECK_EQ(ub.row(last_row)[255], 0);
}

TEST(ub_row_writes_read_back) {
    UnifiedBuffer ub;
    const u32 last_row = UnifiedBuffer::kRows - 1;

    // Fill the last row with -128, -127, ..., 127.
    i8* row = ub.row(last_row);
    for (u32 col = 0; col < UnifiedBuffer::kRowBytes; ++col) {
        row[col] = static_cast<i8>(static_cast<int>(col) - 128);
    }

    CHECK_EQ(ub.row(last_row)[0], -128);
    CHECK_EQ(ub.row(last_row)[255], 127);
    CHECK_EQ(ub.row(last_row - 1)[255], 0);   // the row before it is untouched
}

TEST(ub_byte_ranges_are_row_major) {
    UnifiedBuffer ub;
    const std::vector<i8> src = {1, 2, 3, 4};
    const u64 row_2_col_254 = 2 * 256 + 254;   // the last two bytes of row 2, then the first two of row 3

    ub.write(row_2_col_254, src.data(), src.size());
    CHECK_EQ(ub.row(2)[254], 1);
    CHECK_EQ(ub.row(2)[255], 2);
    CHECK_EQ(ub.row(3)[0], 3);
    CHECK_EQ(ub.row(3)[1], 4);

    std::vector<i8> back(4, 0);
    ub.read(row_2_col_254, back.data(), back.size());
    CHECK(back == src);
}

TEST(ub_out_of_range_throws) {
    UnifiedBuffer ub;
    i8 byte = 0;
    const u64 huge = std::numeric_limits<u64>::max();   // addr + huge would wrap around to a small number

    CHECK_THROWS(ub.row(UnifiedBuffer::kRows));
    CHECK_THROWS(ub.read(UnifiedBuffer::kBytes, &byte, 1));
    CHECK_THROWS(ub.write(UnifiedBuffer::kBytes - 1, &byte, 2));
    CHECK_THROWS(ub.read(1, &byte, huge));

    ub.write(UnifiedBuffer::kBytes - 1, &byte, 1);   // the very last byte is fine
    ub.read(0, &byte, 0);                            // an empty range is fine
}
