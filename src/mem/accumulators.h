/// Accumulators: 4 MiB of int32 in 4096 rows of 256, where the MXU's column sums land.

#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "common/config.h"

class Accumulators {
public:
    static constexpr u32 kRows = v1::kAccRows;
    static constexpr u32 kCols = v1::kMxuDim;

    Accumulators() {
        values_.resize(static_cast<u64>(kRows) * kCols, 0);
    }

    // The pointer covers that row's 256 values. Only the MXU asks for it, to write.
    i32* row(u32 r) {
        check_row(r);
        return values_.data() + start_of_row(r);
    }

    const i32* row(u32 r) const {
        check_row(r);
        return values_.data() + start_of_row(r);
    }

private:
    std::vector<i32> values_;

    static u64 start_of_row(u32 r) {
        return static_cast<u64>(r) * kCols;
    }

    static void check_row(u32 r) {
        if (r >= kRows) {
            const std::string last = std::to_string(kRows - 1);
            throw std::out_of_range("acc row " + std::to_string(r) + " is past the last row " + last);
        }
    }
};
