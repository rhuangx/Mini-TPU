/// The int32-to-int8 arithmetic for each function, pooling, and the activation unit's one-row-per-cycle loop.

#include "units/activation.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

i8 saturate(std::int64_t value) {
    if (value > 127) {
        return 127;
    }
    if (value < -128) {
        return -128;
    }
    return static_cast<i8>(value);
}

// value / 2^shift to the nearest integer, halves away from zero; 64 bits, so negating -2^31 cannot overflow.
std::int64_t shift_and_round(i32 value, u32 shift) {
    if (shift == 0) {
        return value;
    }

    const std::int64_t half = std::int64_t{1} << (shift - 1);
    const std::int64_t wide = value;
    if (wide >= 0) {
        return (wide + half) >> shift;
    }

    const std::int64_t magnitude = -wide;
    return -((magnitude + half) >> shift);
}

// The accumulator as the real number it stands for; dividing by a power of two is exact in a double.
double real_value(i32 value, u32 shift) {
    const double scale = static_cast<double>(std::uint64_t{1} << shift);
    return static_cast<double>(value) / scale;
}

// y in Q0.7: the int8 that best stands for y at 1/128 per step; std::round takes halves away from zero.
i8 to_q0_7(double y) {
    const double scaled = std::round(y * 128.0);
    return saturate(static_cast<std::int64_t>(scaled));
}

bool overlaps(u32 first_a, u32 count_a, u32 first_b, u32 count_b) {
    if (count_a == 0 || count_b == 0) {
        return false;
    }
    const u64 end_a = static_cast<u64>(first_a) + count_a;
    const u64 end_b = static_cast<u64>(first_b) + count_b;
    return first_a < end_b && first_b < end_a;
}

// Checked as two steps rather than `first + count > limit`, which a huge count could wrap around.
void check_rows(const std::string& memory, u32 first, u32 count, u32 limit) {
    bool fits = true;
    if (count > limit) {
        fits = false;
    } else if (first > limit - count) {
        fits = false;
    }

    if (!fits) {
        const std::string range = "[" + std::to_string(first) + ", +" + std::to_string(count) + ")";
        throw std::out_of_range(memory + " rows " + range + " run past the last row " + std::to_string(limit - 1));
    }
}

}  // namespace

i8 activate(i32 value, u32 shift, ActivationFunction function) {
    switch (function) {
        case ActivationFunction::Identity: {
            return saturate(shift_and_round(value, shift));
        }
        case ActivationFunction::Relu: {
            const i8 out = saturate(shift_and_round(value, shift));
            if (out < 0) {
                return 0;
            }
            return out;
        }
        case ActivationFunction::Sigmoid: {
            const double x = real_value(value, shift);
            const double y = 1.0 / (1.0 + std::exp(-x));
            return to_q0_7(y);
        }
        case ActivationFunction::Tanh: {
            const double x = real_value(value, shift);
            return to_q0_7(std::tanh(x));
        }
    }
    return 0;
}

i8 average(i32 sum, u32 count) {
    const std::int64_t wide = sum;
    const std::int64_t half = count / 2;
    if (wide >= 0) {
        return saturate((wide + half) / count);
    }
    return saturate(-((-wide + half) / count));
}

u32 activate_output_rows(u32 rows, Pooling pool, u32 pool_size) {
    if (pool == Pooling::None || pool_size == 0) {
        return rows;
    }
    return rows / (pool_size * pool_size);
}

ActivationUnit::ActivationUnit(const Accumulators& acc, UnifiedBuffer& ub) : acc_(acc), ub_(ub) {}

void ActivationUnit::start(u32 acc_row, u32 ub_row, u32 rows, u32 shift, ActivationFunction function, Pooling pool,
                           u32 pool_size, u32 pool_width) {
    if (busy()) {
        throw std::logic_error("activation unit: an Activate is already running");
    }

    // Pooling needs whole windows: the width splits into windows, and the rows make whole bands of window-high image rows.
    if (pool != Pooling::None) {
        if (pool_width % pool_size != 0) {
            throw std::invalid_argument("pool width " + std::to_string(pool_width) + " is not a multiple of size " +
                                        std::to_string(pool_size));
        }
        const u32 band_rows = pool_width * pool_size;
        if (rows % band_rows != 0) {
            throw std::invalid_argument("pool rows " + std::to_string(rows) + " are not whole bands of " +
                                        std::to_string(pool_size) + " image rows " + std::to_string(pool_width) +
                                        " pixels wide");
        }
    }

    // Both ranges are checked before any row moves, so a bad Activate changes nothing.
    check_rows("acc", acc_row, rows, Accumulators::kRows);
    check_rows("ub", ub_row, activate_output_rows(rows, pool, pool_size), UnifiedBuffer::kRows);

    acc_row_    = acc_row;
    ub_row_     = ub_row;
    rows_total_ = rows;
    rows_done_  = 0;
    shift_      = shift;
    function_   = function;
    pool_       = pool;
    pool_size_  = pool_size;
    pool_width_ = pool_width;

    line_.clear();
    if (pool != Pooling::None) {
        const u32 pooled_per_band = pool_width / pool_size;
        line_.assign(static_cast<std::size_t>(pooled_per_band) * Accumulators::kCols, 0);
    }
}

void ActivationUnit::tick() {
    if (!busy()) {
        return;
    }

    const i32* in = acc_.row(acc_row_ + rows_done_);
    std::vector<i8> activated(Accumulators::kCols);
    for (u32 n = 0; n < Accumulators::kCols; ++n) {
        activated[n] = activate(in[n], shift_, function_);
    }

    if (pool_ == Pooling::None) {
        i8* out = ub_.row(ub_row_ + rows_done_);
        for (u32 n = 0; n < Accumulators::kCols; ++n) {
            out[n] = activated[n];
        }
    } else {
        pool_row(activated);
    }

    rows_done_ = rows_done_ + 1;
}

// Input row i is pixel (y, x) of the feature map; it joins the window at pooled column x / size of its band.
void ActivationUnit::pool_row(const std::vector<i8>& activated) {
    const u32 i = rows_done_;
    const u32 y = i / pool_width_;
    const u32 x = i % pool_width_;
    const u32 slot = x / pool_size_;
    const bool first_in_window = (y % pool_size_ == 0) && (x % pool_size_ == 0);

    i32* running = line_.data() + static_cast<std::size_t>(slot) * Accumulators::kCols;
    for (u32 n = 0; n < Accumulators::kCols; ++n) {
        const i32 value = activated[n];
        if (first_in_window) {
            running[n] = value;
        } else if (pool_ == Pooling::Max) {
            running[n] = std::max(running[n], value);
        } else {
            running[n] = running[n] + value;
        }
    }

    // The band's last pixel completes every window in it, so the whole band of pooled pixels goes to the UB.
    const bool band_done = (y % pool_size_ == pool_size_ - 1) && (x == pool_width_ - 1);
    if (!band_done) {
        return;
    }

    const u32 pooled_per_band = pool_width_ / pool_size_;
    const u32 band            = y / pool_size_;
    const u32 window_count    = pool_size_ * pool_size_;
    for (u32 s = 0; s < pooled_per_band; ++s) {
        const i32* window = line_.data() + static_cast<std::size_t>(s) * Accumulators::kCols;
        i8*        out    = ub_.row(ub_row_ + band * pooled_per_band + s);
        for (u32 n = 0; n < Accumulators::kCols; ++n) {
            if (pool_ == Pooling::Max) {
                out[n] = static_cast<i8>(window[n]);
            } else {
                out[n] = average(window[n], window_count);
            }
        }
    }
}

void ActivationUnit::reset() {
    acc_row_    = 0;
    ub_row_     = 0;
    rows_total_ = 0;
    rows_done_  = 0;
    shift_      = 0;
    function_   = ActivationFunction::Identity;
    pool_       = Pooling::None;
    pool_size_  = 0;
    pool_width_ = 0;
    line_.clear();
}

bool ActivationUnit::busy() const {
    return rows_done_ < rows_total_;
}

bool ActivationUnit::reads_acc_rows(u32 first, u32 count) const {
    if (!busy()) {
        return false;
    }
    return overlaps(acc_row_, rows_total_, first, count);
}

bool ActivationUnit::writes_ub_rows(u32 first, u32 count) const {
    if (!busy()) {
        return false;
    }
    return overlaps(ub_row_, activate_output_rows(rows_total_, pool_, pool_size_), first, count);
}

u32 ActivationUnit::acc_row() const {
    return acc_row_;
}

u32 ActivationUnit::ub_row() const {
    return ub_row_;
}

u32 ActivationUnit::rows_done() const {
    return rows_done_;
}

u32 ActivationUnit::rows_total() const {
    return rows_total_;
}

u32 ActivationUnit::shift() const {
    return shift_;
}

ActivationFunction ActivationUnit::function() const {
    return function_;
}

Pooling ActivationUnit::pool() const {
    return pool_;
}

u32 ActivationUnit::pool_size() const {
    return pool_size_;
}

u32 ActivationUnit::pool_width() const {
    return pool_width_;
}
