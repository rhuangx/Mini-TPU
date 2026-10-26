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

// The first window, along one direction, that covers pixel v: windows start every `stride` pixels and span `size`.
u32 first_window(u32 v, u32 size, u32 stride) {
    if (v + 1 <= size) {
        return 0;
    }
    return (v + 1 - size + stride - 1) / stride;
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

// Only windows wholly inside the map produce a row; pixels past the last window are read but go nowhere.
u32 activate_output_rows(u32 rows, Pooling pool, u32 pool_size, u32 pool_width, u32 pool_stride) {
    if (pool == Pooling::None) {
        return rows;
    }
    if (pool_size == 0 || pool_width == 0) {
        return 0;
    }
    u32 stride = pool_stride;
    if (stride == 0) {
        stride = pool_size;
    }
    const u32 height = rows / pool_width;
    if (height < pool_size || pool_width < pool_size) {
        return 0;
    }
    const u32 pooled_height = (height - pool_size) / stride + 1;
    const u32 pooled_width  = (pool_width - pool_size) / stride + 1;
    return pooled_height * pooled_width;
}

ActivationUnit::ActivationUnit(const Accumulators& acc, UnifiedBuffer& ub) : acc_(acc), ub_(ub) {}

void ActivationUnit::start(u32 acc_row, u32 ub_row, u32 rows, u32 shift, ActivationFunction function, Pooling pool,
                           u32 pool_size, u32 pool_width, u32 pool_stride) {
    if (busy()) {
        throw std::logic_error("activation unit: an Activate is already running");
    }

    if (pool != Pooling::None) {
        if (pool_size == 0 || pool_width == 0) {
            throw std::invalid_argument("pooling needs a size and a width");
        }
        if (pool_stride == 0) {
            // Side by side: the width splits into windows, and the rows make whole bands of window-high image rows.
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
        } else {
            // Strided: the rows make whole image rows, and at least one window fits each way.
            if (rows % pool_width != 0) {
                throw std::invalid_argument("pool rows " + std::to_string(rows) + " are not whole image rows " +
                                            std::to_string(pool_width) + " pixels wide");
            }
            if (pool_width < pool_size || rows / pool_width < pool_size) {
                throw std::invalid_argument("a " + std::to_string(pool_size) + "x" + std::to_string(pool_size) +
                                            " pool window does not fit a map " + std::to_string(pool_width) +
                                            " pixels wide and " + std::to_string(rows / pool_width) + " high");
            }
        }
    }

    // Both ranges are checked before any row moves, so a bad Activate changes nothing.
    check_rows("acc", acc_row, rows, Accumulators::kRows);
    check_rows("ub", ub_row, activate_output_rows(rows, pool, pool_size, pool_width, pool_stride), UnifiedBuffer::kRows);

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
        pool_stride_ = pool_stride;
        if (pool_stride_ == 0) {
            pool_stride_ = pool_size;
        }
        pooled_height_ = (rows / pool_width - pool_size) / pool_stride_ + 1;
        pooled_width_  = (pool_width - pool_size) / pool_stride_ + 1;
        open_rows_     = (pool_size + pool_stride_ - 1) / pool_stride_;
        line_.assign(static_cast<std::size_t>(open_rows_) * pooled_width_ * Accumulators::kCols, 0);
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

// Input row i is pixel (y, x) of the feature map; it joins every window that covers it, each held in the line-buffer
// row of its pooled row. With windows side by side that is exactly one window, and the line buffer is one band.
void ActivationUnit::pool_row(const std::vector<i8>& activated) {
    const u32 i = rows_done_;
    const u32 y = i / pool_width_;
    const u32 x = i % pool_width_;

    const u32 first_py = first_window(y, pool_size_, pool_stride_);
    const u32 last_py  = std::min(y / pool_stride_, pooled_height_ - 1);
    const u32 first_px = first_window(x, pool_size_, pool_stride_);
    const u32 last_px  = std::min(x / pool_stride_, pooled_width_ - 1);

    for (u32 py = first_py; py <= last_py; ++py) {
        for (u32 px = first_px; px <= last_px; ++px) {
            // A window's first pixel in reading order is its top-left one.
            const bool first_in_window = y == py * pool_stride_ && x == px * pool_stride_;
            const std::size_t slot = static_cast<std::size_t>(py % open_rows_) * pooled_width_ + px;
            i32* running = line_.data() + slot * Accumulators::kCols;
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
        }
    }

    // The last pixel of a window's bottom image row completes every window in that pooled row, so the row goes to the UB.
    const bool row_end = x == pool_width_ - 1;
    if (!row_end || y + 1 < pool_size_) {
        return;
    }
    const u32 window_top = y + 1 - pool_size_;
    if (window_top % pool_stride_ != 0 || window_top / pool_stride_ >= pooled_height_) {
        return;
    }

    const u32 py           = window_top / pool_stride_;
    const u32 window_count = pool_size_ * pool_size_;
    for (u32 px = 0; px < pooled_width_; ++px) {
        const std::size_t slot = static_cast<std::size_t>(py % open_rows_) * pooled_width_ + px;
        const i32* window = line_.data() + slot * Accumulators::kCols;
        i8*        out    = ub_.row(ub_row_ + py * pooled_width_ + px);
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
    acc_row_       = 0;
    ub_row_        = 0;
    rows_total_    = 0;
    rows_done_     = 0;
    shift_         = 0;
    function_      = ActivationFunction::Identity;
    pool_          = Pooling::None;
    pool_size_     = 0;
    pool_width_    = 0;
    pool_stride_   = 0;
    pooled_height_ = 0;
    pooled_width_  = 0;
    open_rows_     = 0;
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
    return overlaps(ub_row_, activate_output_rows(rows_total_, pool_, pool_size_, pool_width_, pool_stride_), first, count);
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
