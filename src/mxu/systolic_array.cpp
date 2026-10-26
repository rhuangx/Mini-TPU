/// The weight shifter and one step of the systolic array: activations move right, partial sums move down.

#include "mxu/systolic_array.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace {

// PE (k, n) lives at index k * 256 + n: k is the array row (input feature), n the column (output).
std::size_t at(u32 k, u32 n) {
    return static_cast<std::size_t>(k) * SystolicArray::kDim + n;
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

SystolicArray::SystolicArray(const UnifiedBuffer& ub, Accumulators& acc, WeightFifo& fifo)
    : ub_(ub), acc_(acc), fifo_(fifo) {
    reset();
}

void SystolicArray::reset() {
    const std::size_t pe_count = static_cast<std::size_t>(kDim) * kDim;
    planes_[0].assign(pe_count, 0);
    planes_[1].assign(pe_count, 0);
    active_plane_ = 0;
    active_tile_  = -1;

    shadow_tile_  = -1;
    rows_shifted_ = 0;
    shadow_full_  = false;

    grid_.assign(pe_count, Pe{});
    next_grid_.assign(pe_count, Pe{});
    busy_       = false;
    ub_row_     = 0;
    acc_row_    = 0;
    rows_       = 0;
    accumulate_ = false;
    step_       = 0;
}

void SystolicArray::start(u32 ub_row, u32 acc_row, u32 rows, bool accumulate, bool new_weights) {
    if (busy_) {
        throw std::logic_error("MXU: a MatrixMultiply is already running");
    }
    if (new_weights && !shadow_full_) {
        throw std::logic_error("MXU: the shadow plane has no complete tile to switch to");
    }
    check_rows("ub", ub_row, rows, UnifiedBuffer::kRows);
    check_rows("acc", acc_row, rows, Accumulators::kRows);

    // Switching planes is free: the shadow plane becomes active, and the old active plane becomes the empty shadow.
    if (new_weights) {
        active_plane_ = shadow_plane();
        active_tile_  = shadow_tile_;
        shadow_tile_  = -1;
        shadow_full_  = false;
        rows_shifted_ = 0;
    }

    ub_row_     = ub_row;
    acc_row_    = acc_row;
    rows_       = rows;
    accumulate_ = accumulate;
    step_       = 0;
    busy_       = rows > 0;

    // The last multiply's final steps can leave a PE or two behind in the grid it no longer writes.
    std::fill(grid_.begin(), grid_.end(), Pe{});
    std::fill(next_grid_.begin(), next_grid_.end(), Pe{});
}

void SystolicArray::tick() {
    shift_weights();
    if (busy_) {
        step_array();
    }
}

// The shadow plane copies the oldest ready FIFO tile one 256-byte row per cycle; the tile leaves the FIFO after its last row.
void SystolicArray::shift_weights() {
    if (shadow_full_) {
        return;
    }
    if (!fifo_.front_ready()) {
        return;
    }

    const FifoTile& front = fifo_.tiles().front();
    if (rows_shifted_ == 0) {
        shadow_tile_ = static_cast<int>(front.tile);
    }

    const std::size_t row_start = static_cast<std::size_t>(rows_shifted_) * kDim;
    std::vector<i8>& shadow = planes_[shadow_plane()];
    for (u32 n = 0; n < kDim; ++n) {
        shadow[row_start + n] = front.bytes[row_start + n];
    }
    rows_shifted_ = rows_shifted_ + 1;

    if (rows_shifted_ == kDim) {
        shadow_full_ = true;
        fifo_.pop();
    }
}

// One step: every PE takes the activation from its left and the partial sum from above, adds its product, and passes both on.
// Input row r reaches PE (k, n) at step r + k + n, so after this step PE (k, n) holds row step - k - n: in each array row
// that is a band `rows_` PEs wide, and only the band is visited. Every PE outside it is empty, and stays so.
void SystolicArray::step_array() {
    const std::vector<i8>& weights = planes_[active_plane_];
    const long long step  = step_;
    const long long rows  = rows_;
    const long long right = kDim - 1;

    for (u32 k = 0; k < kDim; ++k) {
        // next_grid_ still holds the state of two steps ago, whose band started up to 2 PEs further left: empty those.
        const long long first = std::max(0LL, step - k - rows + 1);
        const long long last  = std::min(right, step - k);
        for (long long n = std::max(0LL, step - 2 - k - rows + 1); n < std::min(first, right + 1); ++n) {
            next_grid_[at(k, static_cast<u32>(n))] = Pe{};
        }

        for (long long n = first; n <= last; ++n) {
            const u32 col = static_cast<u32>(n);
            Pe incoming;
            if (col == 0) {
                // The left edge: array row k takes input row (step - k), so each row starts k cycles later (the skew).
                const u32 r  = static_cast<u32>(step - k);
                incoming.act = ub_.row(ub_row_ + r)[k];
                incoming.row = static_cast<int>(r);
            } else {
                const Pe& left = grid_[at(k, col - 1)];
                incoming.act = left.act;
                incoming.row = left.row;
            }

            i32 psum_in = 0;
            if (k > 0) {
                psum_in = grid_[at(k - 1, col)].psum;
            }
            const i32 product = static_cast<i32>(incoming.act) * static_cast<i32>(weights[at(k, col)]);

            Pe& next  = next_grid_[at(k, col)];
            next.act  = incoming.act;
            next.row  = incoming.row;
            next.psum = psum_in + product;
        }
    }

    // The bottom row now holds finished column sums: input row r, column n, arrives at step r + 255 + n.
    const long long bottom_first = std::max(0LL, step - right - rows + 1);
    const long long bottom_last  = std::min(right, step - right);
    for (long long col = bottom_first; col <= bottom_last; ++col) {
        const u32 n = static_cast<u32>(col);
        const Pe& bottom = next_grid_[at(kDim - 1, n)];
        if (bottom.row < 0) {
            continue;
        }
        i32& target = acc_.row(acc_row_ + static_cast<u32>(bottom.row))[n];
        if (accumulate_) {
            // Wrapping add, as 32-bit hardware does; plain signed overflow would be undefined in C++.
            const u32 sum = static_cast<u32>(target) + static_cast<u32>(bottom.psum);
            target = static_cast<i32>(sum);
        } else {
            target = bottom.psum;
        }
    }

    grid_.swap(next_grid_);
    step_ = step_ + 1;
    if (step_ == total_steps()) {
        busy_ = false;
    }
}

bool SystolicArray::busy() const {
    return busy_;
}

bool SystolicArray::shifting() const {
    const bool started = rows_shifted_ > 0;
    return started && !shadow_full_;
}

bool SystolicArray::shadow_ready() const {
    return shadow_full_;
}

bool SystolicArray::reads_ub_rows(u32 first, u32 count) const {
    if (!busy_ || count == 0 || rows_ == 0) {
        return false;
    }
    const u64 mine_end  = static_cast<u64>(ub_row_) + rows_;
    const u64 their_end = static_cast<u64>(first) + count;
    return first < mine_end && ub_row_ < their_end;
}

bool SystolicArray::writes_acc_rows(u32 first, u32 count) const {
    if (!busy_ || count == 0 || rows_ == 0) {
        return false;
    }
    const u64 mine_end  = static_cast<u64>(acc_row_) + rows_;
    const u64 their_end = static_cast<u64>(first) + count;
    return first < mine_end && acc_row_ < their_end;
}

u32 SystolicArray::step() const {
    return step_;
}

u32 SystolicArray::ub_row() const {
    return ub_row_;
}

u32 SystolicArray::acc_row() const {
    return acc_row_;
}

u32 SystolicArray::rows() const {
    return rows_;
}

// Row r's last column sum leaves the bottom at step r + 255 + 255, so B rows take B + 511 steps.
u32 SystolicArray::total_steps() const {
    return rows_ + 2 * kDim - 1;
}

u32 SystolicArray::rows_shifted() const {
    return rows_shifted_;
}

int SystolicArray::active_tile() const {
    return active_tile_;
}

int SystolicArray::shadow_tile() const {
    return shadow_tile_;
}

const Pe& SystolicArray::pe(u32 k, u32 n) const {
    return grid_[at(k, n)];
}

int SystolicArray::shadow_plane() const {
    return 1 - active_plane_;
}
