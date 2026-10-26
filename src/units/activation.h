/// Activation unit: turns int32 accumulator rows into int8 Unified Buffer rows, one 256-value row per cycle, with optional pooling.

#pragma once

#include <vector>

#include "isa/isa.h"
#include "mem/accumulators.h"
#include "mem/unified_buffer.h"

// One value: x = value / 2^shift; identity and relu give round(x), sigmoid and tanh give round(f(x) * 128); saturated to int8.
i8 activate(i32 value, u32 shift, ActivationFunction function);

// The average of `count` int8 values from their sum, rounded to the nearest with halves away from zero.
i8 average(i32 sum, u32 count);

// How many UB rows an Activate writes: `rows`, or rows / (size x size) when it pools.
u32 activate_output_rows(u32 rows, Pooling pool, u32 pool_size);

class ActivationUnit {
public:
    ActivationUnit(const Accumulators& acc, UnifiedBuffer& ub);

    // Starts activating `rows` accumulator rows into UB rows; throws on a bad range, a bad pooling shape, or a running Activate.
    void start(u32 acc_row, u32 ub_row, u32 rows, u32 shift, ActivationFunction function, Pooling pool, u32 pool_size,
               u32 pool_width);

    // One cycle: the next accumulator row goes through the function, straight into its UB row or into the pooling line buffer.
    void tick();

    // Drops any running Activate, as after a machine reset.
    void reset();

    bool busy() const;

    // True while the running Activate covers any of these rows.
    bool reads_acc_rows(u32 first, u32 count) const;
    bool writes_ub_rows(u32 first, u32 count) const;

    // For the terminal: what is running and how far it has got.
    u32 acc_row() const;
    u32 ub_row() const;
    u32 rows_done() const;
    u32 rows_total() const;
    u32 shift() const;
    ActivationFunction function() const;
    Pooling pool() const;
    u32 pool_size() const;
    u32 pool_width() const;

private:
    const Accumulators& acc_;
    UnifiedBuffer&      ub_;

    u32 acc_row_    = 0;
    u32 ub_row_     = 0;
    u32 rows_total_ = 0;
    u32 rows_done_  = 0;
    u32 shift_      = 0;
    ActivationFunction function_ = ActivationFunction::Identity;

    // Pooling: the feature map is pool_width_ pixels wide, one accumulator row per pixel, pooled pool_size_ x pool_size_.
    Pooling pool_       = Pooling::None;
    u32     pool_size_  = 0;
    u32     pool_width_ = 0;

    // The line buffer: one band of pooled pixels, 256 running values each (max so far, or sum so far).
    std::vector<i32> line_;

    void pool_row(const std::vector<i8>& activated);
};
