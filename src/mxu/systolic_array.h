/// Matrix Multiply Unit: a 256x256 weight-stationary systolic array with an active and a shadow weight plane.

#pragma once

#include <vector>

#include "mem/accumulators.h"
#include "mem/unified_buffer.h"
#include "units/weight_fifo.h"

// One processing element's registers after a cycle.
struct Pe {
    i8  act  = 0;    // the activation it passes to the right
    i32 psum = 0;    // the partial sum it passes down
    int row  = -1;   // which input row this data belongs to; -1 when the PE holds nothing
};

class SystolicArray {
public:
    static constexpr u32 kDim = v1::kMxuDim;

    SystolicArray(const UnifiedBuffer& ub, Accumulators& acc, WeightFifo& fifo);

    // Multiplies `rows` UB rows by the weights into accumulator rows; new_weights first makes the (ready) shadow plane active.
    void start(u32 ub_row, u32 acc_row, u32 rows, bool accumulate, bool new_weights);

    // One cycle: the weight shifter moves one tile row into the shadow plane, and the array takes one step.
    void tick();

    // Empties both planes and the grid, as after a machine reset.
    void reset();

    bool busy() const;           // a MatrixMultiply is still in the array
    bool shifting() const;       // a tile is being shifted into the shadow plane
    bool shadow_ready() const;   // the shadow plane holds a whole tile
    bool reads_ub_rows(u32 first, u32 count) const;
    bool writes_acc_rows(u32 first, u32 count) const;

    // For the terminal: progress, planes and the grid.
    u32 step() const;
    u32 ub_row() const;
    u32 acc_row() const;
    u32 rows() const;
    u32 total_steps() const;
    u32 rows_shifted() const;
    int active_tile() const;     // -1 before any tile was made active
    int shadow_tile() const;     // -1 when the shadow plane is empty
    const Pe& pe(u32 k, u32 n) const;

private:
    const UnifiedBuffer& ub_;
    Accumulators&        acc_;
    WeightFifo&          fifo_;

    // The two weight planes, 256x256 each: plane[k * 256 + n] is the weight PE (k, n) holds.
    std::vector<i8> planes_[2];
    int active_plane_ = 0;
    int active_tile_  = -1;

    // The weight shifter: fills the shadow plane from the oldest FIFO slot, one tile row per cycle.
    int  shadow_tile_  = -1;
    u32  rows_shifted_ = 0;
    bool shadow_full_  = false;

    // The MatrixMultiply in the array.
    std::vector<Pe> grid_;
    std::vector<Pe> next_grid_;
    bool busy_       = false;
    u32  ub_row_     = 0;
    u32  acc_row_    = 0;
    u32  rows_       = 0;
    bool accumulate_ = false;
    u32  step_       = 0;

    void shift_weights();
    void step_array();
    int  shadow_plane() const;
};
