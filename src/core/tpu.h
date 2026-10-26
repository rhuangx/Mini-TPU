/// The TPUv1 machine: its memories and units, and the in-order issue that drives them one cycle per tick().

#pragma once

#include <string>
#include <vector>

#include "common/stats.h"
#include "isa/isa.h"
#include "mem/accumulators.h"
#include "mem/dram.h"
#include "mem/unified_buffer.h"
#include "mxu/systolic_array.h"
#include "units/activation.h"
#include "units/host_interface.h"
#include "units/weight_fifo.h"

class Tpu {
public:
    static constexpr u64 kHostMemBytes = 4 * kGiB;   // pages are allocated only when written

    Tpu();

    // The units hold references to the memories, so a Tpu is never copied.
    Tpu(const Tpu&) = delete;
    Tpu& operator=(const Tpu&) = delete;

    // Resets the machine, then places the program's code and its host and Weight Memory data.
    void load(const Program& program);

    // One cycle: issue the instruction at the PC if it can go, then advance every unit.
    void tick();

    // Forward n cycles, stopping early at Halt.
    void run_cycles(u64 cycles);

    // Back n cycles: reloads the program and replays up to that cycle.
    void back_cycles(u64 cycles);

    // Forward until n more instructions have issued, stopping early at Halt.
    void next_instructions(u64 instructions);

    // Back n instruction issues: reloads the program and replays up to just after the earlier issue.
    void prev_instructions(u64 instructions);

    void run_to_halt();

    bool         halted() const;
    Cycle        cycle() const;
    u32          pc() const;
    u64          issued() const;
    Stall        stall() const;   // why the last cycle issued nothing; Stall::None if something issued
    const Stats& stats() const;

    const Program&        program() const;
    const UnifiedBuffer&  ub() const;
    const Dram&           host() const;
    const Dram&           wmem() const;
    const HostInterface&  host_interface() const;
    const WeightFifo&     weight_fifo() const;
    const Accumulators&   acc() const;
    const SystolicArray&  mxu() const;
    const ActivationUnit& activation() const;

private:
    Program        program_;
    UnifiedBuffer  ub_;
    Accumulators   acc_;
    Dram           host_;
    Dram           wmem_;
    HostInterface  host_interface_;
    WeightFifo     weight_fifo_;
    SystolicArray  mxu_;
    ActivationUnit activation_;

    u32   pc_     = 0;
    bool  halted_ = false;
    Stall stall_  = Stall::None;
    Stats stats_;

    bool  units_idle() const;
    Stall why_blocked(const Instr& in) const;
    void  issue(const Instr& in);
    void  reload();
    [[noreturn]] void fail_at_pc(const std::string& what) const;
};
