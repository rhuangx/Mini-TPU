/// The machine: load, in-order issue, exact cycle counts and stall causes, step and step back, and errors at the PC.

#include <stdexcept>
#include <string>
#include <vector>

#include "core/tpu.h"
#include "isa/asm.h"
#include "test_framework.h"

namespace {

// Host row 0 holds 1, 2, 3, 4; one Read_Host_Memory moves that row to UB row 5.
const char* kReadThenHalt =
    ".host 0\n"
    "1 2 3 4\n"
    "Read_Host_Memory host=0 ub=5 rows=1\n"
    "Halt\n";

// Two reads in a row: the second waits for the host interface.
const char* kTwoReads =
    "Read_Host_Memory host=0 ub=0 rows=1\n"
    "Read_Host_Memory host=1 ub=1 rows=1\n"
    "Halt\n";

std::string error_of_run(Tpu& tpu) {
    try {
        tpu.run_to_halt();
    } catch (const std::runtime_error& e) {
        return e.what();
    }
    return "no error";
}

}  // namespace

TEST(tpu_load_places_data_and_resets) {
    Tpu tpu;
    tpu.load(assemble(kReadThenHalt));

    i8 byte = 0;
    tpu.host().read(3, &byte, 1);
    CHECK_EQ(byte, 4);
    CHECK_EQ(tpu.cycle(), Cycle{0});
    CHECK_EQ(tpu.pc(), u32{0});
    CHECK(!tpu.halted());
}

TEST(tpu_halt_waits_for_the_transfer) {
    Tpu tpu;
    tpu.load(assemble(kReadThenHalt));
    tpu.run_to_halt();

    // The read issues on cycle 0 and moves bytes on cycles 0-11; Halt waits 11 cycles, then issues on cycle 12.
    CHECK_EQ(tpu.cycle(), Cycle{13});
    CHECK_EQ(tpu.issued(), u64{2});
    CHECK_EQ(tpu.stats().stalled(Stall::WaitForIdle), u64{11});
    CHECK_EQ(tpu.pc(), u32{1});   // Halt keeps the PC on itself
    CHECK_EQ(tpu.ub().row(5)[3], 4);
}

TEST(tpu_second_transfer_waits_for_the_first) {
    Tpu tpu;
    tpu.load(assemble(kTwoReads));

    tpu.run_cycles(5);
    CHECK(tpu.stall() == Stall::HostInterfaceBusy);

    tpu.run_to_halt();
    CHECK_EQ(tpu.cycle(), Cycle{25});
    CHECK_EQ(tpu.stats().stalled(Stall::HostInterfaceBusy), u64{11});
    CHECK_EQ(tpu.stats().stalled(Stall::WaitForIdle), u64{11});
}

TEST(tpu_round_trip_through_the_ub) {
    const char* source =
        ".host 0\n"
        "-5 6 -7\n"
        "Read_Host_Memory host=0 ub=0 rows=1\n"
        "Write_Host_Memory ub=0 host=7 rows=1\n"
        "Halt\n";
    Tpu tpu;
    tpu.load(assemble(source));
    tpu.run_to_halt();

    std::vector<i8> back(3, 0);
    tpu.host().read(7 * 256, back.data(), back.size());
    CHECK(back == std::vector<i8>({-5, 6, -7}));
}

TEST(tpu_sync_waits_for_idle) {
    Tpu tpu;
    tpu.load(assemble("Read_Host_Memory host=0 ub=0 rows=1\nSync\nNop\nHalt\n"));

    tpu.next_instructions(2);   // the read, then Sync once the read is done
    CHECK_EQ(tpu.cycle(), Cycle{13});
    CHECK(!tpu.host_interface().busy());
}

TEST(tpu_next_instructions_stops_right_after_an_issue) {
    Tpu tpu;
    tpu.load(assemble(kTwoReads));

    tpu.next_instructions(1);
    CHECK_EQ(tpu.issued(), u64{1});
    CHECK_EQ(tpu.cycle(), Cycle{1});

    tpu.next_instructions(1);   // skips the 11 stalled cycles
    CHECK_EQ(tpu.issued(), u64{2});
    CHECK_EQ(tpu.cycle(), Cycle{13});
    CHECK(tpu.stall() == Stall::None);
}

TEST(tpu_prev_instructions_replays_to_the_same_state) {
    Tpu tpu;
    tpu.load(assemble(kTwoReads));
    tpu.run_to_halt();
    CHECK_EQ(tpu.cycle(), Cycle{25});

    tpu.prev_instructions(1);   // undo Halt: back to just after the second read issued
    CHECK_EQ(tpu.issued(), u64{2});
    CHECK_EQ(tpu.cycle(), Cycle{13});
    CHECK_EQ(tpu.pc(), u32{2});
    CHECK(!tpu.halted());
    CHECK_EQ(tpu.host_interface().bytes_done(), u64{22});   // the second transfer's first cycle

    tpu.next_instructions(1);   // forward again lands where it was
    CHECK(tpu.halted());
    CHECK_EQ(tpu.cycle(), Cycle{25});

    tpu.prev_instructions(10);   // more than were issued: back to the start
    CHECK_EQ(tpu.issued(), u64{0});
    CHECK_EQ(tpu.cycle(), Cycle{0});
}

TEST(tpu_back_cycles_replays_to_an_earlier_cycle) {
    Tpu tpu;
    tpu.load(assemble(kTwoReads));
    tpu.run_to_halt();

    tpu.back_cycles(1);   // one cycle before Halt issued
    CHECK_EQ(tpu.cycle(), Cycle{24});
    CHECK(!tpu.halted());
    CHECK(tpu.stall() == Stall::WaitForIdle);

    tpu.back_cycles(20);
    CHECK_EQ(tpu.cycle(), Cycle{4});
    CHECK_EQ(tpu.host_interface().bytes_done(), u64{4 * 22});   // the first transfer, four cycles in

    tpu.run_cycles(21);   // forward again lands where it was
    CHECK_EQ(tpu.cycle(), Cycle{25});
    CHECK(tpu.halted());

    tpu.back_cycles(100);   // more than have run: back to the start
    CHECK_EQ(tpu.cycle(), Cycle{0});
}

TEST(tpu_read_weights_issues_before_the_tile_arrives) {
    Tpu tpu;
    tpu.load(assemble("Read_Weights tile=0\nHalt\n"));

    tpu.next_instructions(1);   // decoupled access: the PC moves on at once
    CHECK_EQ(tpu.cycle(), Cycle{1});
    CHECK_EQ(tpu.pc(), u32{1});
    CHECK(tpu.weight_fifo().fetching());

    // The tile arrives on cycle 1365 and stays in its FIFO slot while the MXU copies one row per cycle through 1620.
    tpu.run_cycles(1400);
    CHECK(tpu.mxu().shifting());
    CHECK_EQ(tpu.mxu().rows_shifted(), u32{36});
    CHECK_EQ(tpu.weight_fifo().tiles().size(), std::size_t{1});

    tpu.run_to_halt();
    CHECK_EQ(tpu.cycle(), Cycle{1622});
    CHECK_EQ(tpu.stats().stalled(Stall::WaitForIdle), u64{1620});
    CHECK(tpu.weight_fifo().tiles().empty());
    CHECK(tpu.mxu().shadow_ready());
    CHECK_EQ(tpu.mxu().shadow_tile(), 0);
}

TEST(tpu_mxu_frees_a_fifo_slot_for_the_fifth_tile) {
    const char* source =
        "Read_Weights tile=0\n"
        "Read_Weights tile=1\n"
        "Read_Weights tile=2\n"
        "Read_Weights tile=3\n"
        "Read_Weights tile=4\n"
        "Halt\n";
    Tpu tpu;
    tpu.load(assemble(source));

    tpu.run_cycles(100);
    CHECK(tpu.stall() == Stall::WeightFifoFull);

    // Tile 0 arrives on cycle 1365 and leaves the FIFO after its last row shifts in on 1620, so the fifth Read_Weights issues on 1621.
    tpu.run_to_halt();
    CHECK_EQ(tpu.stats().stalled(Stall::WeightFifoFull), u64{1617});
    CHECK_EQ(tpu.cycle(), Cycle{5 * 1366 + 1});   // Halt waits for the fifth tile, which arrives on cycle 6829
    CHECK_EQ(tpu.weight_fifo().tiles().size(), std::size_t{4});
}

TEST(tpu_new_weights_with_none_coming_deadlocks) {
    Tpu tpu;
    tpu.load(assemble("MatrixMultiply ub=0 acc=0 rows=1 accumulate=0 new_weights=1\nHalt\n"));
    const std::string expected =
        "pc 0: MatrixMultiply ub=0x0 acc=0x0 rows=1 accumulate=0 new_weights=1: deadlock: weights not ready, but no unit is working";
    CHECK_EQ(error_of_run(tpu), expected);
    CHECK_EQ(tpu.cycle(), Cycle{0});
}

TEST(tpu_errors_name_the_pc_and_leave_the_cycle_alone) {
    Tpu tpu;
    tpu.load(assemble("Read_Host_Memory host=0 ub=0x20000 rows=1\nHalt\n"));   // UB row 131072 is past the last
    const std::string message = error_of_run(tpu);
    const std::string expected_start = "pc 0: Read_Host_Memory host=0x0 ub=0x20000 rows=1: ub bytes";
    CHECK_EQ(message.substr(0, expected_start.size()), expected_start);
    CHECK_EQ(tpu.cycle(), Cycle{0});
    CHECK_EQ(tpu.issued(), u64{0});

    tpu.load(assemble("Nop\n"));
    CHECK_EQ(error_of_run(tpu), std::string("pc 1: past the end of the program; is a Halt missing?"));

    Program unknown;
    InstrBytes bad{};
    bad[0] = 0x03;
    unknown.code.push_back(bad);
    tpu.load(unknown);
    CHECK_EQ(error_of_run(tpu), std::string("pc 0: unknown opcode 0x3"));

    tpu.load(assemble("Read_Weights tile=0x20000\nHalt\n"));
    CHECK_EQ(error_of_run(tpu), std::string("pc 0: Read_Weights tile=0x20000: weight tile 131072 is past the last tile 131071"));
}
