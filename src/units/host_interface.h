/// Host interface: the PCIe link that moves rows between host memory and the Unified Buffer at 22 bytes per cycle.

#pragma once

#include "mem/dram.h"
#include "mem/unified_buffer.h"

enum class Direction {
    HostToUb,   // Read_Host_Memory
    UbToHost,   // Write_Host_Memory
};

class HostInterface {
public:
    HostInterface(Dram& host, UnifiedBuffer& ub);

    // Starts moving `rows` 256-byte rows; throws if either range runs past its memory or a transfer is running.
    void start(Direction direction, u32 host_row, u32 ub_row, u32 rows);

    // One cycle: moves the next 22 bytes (or fewer, at the end) of the running transfer.
    void tick();

    // Drops any running transfer, as after a machine reset.
    void reset();

    bool busy() const;

    // True while a Read_Host_Memory is still writing any of these UB rows.
    bool writes_ub_rows(u32 first, u32 count) const;

    // True while a Write_Host_Memory is still reading any of these UB rows.
    bool reads_ub_rows(u32 first, u32 count) const;

    u64  bytes_done() const;
    u64  bytes_total() const;

private:
    Dram&          host_;
    UnifiedBuffer& ub_;

    Direction direction_   = Direction::HostToUb;
    u64       host_addr_   = 0;   // byte address where the transfer starts
    u64       ub_addr_     = 0;
    u64       bytes_total_ = 0;
    u64       bytes_done_  = 0;
};
