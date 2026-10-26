/// The host interface's transfer state and its 22-bytes-per-cycle copy loop.

#include "units/host_interface.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>

namespace {

// Checked as two steps rather than `addr + n > size`, which a huge n could wrap around.
void check_fits(const std::string& memory, u64 addr, u64 n, u64 size) {
    bool fits = true;
    if (n > size) {
        fits = false;
    } else if (addr > size - n) {
        fits = false;
    }

    if (!fits) {
        const std::string range = "[" + std::to_string(addr) + ", +" + std::to_string(n) + ")";
        throw std::out_of_range(memory + " bytes " + range + " run past its " + std::to_string(size) + " bytes");
    }
}

}  // namespace

HostInterface::HostInterface(Dram& host, UnifiedBuffer& ub) : host_(host), ub_(ub) {}

void HostInterface::start(Direction direction, u32 host_row, u32 ub_row, u32 rows) {
    if (busy()) {
        throw std::logic_error("host interface: a transfer is already running");
    }

    const u64 bytes     = static_cast<u64>(rows) * UnifiedBuffer::kRowBytes;
    const u64 host_addr = static_cast<u64>(host_row) * UnifiedBuffer::kRowBytes;
    const u64 ub_addr   = static_cast<u64>(ub_row) * UnifiedBuffer::kRowBytes;

    // Both ranges are checked before any byte moves, so a bad transfer changes nothing.
    check_fits("host", host_addr, bytes, host_.size());
    check_fits("ub", ub_addr, bytes, UnifiedBuffer::kBytes);

    direction_   = direction;
    host_addr_   = host_addr;
    ub_addr_     = ub_addr;
    bytes_total_ = bytes;
    bytes_done_  = 0;
}

void HostInterface::tick() {
    if (!busy()) {
        return;
    }

    const u64 bytes_left = bytes_total_ - bytes_done_;
    const u64 piece      = std::min<u64>(bytes_left, v1::kHostBytesPerCycle);

    const u64 host_addr = host_addr_ + bytes_done_;
    const u64 ub_addr   = ub_addr_ + bytes_done_;

    std::array<i8, v1::kHostBytesPerCycle> chunk{};
    if (direction_ == Direction::HostToUb) {
        host_.read(host_addr, chunk.data(), piece);
        ub_.write(ub_addr, chunk.data(), piece);
    } else {
        ub_.read(ub_addr, chunk.data(), piece);
        host_.write(host_addr, chunk.data(), piece);
    }

    bytes_done_ = bytes_done_ + piece;
}

void HostInterface::reset() {
    direction_   = Direction::HostToUb;
    host_addr_   = 0;
    ub_addr_     = 0;
    bytes_total_ = 0;
    bytes_done_  = 0;
}

bool HostInterface::busy() const {
    return bytes_done_ < bytes_total_;
}

bool HostInterface::writes_ub_rows(u32 first, u32 count) const {
    if (!busy() || direction_ != Direction::HostToUb || count == 0) {
        return false;
    }
    const u64 mine_first = ub_addr_ / UnifiedBuffer::kRowBytes;
    const u64 mine_end   = (ub_addr_ + bytes_total_) / UnifiedBuffer::kRowBytes;
    const u64 their_end  = static_cast<u64>(first) + count;
    return first < mine_end && mine_first < their_end;
}

bool HostInterface::reads_ub_rows(u32 first, u32 count) const {
    if (!busy() || direction_ != Direction::UbToHost || count == 0) {
        return false;
    }
    const u64 mine_first = ub_addr_ / UnifiedBuffer::kRowBytes;
    const u64 mine_end   = (ub_addr_ + bytes_total_) / UnifiedBuffer::kRowBytes;
    const u64 their_end  = static_cast<u64>(first) + count;
    return first < mine_end && mine_first < their_end;
}

u64 HostInterface::bytes_done() const {
    return bytes_done_;
}

u64 HostInterface::bytes_total() const {
    return bytes_total_;
}
