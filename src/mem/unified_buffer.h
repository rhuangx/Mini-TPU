/// Unified Buffer: 24 MiB of int8 activations in 256-byte rows, one row per MXU input vector.

#pragma once

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/config.h"

class UnifiedBuffer {
public:
    static constexpr u32 kRows     = v1::kUbRows;
    static constexpr u32 kRowBytes = v1::kMxuDim;
    static constexpr u64 kBytes    = static_cast<u64>(kRows) * kRowBytes;

    UnifiedBuffer() {
        bytes_.resize(kBytes, 0);
    }

    // Instructions address the buffer by row; the pointer covers that row's 256 bytes. Only writers ask for it.
    i8* row(u32 r) {
        check_row(r);
        return bytes_.data() + start_of_row(r);
    }

    const i8* row(u32 r) const {
        check_row(r);
        return bytes_.data() + start_of_row(r);
    }

    // The host interface moves byte ranges; addr is a byte address, rows laid end to end.
    void write(u64 addr, const i8* src, u64 n) {
        check_range(addr, n);
        std::memcpy(bytes_.data() + addr, src, n);
    }

    void read(u64 addr, i8* dst, u64 n) const {
        check_range(addr, n);
        std::memcpy(dst, bytes_.data() + addr, n);
    }

private:
    std::vector<i8> bytes_;

    static u64 start_of_row(u32 r) {
        return static_cast<u64>(r) * kRowBytes;
    }

    static void check_row(u32 r) {
        if (r >= kRows) {
            const std::string last = std::to_string(kRows - 1);
            throw std::out_of_range("ub row " + std::to_string(r) + " is past the last row " + last);
        }
    }

    // Checked as two steps rather than `addr + n > kBytes`, which a huge n could wrap around.
    static void check_range(u64 addr, u64 n) {
        if (n > kBytes) {
            throw_range_error(addr, n);
        }
        if (addr > kBytes - n) {
            throw_range_error(addr, n);
        }
    }

    [[noreturn]] static void throw_range_error(u64 addr, u64 n) {
        const std::string range = "[" + std::to_string(addr) + ", +" + std::to_string(n) + ")";
        throw std::out_of_range("ub bytes " + range + " run past 24 MiB");
    }
};
