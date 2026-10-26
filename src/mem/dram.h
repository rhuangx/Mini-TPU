/// Off-chip DRAM (host memory, Weight Memory): byte-addressed, with 64 KiB pages allocated on first write.

#pragma once

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/config.h"

class Dram {
public:
    static constexpr u64 kPageBytes = v1::kTileBytes;   // a weight tile fills exactly one page

    Dram(std::string name, u64 bytes) : name_(std::move(name)), size_(bytes) {}

    u64 size() const {
        return size_;
    }

    u64 pages_allocated() const {
        return pages_.size();
    }

    // A copy can cross page boundaries, so it is done one page-sized piece at a time.
    void write(u64 addr, const i8* src, u64 n) {
        check_range(addr, n);

        while (n > 0) {
            const u64 page_number    = addr / kPageBytes;
            const u64 offset_in_page = addr % kPageBytes;
            const u64 room_in_page   = kPageBytes - offset_in_page;
            const u64 piece          = std::min(n, room_in_page);

            std::vector<i8>& page = page_for_writing(page_number);
            std::memcpy(page.data() + offset_in_page, src, piece);

            addr += piece;
            src  += piece;
            n    -= piece;
        }
    }

    // Bytes never written read as zero, and reading never allocates a page.
    void read(u64 addr, i8* dst, u64 n) const {
        check_range(addr, n);

        while (n > 0) {
            const u64 page_number    = addr / kPageBytes;
            const u64 offset_in_page = addr % kPageBytes;
            const u64 room_in_page   = kPageBytes - offset_in_page;
            const u64 piece          = std::min(n, room_in_page);

            const auto found = pages_.find(page_number);
            if (found == pages_.end()) {
                std::memset(dst, 0, piece);
            } else {
                const std::vector<i8>& page = found->second;
                std::memcpy(dst, page.data() + offset_in_page, piece);
            }

            addr += piece;
            dst  += piece;
            n    -= piece;
        }
    }

private:
    std::string name_;   // "host" or "wmem", for error messages
    u64         size_;
    std::unordered_map<u64, std::vector<i8>> pages_;   // page number -> that page's bytes

    std::vector<i8>& page_for_writing(u64 page_number) {
        std::vector<i8>& page = pages_[page_number];
        if (page.empty()) {
            page.resize(kPageBytes, 0);
        }
        return page;
    }

    // Checked as two steps rather than `addr + n > size_`, which a huge n could wrap around.
    void check_range(u64 addr, u64 n) const {
        if (n > size_) {
            throw_range_error(addr, n);
        }
        if (addr > size_ - n) {
            throw_range_error(addr, n);
        }
    }

    [[noreturn]] void throw_range_error(u64 addr, u64 n) const {
        const std::string range = "[" + std::to_string(addr) + ", +" + std::to_string(n) + ")";
        throw std::out_of_range(name_ + " bytes " + range + " run past its " + std::to_string(size_) + " bytes");
    }
};
