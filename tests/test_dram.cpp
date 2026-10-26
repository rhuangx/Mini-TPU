/// DRAM: 8 GiB addressable without allocating it, page-straddling copies, zero reads, bounds.

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "mem/dram.h"
#include "test_framework.h"

namespace {

// The message a read throws, or "" if it does not throw.
std::string read_error(const Dram& dram, u64 addr, u64 n) {
    std::vector<i8> buffer(n, 0);
    try {
        dram.read(addr, buffer.data(), n);
    } catch (const std::out_of_range& e) {
        return e.what();
    }
    return "";
}

}  // namespace

TEST(dram_starts_empty_and_reads_zero) {
    const Dram wmem("wmem", v1::kWeightMemBytes);
    std::vector<i8> buffer(100, 7);

    wmem.read(v1::kWeightMemBytes - 100, buffer.data(), buffer.size());
    CHECK(buffer == std::vector<i8>(100, 0));
    CHECK_EQ(wmem.pages_allocated(), u64{0});
}

TEST(dram_allocates_only_the_pages_written) {
    Dram wmem("wmem", v1::kWeightMemBytes);

    const i8 value = 42;
    const u64 last_byte = v1::kWeightMemBytes - 1;
    wmem.write(last_byte, &value, 1);
    CHECK_EQ(wmem.pages_allocated(), u64{1});

    // A tile written at a tile boundary fills exactly one page.
    const std::vector<i8> tile(v1::kTileBytes, 3);
    wmem.write(5 * v1::kTileBytes, tile.data(), tile.size());
    CHECK_EQ(wmem.pages_allocated(), u64{2});

    i8 back = 0;
    wmem.read(last_byte, &back, 1);
    CHECK_EQ(back, 42);
}

TEST(dram_copies_straddle_pages) {
    Dram host("host", 1 * kMiB);
    const u64 page_edge = Dram::kPageBytes;

    // Three bytes land at the end of page 0 and three at the start of page 1.
    const std::vector<i8> src = {1, 2, 3, 4, 5, 6};
    host.write(page_edge - 3, src.data(), src.size());
    CHECK_EQ(host.pages_allocated(), u64{2});

    std::vector<i8> back(10, -1);
    host.read(page_edge - 5, back.data(), back.size());
    const std::vector<i8> expected = {0, 0, 1, 2, 3, 4, 5, 6, 0, 0};
    CHECK(back == expected);
}

TEST(dram_out_of_range_throws_with_its_name) {
    Dram host("host", 1024);
    i8 byte = 0;
    const u64 huge = std::numeric_limits<u64>::max();

    CHECK_THROWS(host.write(1024, &byte, 1));
    CHECK_THROWS(host.read(1023, &byte, 2));
    CHECK_THROWS(host.read(1, &byte, huge));

    host.write(1023, &byte, 1);   // the last byte is fine
    host.read(0, &byte, 0);       // an empty range is fine

    const std::string message = read_error(host, 2000, 1);
    CHECK_EQ(message, std::string("host bytes [2000, +1) run past its 1024 bytes"));
}
