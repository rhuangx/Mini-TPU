/// The ResNet compiler: the whole network as one program, with every activation map kept in the Unified Buffer.

#pragma once

#include <vector>

#include "isa/isa.h"
#include "mem/dram.h"
#include "mem/unified_buffer.h"
#include "ref/ref_resnet.h"

// Where one activation map sits in the Unified Buffer, one UB row per pixel and 256 channels per block:
// - Frame: the map inside a one-pixel zero border, image rows `pitch` UB rows apart. A 3x3 tap is then the frame
//   shifted by dy * pitch + dx, so a stride-1 convolution is one multiply per tap.
// - Phases: the bordered map split into its even and odd rows and columns, four phases of `phase_rows` UB rows each,
//   for a stride-2 consumer: each of its taps then reads one phase, shifted, as a single run of rows.
// - Vector: one row per block, the globally pooled map.
struct MapPlacement {
    enum class Layout { Frame, Phases, Vector };

    Layout           layout     = Layout::Frame;
    u32              side       = 0;   // height and width in pixels
    u32              channels   = 0;
    u32              pitch      = 0;   // UB rows from one image row to the next (frames and phases)
    u32              phase_rows = 0;   // UB rows per phase (phases only)
    std::vector<u32> blocks;           // first UB row of each 256-channel block
};

struct CompiledResNet {
    Program                   program;
    u32                       result_host_row = 0;   // the logits: one host row per 256 classes
    u32                       classes         = 0;
    MapPlacement                           stem;
    std::vector<std::vector<MapPlacement>> block_convs;   // each block's maps after every conv but the last
    std::vector<MapPlacement>              block_out;     // every block's output but the last, pooled straight into `pooled`
    MapPlacement              pooled;
    u32                       tiles = 0;   // weight tiles fetched from DDR3
};

// The image goes to host memory im2col'd for the stem, with an extra zero row and column that become the max pool's
// padding, and so do the bias masks. Everything after the stem stays on the chip until the logits are written back.
CompiledResNet compile_resnet(const ResNet& net, const Matrix8& image);

// A map read back out of the Unified Buffer, one pixel per row, to compare with the reference.
Matrix8 read_map(const UnifiedBuffer& ub, const MapPlacement& map);

// The logits, read back out of host memory.
Matrix8 read_logits(const Dram& host, const CompiledResNet& compiled);
