/// ResNet lowering: frames and phases in the UB, convolutions as one accumulating multiply per tap, residuals and biases as more.

#include "compiler/resnet.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

#include "common/config.h"

namespace {

constexpr u32 kDim           = v1::kMxuDim;
constexpr u32 kPrefetchTiles = 4;                   // Read_Weights run this many tiles ahead of the multiply using them
constexpr u32 kAccHalf       = v1::kAccRows / 2;    // a map this small alternates between the accumulator halves
constexpr u32 kStemTaps      = 7 * 7 * 3;           // the stem's im2col row: every tap of its 7x7 window, 3 channels each

using Layout = MapPlacement::Layout;
using Tile   = std::vector<i8>;

u32 block_count(u32 channels) {
    return (channels + kDim - 1) / kDim;
}

// The channels in block kb of a map with `channels` channels.
u32 block_width(u32 channels, u32 kb) {
    return std::min(kDim, channels - kb * kDim);
}

// Rows [first_row + 256 kb, +256) of a weight matrix, capped at row_count rows past first_row, and columns
// [256 nb, +256), zero-padded: one Weight Memory tile. A conv tap's rows start at (dy * kernel + dx) * in_channels.
Tile weight_tile(const Matrix8& w, u32 first_row, u32 row_count, u32 kb, u32 nb) {
    Tile tile(v1::kTileBytes, 0);
    for (u32 k = 0; k < kDim && kb * kDim + k < row_count; ++k) {
        for (u32 n = 0; n < kDim && nb * kDim + n < w.cols; ++n) {
            tile[static_cast<std::size_t>(k) * kDim + n] = w.at(first_row + kb * kDim + k, nb * kDim + n);
        }
    }
    return tile;
}

// `scale` on the diagonal of the first `channels` rows: multiplying by it adds scale * x into the sums.
Tile identity_tile(i32 scale, u32 channels) {
    Tile tile(v1::kTileBytes, 0);
    for (u32 k = 0; k < channels; ++k) {
        tile[static_cast<std::size_t>(k) * kDim + k] = static_cast<i8>(scale);
    }
    return tile;
}

// The UB row a bias multiplies: 1 in column `first`, 127 in every column after it.
std::vector<i8> constant_row(u32 first) {
    std::vector<i8> row(kDim, 0);
    row[first] = 1;
    std::fill(row.begin() + first + 1, row.end(), static_cast<i8>(127));
    return row;
}

// Rows [first, 256) of a tile that, multiplied by constant_row(first), add bias[256 nb + n] into column n: the bias is
// r + 127 q with |r| <= 63, r goes in row `first`, and q is spread over the rows after it, 127 at most each.
void put_bias(Tile& tile, u32 first, const std::vector<i32>& bias, u32 nb) {
    for (u32 n = 0; n < kDim && nb * kDim + n < bias.size(); ++n) {
        const long long value = bias[nb * kDim + n];
        long long q = (value >= 0 ? value + 63 : value - 63) / 127;
        tile[static_cast<std::size_t>(first) * kDim + n] = static_cast<i8>(value - 127 * q);
        for (u32 k = first + 1; k < kDim && q != 0; ++k) {
            const long long step = std::clamp(q, -127LL, 127LL);
            tile[static_cast<std::size_t>(k) * kDim + n] = static_cast<i8>(step);
            q = q - step;
        }
        if (q != 0) {
            throw std::invalid_argument("a bias of " + std::to_string(value) + " is more than " +
                                        std::to_string(kDim - first - 1) + " rows of 127 x 127 can add");
        }
    }
}

// The stem's input: one row per pixel of the stem conv's output, holding that pixel's 7x7 window of the image (stride 2,
// padding 3), column (dy * 7 + dx) * 3 + c, then the constant row's 1 and 127s for the bias. Row 0 and column 0 of this
// grid stay zero: after relu every value is at least zero, so they act as the max pool's padding of one pixel at the
// top and left.
std::vector<i8> stem_columns(const Matrix8& image, u32 size) {
    const u32 extended = size / 2 + 1;
    const std::vector<i8> constants = constant_row(kStemTaps);
    std::vector<i8> bytes(static_cast<std::size_t>(extended) * extended * kDim, 0);
    for (u32 yy = 1; yy < extended; ++yy) {
        for (u32 xx = 1; xx < extended; ++xx) {
            i8* row = bytes.data() + (static_cast<std::size_t>(yy) * extended + xx) * kDim;
            std::copy(constants.begin() + kStemTaps, constants.end(), row + kStemTaps);
            for (u32 dy = 0; dy < 7; ++dy) {
                for (u32 dx = 0; dx < 7; ++dx) {
                    const long long iy = 2LL * (yy - 1) + dy - 3;
                    const long long ix = 2LL * (xx - 1) + dx - 3;
                    if (iy < 0 || iy >= size || ix < 0 || ix >= size) {
                        continue;
                    }
                    for (u32 c = 0; c < 3; ++c) {
                        row[(dy * 7 + dx) * 3 + c] = image.at(static_cast<u32>(iy * size + ix), c);
                    }
                }
            }
        }
    }
    return bytes;
}

u32 frame_row(const MapPlacement& map, u32 kb, u32 y, u32 x) {
    return map.blocks[kb] + (y + 1) * map.pitch + (x + 1);
}

u32 phase_base(const MapPlacement& map, u32 kb, u32 row_parity, u32 col_parity) {
    return map.blocks[kb] + (2 * row_parity + col_parity) * map.phase_rows;
}

// Pixel (y, x) sits at (y + 1, x + 1) of the bordered map, which is in the phase of that row's and column's parity.
u32 phase_row(const MapPlacement& map, u32 kb, u32 y, u32 x) {
    return phase_base(map, kb, (y + 1) % 2, (x + 1) % 2) + ((y + 1) / 2) * map.pitch + (x + 1) / 2;
}

u32 pixel_row(const MapPlacement& map, u32 kb, u32 y, u32 x) {
    if (map.layout == Layout::Frame) {
        return frame_row(map, kb, y, x);
    }
    if (map.layout == Layout::Phases) {
        return phase_row(map, kb, y, x);
    }
    return map.blocks[kb];
}

// The UB rows one 256-channel block of the map takes.
u32 map_rows(const MapPlacement& map) {
    if (map.layout == Layout::Frame) {
        return (map.side + 2) * map.pitch;
    }
    if (map.layout == Layout::Phases) {
        return 4 * map.phase_rows;
    }
    return 1;
}

// The first UB row tap (dy, dx) reads, for a conv whose output pixel (y, x) is accumulator row y * pitch + x.
u32 tap_row(const MapPlacement& in, u32 kb, u32 dy, u32 dx) {
    if (in.layout == Layout::Frame) {
        return in.blocks[kb] + dy * in.pitch + dx;
    }
    // Stride 2: bordered pixel (2y + dy, 2x + dx) is pixel (y + dy / 2, x + dx / 2) of phase (dy % 2, dx % 2).
    return phase_base(in, kb, dy % 2, dx % 2) + (dy / 2) * in.pitch + dx / 2;
}

// Builds the code, the Weight Memory tiles, the host data and the UB layout as the network is walked.
class Builder {
public:
    Program program;

    u32 alloc_ub(u32 rows) {
        const u32 first = next_ub_;
        if (rows > v1::kUbRows - next_ub_) {
            throw std::invalid_argument("the network's maps need more than the Unified Buffer's " +
                                        std::to_string(v1::kUbRows) + " rows");
        }
        next_ub_ = next_ub_ + rows;
        return first;
    }

    // Host rows for this data, or for a result when `bytes` is empty and `rows` says how many.
    u32 alloc_host(std::vector<i8> bytes, u32 rows = 0) {
        const u32 first = next_host_;
        if (!bytes.empty()) {
            rows = static_cast<u32>(bytes.size() / kDim);
            program.host.push_back({static_cast<u64>(first) * kDim, std::move(bytes)});
        }
        next_host_ = next_host_ + rows;
        return first;
    }

    MapPlacement frame_map(u32 side, u32 channels) {
        MapPlacement map;
        map.layout   = Layout::Frame;
        map.side     = side;
        map.channels = channels;
        map.pitch    = side + 2;
        for (u32 kb = 0; kb < block_count(channels); ++kb) {
            map.blocks.push_back(alloc_ub(map_rows(map)));
        }
        return map;
    }

    // Phases of a side x side map, laid out for its stride-2 consumer: the consumer's pitch, side / 2 + 2.
    MapPlacement phase_map(u32 side, u32 channels) {
        MapPlacement map;
        map.layout     = Layout::Phases;
        map.side       = side;
        map.channels   = channels;
        map.pitch      = side / 2 + 2;
        map.phase_rows = (side / 2 + 1) * map.pitch;
        for (u32 kb = 0; kb < block_count(channels); ++kb) {
            map.blocks.push_back(alloc_ub(map_rows(map)));
        }
        return map;
    }

    MapPlacement vector_map(u32 channels) {
        MapPlacement map;
        map.layout   = Layout::Vector;
        map.side     = 1;
        map.channels = channels;
        for (u32 kb = 0; kb < block_count(channels); ++kb) {
            map.blocks.push_back(alloc_ub(1));
        }
        return map;
    }

    MapPlacement new_map(Layout layout, u32 side, u32 channels) {
        if (layout == Layout::Frame) {
            return frame_map(side, channels);
        }
        if (layout == Layout::Phases) {
            return phase_map(side, channels);
        }
        return vector_map(channels);
    }

    // A map of this layout and side that holds constant_row(0) at every pixel and zeros elsewhere, sent from the host
    // the first time a bias needs it: multiplied by a bias tile, it adds the bias at every pixel and nowhere else.
    MapPlacement mask(Layout layout, u32 side) {
        for (const MapPlacement& map : masks_) {
            if (map.layout == layout && map.side == side) {
                return map;
            }
        }
        const MapPlacement map = new_map(layout, side, 1);
        const std::vector<i8> constants = constant_row(0);
        std::vector<i8> bytes(static_cast<std::size_t>(map_rows(map)) * kDim, 0);
        for (u32 y = 0; y < side; ++y) {
            for (u32 x = 0; x < side; ++x) {
                const std::size_t row = pixel_row(map, 0, y, x) - map.blocks[0];
                std::copy(constants.begin(), constants.end(), bytes.begin() + static_cast<std::ptrdiff_t>(row * kDim));
            }
        }
        host_transfer(Op::ReadHostMemory, alloc_host(std::move(bytes)), map.blocks[0], map_rows(map));
        masks_.push_back(map);
        return map;
    }

    // Accumulator rows for one output block. Small ones alternate halves, so the next block's multiplies can fill one
    // half while Activate drains the other; the interlocks keep any overlap correct.
    u32 acc_region(u32 rows) {
        if (rows <= kAccHalf) {
            const u32 first = half_ * kAccHalf;
            half_ = 1 - half_;
            return first;
        }
        if (rows <= v1::kAccRows) {
            return 0;
        }
        throw std::invalid_argument("a map needs " + std::to_string(rows) + " accumulator rows; there are " +
                                    std::to_string(v1::kAccRows));
    }

    u32 add_tile(Tile tile) {
        const u32 index = static_cast<u32>(program.weights.size());
        program.weights.push_back({static_cast<u64>(index) * v1::kTileBytes, std::move(tile)});
        return index;
    }

    // A multiply switches to the shadow plane only when it needs a different tile than the array already holds.
    void multiply(u32 ub_row, u32 acc_row, u32 rows, bool accumulate, u32 tile) {
        Instr in;
        in.op         = Op::MatrixMultiply;
        in.ub_row     = ub_row;
        in.acc_row    = acc_row;
        in.rows       = rows;
        in.accumulate = accumulate ? 1 : 0;
        if (static_cast<long long>(tile) != active_tile_) {
            in.new_weights = 1;
            fetches_.push_back(tile);
            active_tile_ = tile;
        }
        code_.push_back(in);
    }

    // Adds output block nb of `bias` into the accumulators by multiplying `rows` rows of a mask from `ub_row`.
    void bias(const std::vector<i32>& bias, u32 nb, u32 ub_row, u32 acc_row, u32 rows) {
        if (bias.empty()) {
            return;
        }
        Tile tile(v1::kTileBytes, 0);
        put_bias(tile, 0, bias, nb);
        multiply(ub_row, acc_row, rows, true, add_tile(std::move(tile)));
    }

    void activate(u32 acc_row, u32 ub_row, u32 rows, u32 shift, ActivationFunction function) {
        Instr in;
        in.op       = Op::Activate;
        in.acc_row  = acc_row;
        in.ub_row   = ub_row;
        in.rows     = rows;
        in.shift    = shift;
        in.function = function;
        code_.push_back(in);
    }

    void activate_pooled(u32 acc_row, u32 ub_row, u32 rows, u32 shift, Pooling pool, u32 size, u32 width, u32 stride) {
        activate(acc_row, ub_row, rows, shift, ActivationFunction::Relu);
        code_.back().pool        = pool;
        code_.back().pool_size   = size;
        code_.back().pool_width  = width;
        code_.back().pool_stride = stride;
    }

    // One transfer moves at most 65,535 rows.
    void host_transfer(Op op, u32 host_row, u32 ub_row, u32 rows) {
        while (rows > 0) {
            const u32 piece = std::min<u32>(rows, 0xFFFF);
            Instr in;
            in.op       = op;
            in.host_row = host_row;
            in.ub_row   = ub_row;
            in.rows     = piece;
            code_.push_back(in);
            host_row = host_row + piece;
            ub_row   = ub_row + piece;
            rows     = rows - piece;
        }
    }

    // The code with its Read_Weights: the first 4 tiles at the start, then each new tile right after the multiply that
    // frees a FIFO slot for it.
    void finish() {
        Instr halt;
        halt.op = Op::Halt;
        code_.push_back(halt);

        for (std::size_t t = 0; t < fetches_.size() && t < kPrefetchTiles; ++t) {
            program.code.push_back(encode(read_weights(fetches_[t])));
        }
        std::size_t used = 0;
        for (const Instr& in : code_) {
            program.code.push_back(encode(in));
            const bool takes_tile = in.op == Op::MatrixMultiply && in.new_weights == 1;
            if (!takes_tile) {
                continue;
            }
            const std::size_t ahead = used + kPrefetchTiles;
            if (ahead < fetches_.size()) {
                program.code.push_back(encode(read_weights(fetches_[ahead])));
            }
            used = used + 1;
        }
    }

    u32 tiles_fetched() const {
        return static_cast<u32>(fetches_.size());
    }

private:
    std::vector<Instr>        code_;
    std::vector<u32>          fetches_;   // the tile each new-weights multiply takes, in order
    std::vector<MapPlacement> masks_;
    long long                 active_tile_ = -1;
    u32                       next_ub_     = 0;
    u32                       next_host_   = 0;
    u32                       half_        = 0;

    static Instr read_weights(u32 tile) {
        Instr in;
        in.op   = Op::ReadWeights;
        in.tile = tile;
        return in;
    }
};

// The largest stride in the block: its input is laid out as phases when this is 2.
u32 block_stride(const Block& block) {
    u32 stride = block.downsample ? block.down.stride : 1;
    for (const ConvLayer& conv : block.convs) {
        stride = std::max(stride, conv.stride);
    }
    return stride;
}

Layout layout_for(u32 consumer_stride) {
    return consumer_stride == 2 ? Layout::Phases : Layout::Frame;
}

void check_shapes(const ResNet& net, const Matrix8& image) {
    const ResNetConfig& config = net.config;
    if (image.rows != config.size * config.size || image.cols != 3) {
        throw std::invalid_argument("the image must be " + std::to_string(config.size) + " x " +
                                    std::to_string(config.size) + " pixels of 3 channels, one pixel per row");
    }
    if (config.size / 2 + 1 > 255) {
        throw std::invalid_argument("the stem's max pool reads a map " + std::to_string(config.size / 2 + 1) +
                                    " pixels wide; Activate's width goes up to 255");
    }
    const u32 last_side = stage_side(config, static_cast<u32>(config.channels.size() - 1));
    if (last_side > 7) {
        throw std::invalid_argument("the final map is " + std::to_string(last_side) + " x " + std::to_string(last_side) +
                                    "; the global average pool's window goes up to 7 x 7");
    }
    if (block_stride(net.blocks.front()) != 1) {
        throw std::invalid_argument("the first block must keep the stem's resolution");
    }
}

// The stem: im2col'd input from the host in bands, one multiply per band, and one Activate per pooled row that applies
// relu and the 3x3 stride-2 max pool and writes the row into the first frame.
void compile_stem(Builder& b, const ResNet& net, const MapPlacement& out, u32 input_host, u32 input_ub) {
    const u32 extended = net.config.size / 2 + 1;   // the stem conv's output side, plus the padding row and column
    const u32 pooled   = out.side;

    // Each band of pooled rows needs 2 * band + 1 conv rows in one accumulator half.
    const u32 band = std::min(pooled, (kAccHalf / extended - 1) / 2);
    if (band == 0) {
        throw std::invalid_argument("the stem's rows are too wide for the accumulators");
    }

    std::vector<u32> tiles;
    for (u32 nb = 0; nb < block_count(out.channels); ++nb) {
        Tile tile = weight_tile(net.stem.weights, 0, kStemTaps, 0, nb);
        put_bias(tile, kStemTaps, net.stem.bias, nb);
        tiles.push_back(b.add_tile(std::move(tile)));
    }

    // Each band's rows arrive while the band before it computes; the transfers run one after another.
    u32 transferred = 0;
    const auto read_through = [&](u32 first_pooled_row) {
        const u32 last_pooled = std::min(pooled, first_pooled_row + band);
        const u32 end         = (2 * last_pooled + 1) * extended;
        if (end > transferred) {
            b.host_transfer(Op::ReadHostMemory, input_host + transferred, input_ub + transferred, end - transferred);
            transferred = end;
        }
    };

    read_through(0);
    for (u32 first = 0; first < pooled; first += band) {
        const u32 last = std::min(pooled, first + band);
        if (last < pooled) {
            read_through(last);
        }
        const u32 conv_rows = (2 * (last - first) + 1) * extended;
        for (u32 nb = 0; nb < block_count(out.channels); ++nb) {
            const u32 acc = b.acc_region(conv_rows);
            b.multiply(input_ub + 2 * first * extended, acc, conv_rows, false, tiles[nb]);
            for (u32 oy = first; oy < last; ++oy) {
                b.activate_pooled(acc + 2 * (oy - first) * extended, frame_row(out, nb, oy, 0), 3 * extended,
                                  net.stem.shift, Pooling::Max, 3, extended, 2);
            }
        }
    }
}

// A conv's multiplies for output block nb, for output pixel (y, x) at accumulator row acc + y * pitch + x: one per tap
// and input block, all adding into the same rows. A 1x1 conv is the centre tap of a 3x3 one.
void conv_taps(Builder& b, const ConvLayer& conv, const MapPlacement& in, u32 nb, u32 acc, u32 rows, bool accumulate) {
    const bool strided = in.layout == Layout::Phases;
    if (in.layout == Layout::Vector || strided != (conv.stride == 2)) {
        throw std::logic_error("a conv reads a frame at stride 1 and phases at stride 2");
    }
    for (u32 dy = 0; dy < conv.kernel; ++dy) {
        for (u32 dx = 0; dx < conv.kernel; ++dx) {
            const u32 tap_y = conv.kernel == 1 ? 1 : dy;
            const u32 tap_x = conv.kernel == 1 ? 1 : dx;
            for (u32 kb = 0; kb < block_count(conv.in_channels); ++kb) {
                const u32 tile = b.add_tile(
                    weight_tile(conv.weights, (dy * conv.kernel + dx) * conv.in_channels, conv.in_channels, kb, nb));
                b.multiply(tap_row(in, kb, tap_y, tap_x), acc, rows, accumulate, tile);
                accumulate = true;
            }
        }
    }
}

// Accumulator rows y * pitch + x of a side x side output into block nb of `out`: a frame by image rows, phases by
// pixels, since neighbouring pixels go to different phases, and the pooled vector by one global average pool.
void write_out(Builder& b, u32 acc, u32 side, const MapPlacement& out, u32 nb, u32 shift) {
    const u32 pitch = side + 2;
    if (out.layout == Layout::Frame) {
        for (u32 y = 0; y < side; ++y) {
            b.activate(acc + y * pitch, frame_row(out, nb, y, 0), side, shift, ActivationFunction::Relu);
        }
    } else if (out.layout == Layout::Phases) {
        for (u32 y = 0; y < side; ++y) {
            for (u32 x = 0; x < side; ++x) {
                b.activate(acc + y * pitch + x, phase_row(out, nb, y, x), 1, shift, ActivationFunction::Relu);
            }
        }
    } else {
        // One side x side window over a map `pitch` wide; stride 7 keeps a second window from fitting.
        b.activate_pooled(acc, out.blocks[nb], side * pitch, shift, Pooling::Average, side, pitch, 7);
    }
}

// Every conv of a block but the last: relu(conv(in) + bias) into `out`. A 1x1 stride-1 conv between maps of one layout
// is pointwise, so it runs over the whole map at once, border and all; the mask keeps the bias off the border.
void compile_conv(Builder& b, const ConvLayer& conv, const MapPlacement& in, const MapPlacement& out) {
    if (conv.kernel == 1 && conv.stride == 1 && in.layout == out.layout && in.side == out.side &&
        in.layout != Layout::Vector) {
        const u32 rows = map_rows(in);
        const MapPlacement mask = b.mask(in.layout, in.side);
        for (u32 nb = 0; nb < block_count(conv.out_channels); ++nb) {
            const u32 acc = b.acc_region(rows);
            for (u32 kb = 0; kb < block_count(conv.in_channels); ++kb) {
                const u32 tile = b.add_tile(weight_tile(conv.weights, 0, conv.in_channels, kb, nb));
                b.multiply(in.blocks[kb], acc, rows, kb > 0, tile);
            }
            b.bias(conv.bias, nb, mask.blocks[0], acc, rows);
            b.activate(acc, out.blocks[nb], rows, conv.shift, ActivationFunction::Relu);
        }
        return;
    }

    const u32 side = out.side;
    const u32 rows = (side - 1) * (side + 2) + side;   // accumulator row y * pitch + x for every output pixel
    const MapPlacement mask = b.mask(Layout::Frame, side);
    for (u32 nb = 0; nb < block_count(conv.out_channels); ++nb) {
        const u32 acc = b.acc_region(rows);
        conv_taps(b, conv, in, nb, acc, rows, false);
        b.bias(conv.bias, nb, frame_row(mask, 0, 0, 0), acc, rows);
        write_out(b, acc, side, out, nb, conv.shift);
    }
}

// One block: every conv but the last into `maps`, then the last plus the shortcut and both biases into `out`, which is
// a frame, phases, or the pooled vector.
void compile_block(Builder& b, const Block& block, const MapPlacement& x, const std::vector<MapPlacement>& maps,
                   const MapPlacement& out) {
    MapPlacement in = x;
    for (std::size_t i = 0; i + 1 < block.convs.size(); ++i) {
        compile_conv(b, block.convs[i], in, maps[i]);
        in = maps[i];
    }

    const ConvLayer& last = block.convs.back();
    const u32 side  = in.side / last.stride;
    const u32 pitch = side + 2;   // a frame input keeps its pitch, and phases are laid out at this side's pitch
    const u32 rows  = (side - 1) * pitch + side;

    std::vector<i32> bias = last.bias;
    if (block.downsample && !block.down.bias.empty()) {
        bias.resize(last.out_channels, 0);
        for (u32 c = 0; c < last.out_channels; ++c) {
            bias[c] = bias[c] + block.down.bias[c];
        }
    }
    const MapPlacement mask = b.mask(Layout::Frame, side);

    for (u32 nb = 0; nb < block_count(last.out_channels); ++nb) {
        // The global pool reads whole image rows of the accumulators, including the two past the last pixel.
        const u32 acc = b.acc_region(out.layout == Layout::Vector ? side * pitch : rows);
        conv_taps(b, last, in, nb, acc, rows, false);
        if (block.downsample) {
            // A 1x1 conv at the block's stride: frame pixel (y, x), or bordered pixel (2y + 1, 2x + 1), which is
            // pixel (y, x) of phase (1, 1).
            conv_taps(b, block.down, x, nb, acc, rows, true);
        } else {
            const u32 tile = b.add_tile(identity_tile(block.residual_scale, block_width(last.out_channels, nb)));
            b.multiply(frame_row(x, nb, 0, 0), acc, rows, true, tile);
        }
        b.bias(bias, nb, frame_row(mask, 0, 0, 0), acc, rows);
        write_out(b, acc, side, out, nb, last.shift);
    }
}

}  // namespace

CompiledResNet compile_resnet(const ResNet& net, const Matrix8& image) {
    check_shapes(net, image);
    const ResNetConfig& config = net.config;

    Builder b;
    CompiledResNet out;
    out.classes = config.classes;

    const u32 extended   = config.size / 2 + 1;
    const u32 input_rows = extended * extended;
    const u32 input_host = b.alloc_host(stem_columns(image, config.size));
    const u32 input_ub   = b.alloc_ub(input_rows);

    out.stem = b.frame_map(stem_side(config), net.stem.out_channels);
    compile_stem(b, net, out.stem, input_host, input_ub);

    MapPlacement x = out.stem;
    for (std::size_t i = 0; i < net.blocks.size(); ++i) {
        const Block& block = net.blocks[i];

        // Each map is laid out for the conv that reads it.
        std::vector<MapPlacement> maps;
        u32 side = x.side;
        for (std::size_t c = 0; c + 1 < block.convs.size(); ++c) {
            side = side / block.convs[c].stride;
            maps.push_back(b.new_map(layout_for(block.convs[c + 1].stride), side, block.convs[c].out_channels));
        }
        side = side / block.convs.back().stride;

        const bool last = i + 1 == net.blocks.size();
        const u32  out_channels = block.convs.back().out_channels;
        MapPlacement y;
        if (last) {
            y = b.vector_map(out_channels);
        } else {
            y = b.new_map(layout_for(block_stride(net.blocks[i + 1])), side, out_channels);
        }
        compile_block(b, block, x, maps, y);

        out.block_convs.push_back(maps);
        if (last) {
            out.pooled = y;
        } else {
            out.block_out.push_back(y);
        }
        x = y;
    }

    // The fully connected layer on the pooled vector, then the logits back to the host.
    const u32 channels = net.fc_weights.rows;
    const u32 logits   = b.alloc_ub(block_count(config.classes));
    const MapPlacement mask = b.mask(Layout::Vector, 1);
    for (u32 nb = 0; nb < block_count(config.classes); ++nb) {
        const u32 acc = b.acc_region(1);
        for (u32 kb = 0; kb < block_count(channels); ++kb) {
            const u32 tile = b.add_tile(weight_tile(net.fc_weights, 0, channels, kb, nb));
            b.multiply(out.pooled.blocks[kb], acc, 1, kb > 0, tile);
        }
        b.bias(net.fc_bias, nb, mask.blocks[0], acc, 1);
        b.activate(acc, logits + nb, 1, net.fc_shift, ActivationFunction::Identity);
    }
    out.result_host_row = b.alloc_host({}, block_count(config.classes));
    b.host_transfer(Op::WriteHostMemory, out.result_host_row, logits, block_count(config.classes));

    b.finish();
    out.tiles   = b.tiles_fetched();
    out.program = std::move(b.program);
    return out;
}

Matrix8 read_map(const UnifiedBuffer& ub, const MapPlacement& map) {
    const u32 pixels = map.side * map.side;
    Matrix8 m(pixels, map.channels);
    for (u32 p = 0; p < pixels; ++p) {
        for (u32 kb = 0; kb < block_count(map.channels); ++kb) {
            const i8* bytes = ub.row(pixel_row(map, kb, p / map.side, p % map.side));
            for (u32 c = 0; c < block_width(map.channels, kb); ++c) {
                m.at(p, kb * kDim + c) = bytes[c];
            }
        }
    }
    return m;
}

Matrix8 read_logits(const Dram& host, const CompiledResNet& compiled) {
    Matrix8 logits(1, compiled.classes);
    std::vector<i8> row(kDim, 0);
    for (u32 nb = 0; nb < block_count(compiled.classes); ++nb) {
        host.read(static_cast<u64>(compiled.result_host_row + nb) * kDim, row.data(), kDim);
        for (u32 c = 0; c < block_width(compiled.classes, nb); ++c) {
            logits.at(0, nb * kDim + c) = row[c];
        }
    }
    return logits;
}
