/// ResNet lowering: frames and phases in the UB, convolutions as one accumulating multiply per tap, residuals as one more.

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

using Tile = std::vector<i8>;

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

// The stem's input: one row per pixel of the stem conv's output, holding that pixel's 7x7 window of the image (stride 2,
// padding 3), column (dy * 7 + dx) * 3 + c. Row 0 and column 0 of this grid stay zero: after relu every value is at
// least zero, so they act as the max pool's padding of one pixel at the top and left.
std::vector<i8> stem_columns(const Matrix8& image, u32 size) {
    const u32 extended = size / 2 + 1;
    std::vector<i8> bytes(static_cast<std::size_t>(extended) * extended * kDim, 0);
    for (u32 yy = 1; yy < extended; ++yy) {
        for (u32 xx = 1; xx < extended; ++xx) {
            i8* row = bytes.data() + (static_cast<std::size_t>(yy) * extended + xx) * kDim;
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

// The first UB row tap (dy, dx) reads, for a conv whose output pixel (y, x) is accumulator row y * pitch + x.
u32 tap_row(const MapPlacement& in, u32 kb, u32 dy, u32 dx) {
    if (in.layout == MapPlacement::Layout::Frame) {
        return in.blocks[kb] + dy * in.pitch + dx;
    }
    // Stride 2: bordered pixel (2y + dy, 2x + dx) is pixel (y + dy / 2, x + dx / 2) of phase (dy % 2, dx % 2).
    return phase_base(in, kb, dy % 2, dx % 2) + (dy / 2) * in.pitch + dx / 2;
}

// Builds the code, the Weight Memory tiles and the UB layout as the network is walked.
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

    MapPlacement frame_map(u32 side, u32 channels) {
        MapPlacement map;
        map.layout   = MapPlacement::Layout::Frame;
        map.side     = side;
        map.channels = channels;
        map.pitch    = side + 2;
        for (u32 kb = 0; kb < block_count(channels); ++kb) {
            map.blocks.push_back(alloc_ub((side + 2) * map.pitch));
        }
        return map;
    }

    // Phases of a side x side map, laid out for its stride-2 consumer: the consumer's pitch, side / 2 + 2.
    MapPlacement phase_map(u32 side, u32 channels) {
        MapPlacement map;
        map.layout     = MapPlacement::Layout::Phases;
        map.side       = side;
        map.channels   = channels;
        map.pitch      = side / 2 + 2;
        map.phase_rows = (side / 2 + 1) * map.pitch;
        for (u32 kb = 0; kb < block_count(channels); ++kb) {
            map.blocks.push_back(alloc_ub(4 * map.phase_rows));
        }
        return map;
    }

    MapPlacement vector_map(u32 channels) {
        MapPlacement map;
        map.layout   = MapPlacement::Layout::Vector;
        map.side     = 1;
        map.channels = channels;
        for (u32 kb = 0; kb < block_count(channels); ++kb) {
            map.blocks.push_back(alloc_ub(1));
        }
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
    std::vector<Instr> code_;
    std::vector<u32>   fetches_;   // the tile each new-weights multiply takes, in order
    long long          active_tile_ = -1;
    u32                next_ub_     = 0;
    u32                half_        = 0;

    static Instr read_weights(u32 tile) {
        Instr in;
        in.op   = Op::ReadWeights;
        in.tile = tile;
        return in;
    }
};

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
}

// The stem: im2col'd input from the host in bands, one multiply per band, and one Activate per pooled row that applies
// relu and the 3x3 stride-2 max pool and writes the row into the first frame.
void compile_stem(Builder& b, const ResNet& net, const MapPlacement& out, u32 input_ub) {
    const u32 extended = net.config.size / 2 + 1;   // the stem conv's output side, plus the padding row and column
    const u32 pooled   = out.side;

    // Each band of pooled rows needs 2 * band + 1 conv rows in one accumulator half.
    const u32 band = std::min(pooled, (kAccHalf / extended - 1) / 2);
    if (band == 0) {
        throw std::invalid_argument("the stem's rows are too wide for the accumulators");
    }

    std::vector<u32> tiles;
    for (u32 nb = 0; nb < block_count(out.channels); ++nb) {
        tiles.push_back(b.add_tile(weight_tile(net.stem.weights, 0, kStemTaps, 0, nb)));
    }

    // Each band's rows arrive while the band before it computes; the transfers run one after another.
    u32 transferred = 0;
    const auto read_through = [&](u32 first_pooled_row) {
        const u32 last_pooled = std::min(pooled, first_pooled_row + band);
        const u32 end         = (2 * last_pooled + 1) * extended;
        if (end > transferred) {
            b.host_transfer(Op::ReadHostMemory, transferred, input_ub + transferred, end - transferred);
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

// A 3x3 conv's multiplies for output block nb: one per tap and input block, all adding into the same rows.
void conv_taps(Builder& b, const ConvLayer& conv, const MapPlacement& in, u32 nb, u32 acc, u32 rows) {
    bool first = true;
    for (u32 dy = 0; dy < 3; ++dy) {
        for (u32 dx = 0; dx < 3; ++dx) {
            for (u32 kb = 0; kb < block_count(conv.in_channels); ++kb) {
                const u32 tile = b.add_tile(weight_tile(conv.weights, (dy * 3 + dx) * conv.in_channels, conv.in_channels, kb, nb));
                b.multiply(tap_row(in, kb, dy, dx), acc, rows, !first, tile);
                first = false;
            }
        }
    }
}

// One basic block: a into a frame; b plus the shortcut into `out`, which is a frame, phases, or the pooled vector.
void compile_block(Builder& b, const BasicBlock& block, const MapPlacement& x, const MapPlacement& a_map,
                   const MapPlacement& out) {
    const u32 side  = a_map.side;
    const u32 pitch = x.pitch;   // a frame input keeps its pitch, and phases are laid out at this side's pitch
    const u32 rows  = (side - 1) * pitch + side;   // accumulator row y * pitch + x for every output pixel

    for (u32 nb = 0; nb < block_count(block.a.out_channels); ++nb) {
        const u32 acc = b.acc_region(rows);
        conv_taps(b, block.a, x, nb, acc, rows);
        for (u32 y = 0; y < side; ++y) {
            b.activate(acc + y * pitch, frame_row(a_map, nb, y, 0), side, block.a.shift, ActivationFunction::Relu);
        }
    }

    for (u32 nb = 0; nb < block_count(block.b.out_channels); ++nb) {
        // The global pool reads whole image rows of the accumulators, including the two past the last pixel.
        u32 acc_rows = rows;
        if (out.layout == MapPlacement::Layout::Vector) {
            acc_rows = side * pitch;
        }
        const u32 acc = b.acc_region(acc_rows);
        conv_taps(b, block.b, a_map, nb, acc, rows);

        if (block.downsample) {
            // The 1x1 stride-2 conv reads bordered pixel (2y + 1, 2x + 1): pixel (y, x) of phase (1, 1).
            for (u32 kb = 0; kb < block_count(block.down.in_channels); ++kb) {
                const u32 tile = b.add_tile(weight_tile(block.down.weights, 0, block.down.in_channels, kb, nb));
                b.multiply(phase_base(x, kb, 1, 1), acc, rows, true, tile);
            }
        } else {
            const u32 tile = b.add_tile(identity_tile(block.residual_scale, block_width(block.b.out_channels, nb)));
            b.multiply(frame_row(x, nb, 0, 0), acc, rows, true, tile);
        }

        if (out.layout == MapPlacement::Layout::Frame) {
            for (u32 y = 0; y < side; ++y) {
                b.activate(acc + y * pitch, frame_row(out, nb, y, 0), side, block.b.shift, ActivationFunction::Relu);
            }
        } else if (out.layout == MapPlacement::Layout::Phases) {
            // Neighbouring pixels go to different phases, so each pixel is its own Activate.
            for (u32 y = 0; y < side; ++y) {
                for (u32 xx = 0; xx < side; ++xx) {
                    b.activate(acc + y * pitch + xx, phase_row(out, nb, y, xx), 1, block.b.shift, ActivationFunction::Relu);
                }
            }
        } else {
            // One side x side window over a map `pitch` wide; stride 7 keeps a second window from fitting.
            b.activate_pooled(acc, out.blocks[nb], side * pitch, block.b.shift, Pooling::Average, side, pitch, 7);
        }
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
    b.program.host.push_back({0, stem_columns(image, config.size)});
    const u32 input_ub = b.alloc_ub(input_rows);

    out.stem = b.frame_map(stem_side(config), config.channels[0]);
    compile_stem(b, net, out.stem, input_ub);

    MapPlacement x = out.stem;
    for (std::size_t i = 0; i < net.blocks.size(); ++i) {
        const BasicBlock& block = net.blocks[i];
        const u32 side = x.side / block.a.stride;
        const MapPlacement a_map = b.frame_map(side, block.a.out_channels);

        MapPlacement y;
        const bool last = i + 1 == net.blocks.size();
        if (last) {
            y = b.vector_map(block.b.out_channels);
        } else if (net.blocks[i + 1].a.stride == 2) {
            y = b.phase_map(side, block.b.out_channels);
        } else {
            y = b.frame_map(side, block.b.out_channels);
        }
        compile_block(b, block, x, a_map, y);

        out.block_a.push_back(a_map);
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
    for (u32 nb = 0; nb < block_count(config.classes); ++nb) {
        const u32 acc = b.acc_region(1);
        for (u32 kb = 0; kb < block_count(channels); ++kb) {
            const u32 tile = b.add_tile(weight_tile(net.fc_weights, 0, channels, kb, nb));
            b.multiply(out.pooled.blocks[kb], acc, 1, kb > 0, tile);
        }
        b.activate(acc, logits + nb, 1, net.fc_shift, ActivationFunction::Identity);
    }
    out.result_host_row = input_rows;
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
        const u32 y = p / map.side;
        const u32 x = p % map.side;
        for (u32 kb = 0; kb < block_count(map.channels); ++kb) {
            u32 row = map.blocks[kb];
            if (map.layout == MapPlacement::Layout::Frame) {
                row = frame_row(map, kb, y, x);
            } else if (map.layout == MapPlacement::Layout::Phases) {
                row = phase_row(map, kb, y, x);
            }
            const i8* bytes = ub.row(row);
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
