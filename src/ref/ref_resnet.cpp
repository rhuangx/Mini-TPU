/// The ResNet-50 reference: random weights with shifts calibrated on one image, pretrained ones from a file, and the forward pass.

#include "ref/ref_resnet.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace {

// Weights from -7 to 7, spread evenly around zero so relu keeps about half of every map, from a fixed sequence.
Matrix8 small_weights(u32 rows, u32 cols, u32 seed) {
    Matrix8 m(rows, cols);
    u32 state = seed;
    for (i8& value : m.values) {
        state = state * 1664525u + 1013904223u;
        value = static_cast<i8>(static_cast<int>((state >> 24) % 15) - 7);
    }
    return m;
}

// Biases from -limit to limit, from a fixed sequence.
std::vector<i32> random_bias(u32 count, long long limit, u32 seed) {
    std::vector<i32> bias(count);
    u32 state = seed;
    for (i32& value : bias) {
        state = state * 1664525u + 1013904223u;
        value = static_cast<i32>(static_cast<long long>(state % (2 * limit + 1)) - limit);
    }
    return bias;
}

ConvLayer conv_layer(u32 kernel, u32 stride, u32 in_channels, u32 out_channels, u32 seed) {
    ConvLayer conv;
    conv.kernel       = kernel;
    conv.stride       = stride;
    conv.pad          = kernel / 2;
    conv.in_channels  = in_channels;
    conv.out_channels = out_channels;
    conv.weights      = small_weights(kernel * kernel * in_channels, out_channels, seed);
    return conv;
}

// The 99.9th percentile of |sum|: the largest 0.1% are allowed to saturate.
long long typical_magnitude(const Matrix32& sums) {
    std::vector<long long> magnitudes;
    magnitudes.reserve(sums.values.size());
    for (i32 value : sums.values) {
        magnitudes.push_back(std::llabs(static_cast<long long>(value)));
    }
    const std::size_t index = std::min(magnitudes.size() - 1, magnitudes.size() * 999 / 1000);
    std::nth_element(magnitudes.begin(), magnitudes.begin() + static_cast<std::ptrdiff_t>(index), magnitudes.end());
    return magnitudes[index];
}

// The smallest shift that rounds the typical magnitude to 127 or less.
u32 pick_shift(const Matrix32& sums) {
    const long long q = typical_magnitude(sums);
    u32 shift = 0;
    while (shift < 31) {
        long long rounded = q;
        if (shift > 0) {
            rounded = (q + (1LL << (shift - 1))) >> shift;
        }
        if (rounded <= 127) {
            break;
        }
        shift = shift + 1;
    }
    return shift;
}

Matrix32 add(const Matrix32& a, const Matrix32& b) {
    Matrix32 out = a;
    for (std::size_t i = 0; i < out.values.size(); ++i) {
        out.values[i] = out.values[i] + b.values[i];
    }
    return out;
}

Matrix32 add_bias(Matrix32 sums, const std::vector<i32>& bias) {
    if (bias.empty()) {
        return sums;
    }
    for (u32 r = 0; r < sums.rows; ++r) {
        for (u32 c = 0; c < sums.cols; ++c) {
            sums.at(r, c) = sums.at(r, c) + bias[c];
        }
    }
    return sums;
}

Matrix32 scaled(const Matrix8& x, i32 scale) {
    Matrix32 out(x.rows, x.cols);
    for (std::size_t i = 0; i < out.values.size(); ++i) {
        out.values[i] = static_cast<i32>(x.values[i]) * scale;
    }
    return out;
}

// conv(x) + bias, for a side x side map.
Matrix32 conv_sums(const ConvLayer& conv, const Matrix8& x, u32 side) {
    return add_bias(ref_conv2d(x, side, side, conv.kernel, conv.weights, conv.stride, conv.pad), conv.bias);
}

// The block's shortcut, as sums at its last conv's scale: the 1x1 conv, or x times the residual scale.
Matrix32 shortcut_sums(const Block& block, const Matrix8& x, u32 side_in) {
    if (block.downsample) {
        return conv_sums(block.down, x, side_in);
    }
    return scaled(x, block.residual_scale);
}

// Every block's convs and shortcut, before weights: (kernel, stride, in, out) for each.
struct BlockShape {
    std::vector<ConvLayer> convs;
    bool                   downsample = false;
    ConvLayer              down;
};

std::vector<BlockShape> block_shapes(const ResNetConfig& config) {
    std::vector<BlockShape> shapes;
    u32 in_channels = config.channels[0];
    for (std::size_t s = 0; s < config.channels.size(); ++s) {
        for (u32 i = 0; i < config.blocks[s]; ++i) {
            const u32 stride = (s > 0 && i == 0) ? 2 : 1;
            const u32 width  = config.channels[s];
            const u32 out    = 4 * width;

            BlockShape shape;
            shape.convs = {conv_layer(1, 1, in_channels, width, 0), conv_layer(3, stride, width, width, 0),
                           conv_layer(1, 1, width, out, 0)};
            shape.downsample = stride == 2 || in_channels != out;
            if (shape.downsample) {
                shape.down = conv_layer(1, stride, in_channels, out, 0);
            }
            shapes.push_back(shape);
            in_channels = out;
        }
    }
    return shapes;
}

u32 final_channels(const ResNetConfig& config) {
    return 4 * config.channels.back();
}

void check_config(const ResNetConfig& config) {
    const std::size_t stages = config.channels.size();
    if (stages == 0 || config.blocks.size() != stages) {
        throw std::invalid_argument("a ResNet needs one channel count and one block count per stage");
    }
    for (u32 count : config.blocks) {
        if (count == 0) {
            throw std::invalid_argument("every stage needs at least one block");
        }
    }
    const u32 step = 4u << (stages - 1);   // the stem divides by 4, and each later stage by 2
    if (config.size == 0 || config.size % step != 0) {
        throw std::invalid_argument("the input size " + std::to_string(config.size) + " must be a multiple of " +
                                    std::to_string(step) + " for " + std::to_string(stages) + " stages");
    }
    if (config.classes == 0) {
        throw std::invalid_argument("a ResNet needs at least one class");
    }
}

// Reads the little-endian file tools/export_resnet50.py writes.
class Reader {
public:
    explicit Reader(const std::string& path) : path_(path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("cannot open " + path);
        }
        bytes_.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    }

    void take(void* out, std::size_t count) {
        if (count > bytes_.size() - at_) {
            throw std::runtime_error(path_ + " ends early");
        }
        std::memcpy(out, bytes_.data() + at_, count);
        at_ = at_ + count;
    }

    u32 u32_value() {
        u32 value = 0;
        take(&value, sizeof(value));
        return value;
    }

    void expect_end() const {
        if (at_ != bytes_.size()) {
            throw std::runtime_error(path_ + " has " + std::to_string(bytes_.size() - at_) + " bytes past the network");
        }
    }

    // A conv record: kernel, stride, pad, in, out, shift, has_bias, then the weights and the biases.
    ConvLayer conv() {
        ConvLayer conv;
        conv.kernel       = u32_value();
        conv.stride       = u32_value();
        conv.pad          = u32_value();
        conv.in_channels  = u32_value();
        conv.out_channels = u32_value();
        conv.shift        = u32_value();
        const bool has_bias = u32_value() != 0;
        conv.weights = Matrix8(conv.kernel * conv.kernel * conv.in_channels, conv.out_channels);
        take(conv.weights.values.data(), conv.weights.values.size());
        if (has_bias) {
            conv.bias.resize(conv.out_channels);
            take(conv.bias.data(), conv.bias.size() * sizeof(i32));
        }
        return conv;
    }

private:
    std::string       path_;
    std::vector<char> bytes_;
    std::size_t       at_ = 0;
};

bool same_shape(const ConvLayer& a, const ConvLayer& b) {
    return a.kernel == b.kernel && a.stride == b.stride && a.pad == b.pad && a.in_channels == b.in_channels &&
           a.out_channels == b.out_channels;
}

void check_conv(const ConvLayer& conv, const ConvLayer& expected, const std::string& name) {
    if (!same_shape(conv, expected) || conv.shift > 31) {
        throw std::runtime_error(name + " does not have the shape its config implies");
    }
}

}  // namespace

u32 stem_side(const ResNetConfig& config) {
    return config.size / 4;
}

u32 stage_side(const ResNetConfig& config, u32 stage) {
    if (stage == 0) {
        return stem_side(config);
    }
    return stem_side(config) >> stage;
}

Matrix8 random_image(const ResNetConfig& config) {
    return random_matrix(config.size * config.size, 3, config.seed * 7919u + 17u);
}

ResNet make_resnet(const ResNetConfig& config, const Matrix8& image) {
    check_config(config);
    ResNet net;
    net.config = config;

    u32 seed = config.seed * 1000003u;
    const auto next_seed = [&seed]() {
        seed = seed + 101;
        return seed;
    };
    // Weights for a conv of this shape, and biases about a quarter of its typical sum on this input; returns its sums.
    const auto fill = [&](ConvLayer& conv, const Matrix8& x, u32 side) {
        conv.weights = small_weights(conv.weights.rows, conv.weights.cols, next_seed());
        const Matrix32 sums = ref_conv2d(x, side, side, conv.kernel, conv.weights, conv.stride, conv.pad);
        conv.bias = random_bias(conv.out_channels, std::max(1LL, typical_magnitude(sums) / 4), next_seed());
        return add_bias(sums, conv.bias);
    };

    // The stem, run on the image so its shift fits what it actually produces.
    net.stem = conv_layer(7, 2, 3, config.channels[0], 0);
    const Matrix32 stem_sums = fill(net.stem, image, config.size);
    net.stem.shift = pick_shift(stem_sums);
    Matrix8 x = ref_pool(ref_activate(stem_sums, net.stem.shift, ActivationFunction::Relu), config.size / 2, 3,
                         Pooling::Max, 2, 1);

    u32 side = stem_side(config);
    for (const BlockShape& shape : block_shapes(config)) {
        Block block;
        block.convs      = shape.convs;
        block.downsample = shape.downsample;
        block.down       = shape.down;

        Matrix8 h = x;
        u32     s = side;
        for (std::size_t i = 0; i + 1 < block.convs.size(); ++i) {
            ConvLayer& conv = block.convs[i];
            const Matrix32 sums = fill(conv, h, s);
            conv.shift = pick_shift(sums);
            h = ref_activate(sums, conv.shift, ActivationFunction::Relu);
            s = s / conv.stride;
        }

        // The identity's scale depends on the last conv's shift, so that shift comes from its conv alone; a downsample
        // has its own sums.
        ConvLayer& last = block.convs.back();
        const Matrix32 conv = fill(last, h, s);
        Matrix32 sums;
        if (block.downsample) {
            sums       = add(conv, fill(block.down, x, side));
            last.shift = pick_shift(sums);
        } else {
            last.shift           = pick_shift(conv);
            block.residual_scale = 1 << std::min<u32>(last.shift, 6);
            sums                 = add(conv, shortcut_sums(block, x, side));
        }
        x = ref_activate(sums, last.shift, ActivationFunction::Relu);

        net.blocks.push_back(block);
        side = s;
    }

    const Matrix8 pooled = ref_pool(x, side, side, Pooling::Average);
    net.fc_weights = small_weights(final_channels(config), config.classes, next_seed());
    net.fc_bias    = random_bias(config.classes, std::max(1LL, typical_magnitude(ref_matmul(pooled, net.fc_weights)) / 4),
                                 next_seed());
    net.fc_shift   = pick_shift(add_bias(ref_matmul(pooled, net.fc_weights), net.fc_bias));
    return net;
}

ResNet load_resnet(const std::string& path) {
    Reader in(path);
    char magic[8] = {};
    in.take(magic, sizeof(magic));
    if (std::memcmp(magic, "MTPURES1", sizeof(magic)) != 0) {
        throw std::runtime_error(path + " is not a Mini-TPU ResNet file");
    }

    ResNet net;
    ResNetConfig& config = net.config;
    config.size      = in.u32_value();
    config.classes   = in.u32_value();
    const u32 stages = in.u32_value();
    if (stages == 0 || stages > 8) {
        throw std::runtime_error(path + " has " + std::to_string(stages) + " stages");
    }
    config.channels.resize(stages);
    config.blocks.resize(stages);
    for (u32& channels : config.channels) {
        channels = in.u32_value();
    }
    for (u32& blocks : config.blocks) {
        blocks = in.u32_value();
    }
    check_config(config);

    net.stem = in.conv();
    check_conv(net.stem, conv_layer(7, 2, 3, config.channels[0], 0), "the stem");
    for (const BlockShape& shape : block_shapes(config)) {
        const std::string name = "block " + std::to_string(net.blocks.size());
        Block block;
        const u32 convs = in.u32_value();
        if (convs != shape.convs.size()) {
            throw std::runtime_error(name + " has " + std::to_string(convs) + " convs");
        }
        for (const ConvLayer& expected : shape.convs) {
            block.convs.push_back(in.conv());
            check_conv(block.convs.back(), expected, name + "'s conv " + std::to_string(block.convs.size()));
        }
        block.downsample = in.u32_value() != 0;
        if (block.downsample != shape.downsample) {
            throw std::runtime_error(name + "'s shortcut is not the one its config implies");
        }
        if (block.downsample) {
            block.down = in.conv();
            check_conv(block.down, shape.down, name + "'s downsample");
        }
        block.residual_scale = static_cast<i32>(in.u32_value());
        if (!block.downsample && (block.residual_scale < 1 || block.residual_scale > 127)) {
            throw std::runtime_error(name + "'s residual scale must be 1 to 127");
        }
        net.blocks.push_back(block);
    }

    ConvLayer fc = in.conv();
    check_conv(fc, conv_layer(1, 1, final_channels(config), config.classes, 0), "the classifier");
    net.fc_weights = fc.weights;
    net.fc_bias    = fc.bias;
    net.fc_shift   = fc.shift;
    in.expect_end();
    return net;
}

Matrix8 load_image(const std::string& path, u32 size) {
    Reader in(path);
    Matrix8 image(size * size, 3);
    in.take(image.values.data(), image.values.size());
    in.expect_end();
    return image;
}

ResNetTrace ref_resnet(const ResNet& net, const Matrix8& image) {
    const ResNetConfig& config = net.config;
    ResNetTrace trace;

    trace.stem = ref_pool(ref_activate(conv_sums(net.stem, image, config.size), net.stem.shift, ActivationFunction::Relu),
                          config.size / 2, 3, Pooling::Max, 2, 1);

    Matrix8 x    = trace.stem;
    u32     side = stem_side(config);
    for (const Block& block : net.blocks) {
        std::vector<Matrix8> maps;
        Matrix8 h = x;
        u32     s = side;
        for (std::size_t i = 0; i + 1 < block.convs.size(); ++i) {
            h = ref_activate(conv_sums(block.convs[i], h, s), block.convs[i].shift, ActivationFunction::Relu);
            s = s / block.convs[i].stride;
            maps.push_back(h);
        }
        const Matrix32 sums = add(conv_sums(block.convs.back(), h, s), shortcut_sums(block, x, side));
        x = ref_activate(sums, block.convs.back().shift, ActivationFunction::Relu);

        trace.block_convs.push_back(maps);
        trace.block_out.push_back(x);
        side = s;
    }

    trace.pooled = ref_pool(x, side, side, Pooling::Average);
    trace.logits = ref_activate(add_bias(ref_matmul(trace.pooled, net.fc_weights), net.fc_bias), net.fc_shift,
                                ActivationFunction::Identity);
    return trace;
}
