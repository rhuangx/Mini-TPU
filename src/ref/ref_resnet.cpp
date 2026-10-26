/// The ResNet reference: weights from a fixed sequence, shifts calibrated on one image, and the forward pass.

#include "ref/ref_resnet.h"

#include <algorithm>
#include <cstdlib>
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

ConvLayer conv_layer(u32 kernel, u32 stride, u32 pad, u32 in_channels, u32 out_channels, u32 seed) {
    ConvLayer conv;
    conv.kernel       = kernel;
    conv.stride       = stride;
    conv.pad          = pad;
    conv.in_channels  = in_channels;
    conv.out_channels = out_channels;
    conv.weights      = small_weights(kernel * kernel * in_channels, out_channels, seed);
    return conv;
}

// The smallest shift that rounds the 99.9th percentile of |sum| to 127 or less, so only the largest 0.1% saturate.
u32 pick_shift(const Matrix32& sums) {
    std::vector<long long> magnitudes;
    magnitudes.reserve(sums.values.size());
    for (i32 value : sums.values) {
        magnitudes.push_back(std::llabs(static_cast<long long>(value)));
    }
    const std::size_t index = std::min(magnitudes.size() - 1, magnitudes.size() * 999 / 1000);
    std::nth_element(magnitudes.begin(), magnitudes.begin() + static_cast<std::ptrdiff_t>(index), magnitudes.end());
    const long long q = magnitudes[index];

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

Matrix32 scaled(const Matrix8& x, i32 scale) {
    Matrix32 out(x.rows, x.cols);
    for (std::size_t i = 0; i < out.values.size(); ++i) {
        out.values[i] = static_cast<i32>(x.values[i]) * scale;
    }
    return out;
}

// The block's shortcut, as sums at b's scale: the 1x1 stride-2 conv, or x times the residual scale.
Matrix32 shortcut_sums(const BasicBlock& block, const Matrix8& x, u32 side_in) {
    if (block.downsample) {
        return ref_conv2d(x, side_in, side_in, 1, block.down.weights, 2, 0);
    }
    return scaled(x, block.residual_scale);
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

    // The stem, run on the image so its shift fits what it actually produces.
    net.stem = conv_layer(7, 2, 3, 3, config.channels[0], next_seed());
    const Matrix32 stem_sums = ref_conv2d(image, config.size, config.size, 7, net.stem.weights, 2, 3);
    net.stem.shift = pick_shift(stem_sums);
    Matrix8 x = ref_pool(ref_activate(stem_sums, net.stem.shift, ActivationFunction::Relu), config.size / 2, 3,
                         Pooling::Max, 2, 1);

    u32 side        = stem_side(config);
    u32 in_channels = config.channels[0];
    for (std::size_t s = 0; s < config.channels.size(); ++s) {
        for (u32 i = 0; i < config.blocks[s]; ++i) {
            const u32 stride       = (s > 0 && i == 0) ? 2 : 1;
            const u32 out_channels = config.channels[s];
            if (stride == 1 && in_channels != out_channels) {
                throw std::invalid_argument("a stride-1 block keeps its channel count");
            }

            BasicBlock block;
            block.a          = conv_layer(3, stride, 1, in_channels, out_channels, next_seed());
            block.b          = conv_layer(3, 1, 1, out_channels, out_channels, next_seed());
            block.downsample = stride == 2;
            if (block.downsample) {
                block.down = conv_layer(1, 2, 0, in_channels, out_channels, next_seed());
            }

            const u32 side_out    = side / stride;
            const Matrix32 a_sums = ref_conv2d(x, side, side, 3, block.a.weights, stride, 1);
            block.a.shift         = pick_shift(a_sums);
            const Matrix8 a_out   = ref_activate(a_sums, block.a.shift, ActivationFunction::Relu);

            // The identity's scale depends on b's shift, so b's shift comes from its conv alone; a downsample has its own sums.
            const Matrix32 conv_sums = ref_conv2d(a_out, side_out, side_out, 3, block.b.weights, 1, 1);
            Matrix32 sums;
            if (block.downsample) {
                sums          = add(conv_sums, shortcut_sums(block, x, side));
                block.b.shift = pick_shift(sums);
            } else {
                block.b.shift        = pick_shift(conv_sums);
                block.residual_scale = 1 << std::min<u32>(block.b.shift, 6);
                sums                 = add(conv_sums, shortcut_sums(block, x, side));
            }
            x = ref_activate(sums, block.b.shift, ActivationFunction::Relu);

            net.blocks.push_back(block);
            side        = side_out;
            in_channels = out_channels;
        }
    }

    const Matrix8 pooled = ref_pool(x, side, side, Pooling::Average);
    net.fc_weights = small_weights(in_channels, config.classes, next_seed());
    net.fc_shift   = pick_shift(ref_matmul(pooled, net.fc_weights));
    return net;
}

ResNetTrace ref_resnet(const ResNet& net, const Matrix8& image) {
    const ResNetConfig& config = net.config;
    ResNetTrace trace;

    const Matrix32 stem_sums = ref_conv2d(image, config.size, config.size, 7, net.stem.weights, 2, 3);
    trace.stem = ref_pool(ref_activate(stem_sums, net.stem.shift, ActivationFunction::Relu), config.size / 2, 3,
                          Pooling::Max, 2, 1);

    Matrix8 x    = trace.stem;
    u32     side = stem_side(config);
    for (const BasicBlock& block : net.blocks) {
        const u32 side_out  = side / block.a.stride;
        const Matrix8 a_out = ref_activate(ref_conv2d(x, side, side, 3, block.a.weights, block.a.stride, 1), block.a.shift,
                                           ActivationFunction::Relu);
        const Matrix32 sums = add(ref_conv2d(a_out, side_out, side_out, 3, block.b.weights, 1, 1),
                                  shortcut_sums(block, x, side));
        x = ref_activate(sums, block.b.shift, ActivationFunction::Relu);

        trace.block_a.push_back(a_out);
        trace.block_out.push_back(x);
        side = side_out;
    }

    trace.pooled = ref_pool(x, side, side, Pooling::Average);
    trace.logits = ref_activate(ref_matmul(trace.pooled, net.fc_weights), net.fc_shift, ActivationFunction::Identity);
    return trace;
}
