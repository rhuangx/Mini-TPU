/// ResNet-50 in int8: the network's shape, random or pretrained weights with their shifts, and the forward pass as plain loops.

#pragma once

#include <string>
#include <vector>

#include "ref/ref_ops.h"

// The network's shape. The defaults are ResNet-50 at 224 x 224; tests use smaller ones.
struct ResNetConfig {
    u32              size     = 224;                    // input height and width, 3 channels
    std::vector<u32> channels = {64, 128, 256, 512};    // per stage: the 3x3 convs' width; each block outputs 4x that
    std::vector<u32> blocks   = {3, 4, 6, 3};           // bottleneck blocks per stage
    u32              classes  = 1000;
    u32              seed     = 1;
};

// One convolution. weights is (kernel * kernel * in_channels) x out_channels, row (dy * kernel + dx) * in_channels + c.
struct ConvLayer {
    u32              kernel       = 0;
    u32              stride       = 1;
    u32              pad          = 0;
    u32              in_channels  = 0;
    u32              out_channels = 0;
    Matrix8          weights;
    std::vector<i32> bias;        // one per output channel, added to the sums; empty for none
    u32              shift = 0;   // the Activate shift for this conv's output; a downsample conv has none of its own
};

// Every conv but the last: x = relu((conv(x) + bias) / 2^shift). The last adds the shortcut into the same sums:
// out = relu((conv(x) + bias + shortcut) / 2^shift), where the shortcut is down(x) + down.bias, or x times
// residual_scale. A bottleneck block is 1x1, 3x3, 1x1, with the stride on the 3x3 conv.
struct Block {
    std::vector<ConvLayer> convs;
    bool                   downsample = false;   // when the stride is 2 or the channel count changes
    ConvLayer              down;                 // 1x1 at the block's stride, at the last conv's scale
    i32                    residual_scale = 0;   // without downsample
};

struct ResNet {
    ResNetConfig       config;
    ConvLayer          stem;   // 7x7, stride 2, pad 3, then relu and a 3x3 max pool with stride 2 and pad 1
    std::vector<Block> blocks;
    Matrix8            fc_weights;   // final channels x classes, after a global average pool
    std::vector<i32>   fc_bias;
    u32                fc_shift = 0;
};

// Every intermediate result of a forward pass; each map holds one pixel per row.
struct ResNetTrace {
    Matrix8                           stem;          // after the max pool
    std::vector<std::vector<Matrix8>> block_convs;   // each block's maps after every conv but the last
    std::vector<Matrix8>              block_out;     // each block's output
    Matrix8                           pooled;        // 1 x final channels
    Matrix8                           logits;        // 1 x classes
};

// The side of the stem's output (size / 4), and of stage s's output.
u32 stem_side(const ResNetConfig& config);
u32 stage_side(const ResNetConfig& config, u32 stage);

// A random image of the configured size: size * size pixels, 3 channels.
Matrix8 random_image(const ResNetConfig& config);

// Random weights from -7 to 7 and random biases, then each shift chosen so this image's sums fill int8: the 99.9th
// percentile maps to 127.
ResNet make_resnet(const ResNetConfig& config, const Matrix8& image);

// A pretrained network as tools/export_resnet50.py writes it, and one of its preprocessed images: size * size pixels of
// 3 int8 channels. Both throw std::runtime_error on a file that does not fit.
ResNet  load_resnet(const std::string& path);
Matrix8 load_image(const std::string& path, u32 size);

// The forward pass, by direct convolution loops.
ResNetTrace ref_resnet(const ResNet& net, const Matrix8& image);
