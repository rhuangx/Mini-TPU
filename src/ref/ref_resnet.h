/// ResNet in int8: the network's shape, random weights with calibrated shifts, and the forward pass as plain loops.

#pragma once

#include <vector>

#include "ref/ref_ops.h"

// The network's shape. The defaults are ResNet-18 at 224 x 224; tests use smaller ones.
struct ResNetConfig {
    u32              size     = 224;                    // input height and width, 3 channels
    std::vector<u32> channels = {64, 128, 256, 512};    // per stage; stage 0 keeps the stem's resolution, the rest halve it
    std::vector<u32> blocks   = {2, 2, 2, 2};           // basic blocks per stage
    u32              classes  = 1000;
    u32              seed     = 1;
};

// One convolution. weights is (kernel * kernel * in_channels) x out_channels, row (dy * kernel + dx) * in_channels + c.
struct ConvLayer {
    u32     kernel       = 0;
    u32     stride       = 1;
    u32     pad          = 0;
    u32     in_channels  = 0;
    u32     out_channels = 0;
    Matrix8 weights;
    u32     shift = 0;   // the Activate shift for this conv's output; a downsample conv has none of its own
};

// out = relu((b(relu(a(x))) + shortcut) / 2^b.shift), where the shortcut is down(x), or x times residual_scale.
struct BasicBlock {
    ConvLayer a;                    // 3x3, stride 1 or 2, pad 1
    ConvLayer b;                    // 3x3, stride 1, pad 1
    bool      downsample = false;   // true when a has stride 2
    ConvLayer down;                 // 1x1, stride 2: adds into b's sums, at b's scale
    i32       residual_scale = 0;   // without downsample: 2^min(b.shift, 6), so x passes through b's shift about whole
};

struct ResNet {
    ResNetConfig            config;
    ConvLayer               stem;   // 7x7, stride 2, pad 3, then relu and a 3x3 max pool with stride 2 and pad 1
    std::vector<BasicBlock> blocks;
    Matrix8                 fc_weights;   // channels.back() x classes, after a global average pool
    u32                     fc_shift = 0;
};

// Every intermediate result of a forward pass; each map holds one pixel per row.
struct ResNetTrace {
    Matrix8              stem;        // after the max pool
    std::vector<Matrix8> block_a;     // each block's first conv, after relu
    std::vector<Matrix8> block_out;   // each block's output
    Matrix8              pooled;      // 1 x channels.back()
    Matrix8              logits;      // 1 x classes
};

// The side of the stem's output (size / 4), and of stage s's output.
u32 stem_side(const ResNetConfig& config);
u32 stage_side(const ResNetConfig& config, u32 stage);

// A random image of the configured size: size * size pixels, 3 channels.
Matrix8 random_image(const ResNetConfig& config);

// Random weights from -7 to 7, then each shift chosen so this image's sums fill int8: the 99.9th percentile maps to 127.
ResNet make_resnet(const ResNetConfig& config, const Matrix8& image);

// The forward pass, by direct convolution loops.
ResNetTrace ref_resnet(const ResNet& net, const Matrix8& image);
