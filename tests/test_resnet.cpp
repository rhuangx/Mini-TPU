/// ResNet: small networks compiled to one program and run on the simulator; every map and the logits equal the reference.

#include <stdexcept>
#include <string>

#include "compiler/resnet.h"
#include "core/tpu.h"
#include "test_framework.h"

namespace {

// Runs the network on the simulator and checks every map it keeps against the reference, in network order.
void check_against_reference(const ResNetConfig& config) {
    const Matrix8 image = random_image(config);
    const ResNet net = make_resnet(config, image);
    const ResNetTrace trace = ref_resnet(net, image);
    const CompiledResNet compiled = compile_resnet(net, image);

    Tpu tpu;
    tpu.load(compiled.program);
    tpu.run_to_halt();

    CHECK(read_map(tpu.ub(), compiled.stem) == trace.stem);
    for (std::size_t i = 0; i < compiled.block_a.size(); ++i) {
        CHECK(read_map(tpu.ub(), compiled.block_a[i]) == trace.block_a[i]);
    }
    for (std::size_t i = 0; i < compiled.block_out.size(); ++i) {
        CHECK(read_map(tpu.ub(), compiled.block_out[i]) == trace.block_out[i]);
    }
    CHECK(read_map(tpu.ub(), compiled.pooled) == trace.pooled);
    CHECK(read_logits(tpu.host(), compiled) == trace.logits);

    // A network that relu has silenced would match trivially, so the maps must still carry signal.
    long long nonzero = 0;
    for (i8 value : trace.block_out.back().values) {
        nonzero = nonzero + (value != 0 ? 1 : 0);
    }
    CHECK(nonzero > static_cast<long long>(trace.block_out.back().values.size()) / 10);
}

std::string compile_error(const ResNetConfig& config) {
    try {
        const Matrix8 image = random_image(config);
        compile_resnet(make_resnet(config, image), image);
    } catch (const std::invalid_argument& e) {
        return e.what();
    }
    return "no error";
}

}  // namespace

TEST(resnet_four_stages_match_the_reference) {
    // 32 x 32 in: the stem and its pool give 8 x 8, then 8, 4, 2 and 1; each later stage opens with a downsample block.
    ResNetConfig config;
    config.size     = 32;
    config.channels = {8, 16, 32, 64};
    config.blocks   = {2, 1, 1, 1};
    config.classes  = 10;
    check_against_reference(config);
}

TEST(resnet_wide_channels_span_blocks) {
    // 264 channels and 300 classes take two 256-wide blocks each: multiplies add across input blocks, and the identity
    // and downsample shortcuts and the fully connected layer each cover both.
    ResNetConfig config;
    config.size     = 32;
    config.channels = {8, 264};
    config.blocks   = {1, 2};
    config.classes  = 300;
    check_against_reference(config);
}

TEST(resnet_rejects_shapes_that_do_not_fit) {
    ResNetConfig config;
    config.size     = 64;   // a final map of 8 x 8 is past the 7 x 7 pool window
    config.channels = {8, 16};
    config.blocks   = {1, 1};
    config.classes  = 10;
    CHECK_EQ(compile_error(config), std::string("the final map is 8 x 8; the global average pool's window goes up to 7 x 7"));

    config.size = 36;   // not a multiple of 8 for two stages
    CHECK_EQ(compile_error(config), std::string("the input size 36 must be a multiple of 8 for 2 stages"));
}
