/// resnet18: ResNet-18 with random int8 weights, compiled to one program, run on the simulator, checked against the reference.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "compiler/resnet.h"
#include "core/tpu.h"

namespace {

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// The first map the simulator got wrong, in network order, or "" if every one matches.
std::string first_mismatch(const Tpu& tpu, const CompiledResNet& compiled, const ResNetTrace& trace) {
    if (!(read_map(tpu.ub(), compiled.stem) == trace.stem)) {
        return "the stem";
    }
    for (std::size_t i = 0; i < compiled.block_a.size(); ++i) {
        if (!(read_map(tpu.ub(), compiled.block_a[i]) == trace.block_a[i])) {
            return "block " + std::to_string(i) + "'s first conv";
        }
        if (i < compiled.block_out.size() && !(read_map(tpu.ub(), compiled.block_out[i]) == trace.block_out[i])) {
            return "block " + std::to_string(i) + "'s output";
        }
    }
    if (!(read_map(tpu.ub(), compiled.pooled) == trace.pooled)) {
        return "the pooled vector";
    }
    if (!(read_logits(tpu.host(), compiled) == trace.logits)) {
        return "the logits";
    }
    return "";
}

u64 macs(const ResNet& net) {
    const u32 size = net.config.size;
    u64 total = static_cast<u64>(size / 2) * (size / 2) * net.stem.weights.rows * net.stem.weights.cols;
    u32 side = stem_side(net.config);
    for (const BasicBlock& block : net.blocks) {
        side = side / block.a.stride;
        const u64 pixels = static_cast<u64>(side) * side;
        total = total + pixels * block.a.weights.rows * block.a.weights.cols;
        total = total + pixels * block.b.weights.rows * block.b.weights.cols;
        if (block.downsample) {
            total = total + pixels * block.down.weights.rows * block.down.weights.cols;
        }
    }
    return total + static_cast<u64>(net.fc_weights.rows) * net.fc_weights.cols;
}

}  // namespace

int main(int argc, char** argv) {
    // Options: --size N (input side, default 224), --seed N.
    ResNetConfig config;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string option = argv[i];
        const u32 value = static_cast<u32>(std::strtoul(argv[i + 1], nullptr, 10));
        if (option == "--size") {
            config.size = value;
        } else if (option == "--seed") {
            config.seed = value;
        } else {
            std::printf("unknown option %s (expects --size N, --seed N)\n", option.c_str());
            return 2;
        }
    }

    std::printf("ResNet-18: %ux%ux3 input, %u classes, random int8 weights (seed %u)\n", config.size, config.size,
                config.classes, config.seed);

    auto start = std::chrono::steady_clock::now();
    const Matrix8 image = random_image(config);
    const ResNet net = make_resnet(config, image);
    const ResNetTrace trace = ref_resnet(net, image);
    const double reference_time = seconds_since(start);

    start = std::chrono::steady_clock::now();
    const CompiledResNet compiled = compile_resnet(net, image);
    const double compile_time = seconds_since(start);

    start = std::chrono::steady_clock::now();
    Tpu tpu;
    tpu.load(compiled.program);
    tpu.run_to_halt();
    const double run_time = seconds_since(start);

    const Stats& stats = tpu.stats();
    const double cycles = static_cast<double>(stats.cycles);
    const double total_macs = static_cast<double>(macs(net));
    const double weight_mib = static_cast<double>(compiled.program.weights.size()) * v1::kTileBytes / kMiB;
    std::printf("program:  %zu instructions, %u weight tiles fetched (%.1f MiB), %zu host input rows\n",
                compiled.program.code.size(), compiled.tiles, weight_mib, compiled.program.host[0].bytes.size() / v1::kMxuDim);
    std::printf("cycles:   %llu (%.3f ms at %u MHz)\n", static_cast<unsigned long long>(stats.cycles),
                cycles / (v1::kClockMhz * 1000.0), v1::kClockMhz);
    std::printf("MXU:      busy %.1f%% of cycles; %.2f G MACs at %.0f per cycle, %.1f%% of the 65,536 peak\n",
                100.0 * static_cast<double>(stats.mxu_busy_cycles) / cycles, total_macs / 1e9, total_macs / cycles,
                100.0 * total_macs / cycles / 65536.0);
    std::printf("stalls:\n");
    for (std::size_t k = 1; k < kStallKinds; ++k) {
        if (stats.stall_cycles[k] > 0) {
            std::printf("  %-32s %9llu  %5.1f%%\n", stall_name(static_cast<Stall>(k)),
                        static_cast<unsigned long long>(stats.stall_cycles[k]),
                        100.0 * static_cast<double>(stats.stall_cycles[k]) / cycles);
        }
    }
    std::printf("time:     reference %.1f s, compile %.1f s, simulate %.1f s\n", reference_time, compile_time, run_time);

    const std::string wrong = first_mismatch(tpu, compiled, trace);
    if (!wrong.empty()) {
        std::printf("result:   MISMATCH: %s differs from the reference\n", wrong.c_str());
        return 1;
    }
    std::printf("result:   every map and all %u logits equal the reference bit for bit\n", config.classes);
    return 0;
}
