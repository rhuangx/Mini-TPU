/// resnet: ResNet-50, pretrained or random, compiled to one program, run on the simulator, checked against the reference.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

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
    for (std::size_t i = 0; i < compiled.block_convs.size(); ++i) {
        for (std::size_t c = 0; c < compiled.block_convs[i].size(); ++c) {
            if (!(read_map(tpu.ub(), compiled.block_convs[i][c]) == trace.block_convs[i][c])) {
                return "block " + std::to_string(i) + "'s conv " + std::to_string(c + 1);
            }
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

u64 conv_macs(const ConvLayer& conv, u32 side_out) {
    return static_cast<u64>(side_out) * side_out * conv.weights.rows * conv.weights.cols;
}

u64 macs(const ResNet& net) {
    u64 total = conv_macs(net.stem, net.config.size / 2);
    u32 side = stem_side(net.config);
    for (const Block& block : net.blocks) {
        const u32 side_in = side;
        for (const ConvLayer& conv : block.convs) {
            side = side / conv.stride;
            total = total + conv_macs(conv, side);
        }
        if (block.downsample) {
            total = total + conv_macs(block.down, side_in / block.down.stride);
        }
    }
    return total + static_cast<u64>(net.fc_weights.rows) * net.fc_weights.cols;
}

std::vector<std::string> read_labels(const std::string& path) {
    std::vector<std::string> labels;
    std::ifstream file(path);
    for (std::string line; std::getline(file, line);) {
        labels.push_back(line);
    }
    return labels;
}

// The five largest logits, highest first; ties go to the lower class, as argmax would.
void print_top5(const Matrix8& logits, const std::vector<std::string>& labels) {
    std::vector<u32> order(logits.cols);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](u32 a, u32 b) { return logits.at(0, a) > logits.at(0, b); });
    std::printf("top 5:\n");
    for (u32 k = 0; k < 5 && k < order.size(); ++k) {
        const u32 c = order[k];
        const std::string name = c < labels.size() ? labels[c] : "class " + std::to_string(c);
        std::printf("  %4d  %s\n", logits.at(0, c), name.c_str());
    }
}

int usage() {
    std::printf("usage: resnet --weights FILE --image FILE [--labels FILE]   pretrained, from tools/export_resnet50.py\n"
                "       resnet [--size N] [--seed N]                         random int8 weights\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    ResNetConfig config;
    std::string weights;
    std::string image_path;
    std::string labels_path;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string option = argv[i];
        const std::string value  = argv[i + 1];
        const u32 number = static_cast<u32>(std::strtoul(value.c_str(), nullptr, 10));
        if (option == "--size") {
            config.size = number;
        } else if (option == "--seed") {
            config.seed = number;
        } else if (option == "--weights") {
            weights = value;
        } else if (option == "--image") {
            image_path = value;
        } else if (option == "--labels") {
            labels_path = value;
        } else {
            return usage();
        }
    }
    if (argc % 2 == 0 || weights.empty() != image_path.empty()) {
        return usage();
    }

    auto start = std::chrono::steady_clock::now();
    ResNet net;
    Matrix8 image;
    try {
        if (weights.empty()) {
            image = random_image(config);
            net   = make_resnet(config, image);
        } else {
            net   = load_resnet(weights);
            image = load_image(image_path, net.config.size);
        }
    } catch (const std::exception& e) {
        std::printf("error: %s\n", e.what());
        return 1;
    }
    config = net.config;
    std::printf("ResNet-%u: %ux%ux3 input, %u classes, %s\n", 3 * static_cast<u32>(net.blocks.size()) + 2, config.size, config.size, config.classes,
                weights.empty() ? ("random int8 weights (seed " + std::to_string(config.seed) + ")").c_str()
                                : ("weights " + weights + ", image " + image_path).c_str());

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

    std::size_t host_rows = 0;
    for (const DataBlock& block : compiled.program.host) {
        host_rows = host_rows + block.bytes.size() / v1::kMxuDim;
    }
    const Stats& stats = tpu.stats();
    const double cycles = static_cast<double>(stats.cycles);
    const double total_macs = static_cast<double>(macs(net));
    const double weight_mib = static_cast<double>(compiled.program.weights.size()) * v1::kTileBytes / kMiB;
    std::printf("program:  %zu instructions, %u weight tiles fetched (%.1f MiB), %zu host input rows\n",
                compiled.program.code.size(), compiled.tiles, weight_mib, host_rows);
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
    if (!weights.empty()) {
        if (labels_path.empty()) {
            labels_path = weights.substr(0, weights.find_last_of('/') + 1) + "imagenet_classes.txt";
        }
        print_top5(read_logits(tpu.host(), compiled), read_labels(labels_path));
    }
    return 0;
}
