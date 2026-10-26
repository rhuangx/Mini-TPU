/// The reference operations, written independently of the simulator's units so that agreement means something.

#include "ref/ref_ops.h"

#include <algorithm>
#include <cmath>

namespace {

i8 clamp_to_int8(long long value) {
    if (value > 127) {
        return 127;
    }
    if (value < -128) {
        return -128;
    }
    return static_cast<i8>(value);
}

// round(value / 2^shift) with halves away from zero, through a double: dividing by a power of two is exact there.
long long rounded_quotient(i32 value, u32 shift) {
    const double divisor = std::ldexp(1.0, static_cast<int>(shift));
    return std::llround(static_cast<double>(value) / divisor);
}

i8 reference_value(i32 value, u32 shift, ActivationFunction function) {
    const double x = std::ldexp(static_cast<double>(value), -static_cast<int>(shift));
    switch (function) {
        case ActivationFunction::Identity:
            return clamp_to_int8(rounded_quotient(value, shift));
        case ActivationFunction::Relu:
            return clamp_to_int8(std::max(0LL, rounded_quotient(value, shift)));
        case ActivationFunction::Sigmoid:
            return clamp_to_int8(std::llround(128.0 / (1.0 + std::exp(-x))));
        case ActivationFunction::Tanh:
            return clamp_to_int8(std::llround(128.0 * std::tanh(x)));
    }
    return 0;
}

}  // namespace

Matrix8::Matrix8(u32 row_count, u32 col_count) : rows(row_count), cols(col_count) {
    values.resize(static_cast<std::size_t>(rows) * cols, 0);
}

i8& Matrix8::at(u32 r, u32 c) {
    return values[static_cast<std::size_t>(r) * cols + c];
}

i8 Matrix8::at(u32 r, u32 c) const {
    return values[static_cast<std::size_t>(r) * cols + c];
}

bool Matrix8::operator==(const Matrix8& other) const {
    return rows == other.rows && cols == other.cols && values == other.values;
}

Matrix32::Matrix32(u32 row_count, u32 col_count) : rows(row_count), cols(col_count) {
    values.resize(static_cast<std::size_t>(rows) * cols, 0);
}

i32& Matrix32::at(u32 r, u32 c) {
    return values[static_cast<std::size_t>(r) * cols + c];
}

i32 Matrix32::at(u32 r, u32 c) const {
    return values[static_cast<std::size_t>(r) * cols + c];
}

Matrix8 random_matrix(u32 rows, u32 cols, u32 seed) {
    Matrix8 m(rows, cols);
    u32 state = seed;
    for (i8& value : m.values) {
        state = state * 1664525u + 1013904223u;
        value = static_cast<i8>(state >> 24);
    }
    return m;
}

Matrix32 ref_matmul(const Matrix8& x, const Matrix8& w) {
    Matrix32 out(x.rows, w.cols);
    for (u32 r = 0; r < x.rows; ++r) {
        for (u32 n = 0; n < w.cols; ++n) {
            i32 sum = 0;
            for (u32 k = 0; k < x.cols; ++k) {
                sum = sum + static_cast<i32>(x.at(r, k)) * static_cast<i32>(w.at(k, n));
            }
            out.at(r, n) = sum;
        }
    }
    return out;
}

Matrix8 ref_activate(const Matrix32& sums, u32 shift, ActivationFunction function) {
    Matrix8 out(sums.rows, sums.cols);
    for (std::size_t i = 0; i < sums.values.size(); ++i) {
        out.values[i] = reference_value(sums.values[i], shift, function);
    }
    return out;
}

Matrix8 ref_pool(const Matrix8& map, u32 width, u32 size, Pooling pool, u32 stride, u32 pad) {
    if (pool == Pooling::None) {
        return map;
    }
    if (stride == 0) {
        stride = size;
    }

    const long long height        = map.rows / width;
    const long long pooled_height = (height + 2 * pad - size) / stride + 1;
    const long long pooled_width  = (width + 2 * pad - size) / stride + 1;
    Matrix8 out(static_cast<u32>(pooled_height * pooled_width), map.cols);

    for (long long py = 0; py < pooled_height; ++py) {
        for (long long px = 0; px < pooled_width; ++px) {
            for (u32 c = 0; c < map.cols; ++c) {
                long long largest = -128;
                long long sum     = 0;
                for (u32 dy = 0; dy < size; ++dy) {
                    for (u32 dx = 0; dx < size; ++dx) {
                        const long long y = py * stride + dy - pad;
                        const long long x = px * stride + dx - pad;
                        if (y < 0 || y >= height || x < 0 || x >= width) {
                            continue;
                        }
                        const long long value = map.at(static_cast<u32>(y * width + x), c);
                        largest = std::max(largest, value);
                        sum = sum + value;
                    }
                }

                i8 result = clamp_to_int8(largest);
                if (pool == Pooling::Average) {
                    const double mean = static_cast<double>(sum) / static_cast<double>(size * size);
                    result = clamp_to_int8(std::llround(mean));
                }
                out.at(static_cast<u32>(py * pooled_width + px), c) = result;
            }
        }
    }
    return out;
}

Matrix32 ref_conv2d(const Matrix8& image, u32 height, u32 width, u32 kernel, const Matrix8& weights, u32 stride, u32 pad) {
    const u32 channels   = image.cols;
    const u32 out_height = (height + 2 * pad - kernel) / stride + 1;
    const u32 out_width  = (width + 2 * pad - kernel) / stride + 1;
    Matrix32 out(out_height * out_width, weights.cols);

    // Each input value is spread across every output channel at once (n innermost), which keeps weight reads in order.
    for (u32 y = 0; y < out_height; ++y) {
        for (u32 x = 0; x < out_width; ++x) {
            i32* sums = &out.at(y * out_width + x, 0);
            for (u32 dy = 0; dy < kernel; ++dy) {
                for (u32 dx = 0; dx < kernel; ++dx) {
                    // The tap's pixel in unpadded coordinates; the padding outside the image is zero.
                    const long long iy = static_cast<long long>(y * stride + dy) - pad;
                    const long long ix = static_cast<long long>(x * stride + dx) - pad;
                    if (iy < 0 || iy >= height || ix < 0 || ix >= width) {
                        continue;
                    }
                    for (u32 c = 0; c < channels; ++c) {
                        const i32 value = image.at(static_cast<u32>(iy * width + ix), c);
                        const u32 tap   = (dy * kernel + dx) * channels + c;
                        const i8* row   = weights.values.data() + static_cast<std::size_t>(tap) * weights.cols;
                        for (u32 n = 0; n < weights.cols; ++n) {
                            sums[n] = sums[n] + value * static_cast<i32>(row[n]);
                        }
                    }
                }
            }
        }
    }
    return out;
}
