/// Golden reference: every operation the TPU performs, as plain loops with no timing, to check the simulator bit for bit.

#pragma once

#include <vector>

#include "isa/isa.h"

// A row-major matrix of int8 values.
struct Matrix8 {
    u32             rows = 0;
    u32             cols = 0;
    std::vector<i8> values;

    Matrix8() = default;
    Matrix8(u32 row_count, u32 col_count);

    i8& at(u32 r, u32 c);
    i8  at(u32 r, u32 c) const;

    bool operator==(const Matrix8& other) const;
};

// A row-major matrix of int32 values.
struct Matrix32 {
    u32              rows = 0;
    u32              cols = 0;
    std::vector<i32> values;

    Matrix32() = default;
    Matrix32(u32 row_count, u32 col_count);

    i32& at(u32 r, u32 c);
    i32  at(u32 r, u32 c) const;
};

// A matrix of int8 values from a fixed pseudo-random sequence, so tests and demos repeat exactly.
Matrix8 random_matrix(u32 rows, u32 cols, u32 seed);

// x (M x K) times w (K x N), each sum in int32.
Matrix32 ref_matmul(const Matrix8& x, const Matrix8& w);

// Every value as Activate turns it into an int8: x = value / 2^shift, then the function.
Matrix8 ref_activate(const Matrix32& sums, u32 shift, ActivationFunction function);

// Pools a feature map stored one pixel per row (width pixels per image row) over size x size windows, `stride` apart
// (0 means size). `pad` pixels around the map belong to no window: max ignores them, and avg still divides by size x size.
Matrix8 ref_pool(const Matrix8& map, u32 width, u32 size, Pooling pool, u32 stride = 0, u32 pad = 0);

// Convolution by direct loops, zero-padded by `pad` on every side; image is one pixel per row, weight row
// (dy * kernel + dx) * channels + c. The output is ((height + 2 pad - kernel) / stride + 1) rows of pixels.
Matrix32 ref_conv2d(const Matrix8& image, u32 height, u32 width, u32 kernel, const Matrix8& weights, u32 stride = 1,
                    u32 pad = 0);
