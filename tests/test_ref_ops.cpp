/// Reference ops: hand-checked small cases, and agreement with the activation unit's arithmetic over a sweep of values.

#include "compiler/compiler.h"
#include "ref/ref_ops.h"
#include "test_framework.h"
#include "units/activation.h"

namespace {

Matrix8 matrix(u32 rows, u32 cols, const std::vector<int>& values) {
    Matrix8 m(rows, cols);
    for (std::size_t i = 0; i < values.size(); ++i) {
        m.values[i] = static_cast<i8>(values[i]);
    }
    return m;
}

int value_at(const Matrix8& m, u32 r, u32 c) {
    return m.at(r, c);
}

}  // namespace

TEST(ref_matmul_small) {
    const Matrix8 x = matrix(2, 2, {1, 2, 3, 4});
    const Matrix8 w = matrix(2, 2, {5, 6, 7, -8});
    const Matrix32 out = ref_matmul(x, w);
    CHECK_EQ(out.at(0, 0), 1 * 5 + 2 * 7);
    CHECK_EQ(out.at(0, 1), 1 * 6 + 2 * -8);
    CHECK_EQ(out.at(1, 0), 3 * 5 + 4 * 7);
    CHECK_EQ(out.at(1, 1), 3 * 6 + 4 * -8);
}

TEST(ref_activate_small) {
    Matrix32 sums(1, 5);
    sums.values = {5, -5, 1000, -1000, 0};
    const Matrix8 identity = ref_activate(sums, 1, ActivationFunction::Identity);
    CHECK_EQ(value_at(identity, 0, 0), 3);      // 2.5 rounds away from zero
    CHECK_EQ(value_at(identity, 0, 1), -3);
    CHECK_EQ(value_at(identity, 0, 2), 127);    // saturates
    CHECK_EQ(value_at(identity, 0, 3), -128);

    const Matrix8 relu = ref_activate(sums, 1, ActivationFunction::Relu);
    CHECK_EQ(value_at(relu, 0, 1), 0);

    const Matrix8 sigmoid = ref_activate(sums, 0, ActivationFunction::Sigmoid);
    CHECK_EQ(value_at(sigmoid, 0, 4), 64);      // sigmoid(0) = 0.5 is 64 in Q0.7
    CHECK_EQ(value_at(sigmoid, 0, 2), 127);     // sigmoid(1000) = 1.0 saturates at 127/128
}

TEST(ref_activate_agrees_with_the_activation_unit) {
    int mismatches = 0;
    const ActivationFunction functions[] = {ActivationFunction::Identity, ActivationFunction::Relu, ActivationFunction::Sigmoid,
                                            ActivationFunction::Tanh};
    for (ActivationFunction function : functions) {
        for (u32 shift = 0; shift <= 12; ++shift) {
            Matrix32 sums(1, 1);
            for (i32 value = -70000; value <= 70000; value += 37) {
                sums.values[0] = value;
                if (ref_activate(sums, shift, function).values[0] != activate(value, shift, function)) {
                    mismatches = mismatches + 1;
                }
            }
        }
    }
    CHECK_EQ(mismatches, 0);
}

TEST(ref_pool_small) {
    // A 4x4 map with one channel: 0..15 in reading order.
    Matrix8 map(16, 1);
    for (u32 i = 0; i < 16; ++i) {
        map.values[i] = static_cast<i8>(i);
    }

    const Matrix8 largest = ref_pool(map, 4, 2, Pooling::Max);
    CHECK_EQ(largest.rows, u32{4});
    CHECK_EQ(value_at(largest, 0, 0), 5);
    CHECK_EQ(value_at(largest, 3, 0), 15);

    const Matrix8 mean = ref_pool(map, 4, 2, Pooling::Average);
    CHECK_EQ(value_at(mean, 0, 0), 3);   // (0 + 1 + 4 + 5) / 4 = 2.5 rounds to 3
    CHECK_EQ(value_at(mean, 1, 0), 5);   // (2 + 3 + 6 + 7) / 4 = 4.5 rounds to 5
}

TEST(ref_conv2d_small) {
    // A 3x3 one-channel image and a 2x2 kernel of ones: each output is its window's sum.
    const Matrix8 image   = matrix(9, 1, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    const Matrix8 weights = matrix(4, 1, {1, 1, 1, 1});
    const Matrix32 out = ref_conv2d(image, 3, 3, 2, weights);
    CHECK_EQ(out.rows, u32{4});
    CHECK_EQ(out.at(0, 0), 1 + 2 + 4 + 5);
    CHECK_EQ(out.at(3, 0), 5 + 6 + 8 + 9);
}

TEST(ref_conv2d_padded_and_strided) {
    // The same 3x3 image, a 3x3 kernel of ones, one pixel of zero padding and stride 2: the four corners' windows.
    const Matrix8 image   = matrix(9, 1, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    const Matrix8 weights = matrix(9, 1, {1, 1, 1, 1, 1, 1, 1, 1, 1});
    const Matrix32 out = ref_conv2d(image, 3, 3, 3, weights, 2, 1);
    CHECK_EQ(out.rows, u32{4});
    CHECK_EQ(out.at(0, 0), 1 + 2 + 4 + 5);   // centred on pixel (0, 0): the padding adds nothing
    CHECK_EQ(out.at(1, 0), 2 + 3 + 5 + 6);   // centred on (0, 2)
    CHECK_EQ(out.at(3, 0), 5 + 6 + 8 + 9);   // centred on (2, 2)

    const Matrix32 same = ref_conv2d(image, 3, 3, 3, weights, 1, 1);
    CHECK_EQ(same.rows, u32{9});
    CHECK_EQ(same.at(4, 0), 45);             // the centre sees the whole image
}

TEST(ref_pool_padded_and_strided) {
    // 3x3 max windows 2 apart with one pixel of padding, over a 4x4 map: padding never wins, even against negatives.
    Matrix8 map(16, 1);
    for (u32 i = 0; i < 16; ++i) {
        map.values[i] = static_cast<i8>(static_cast<int>(i) - 20);
    }
    const Matrix8 largest = ref_pool(map, 4, 3, Pooling::Max, 2, 1);
    CHECK_EQ(largest.rows, u32{4});
    CHECK_EQ(value_at(largest, 0, 0), 5 - 20);    // rows and columns {-1, 0, 1}: pixel (1, 1)
    CHECK_EQ(value_at(largest, 3, 0), 15 - 20);   // rows and columns {1, 2, 3}: pixel (3, 3)
}

TEST(ref_im2col_times_weights_is_the_convolution) {
    const Matrix8 image   = random_matrix(7 * 6, 3, 11);
    const Matrix8 weights = random_matrix(3 * 3 * 3, 5, 12);
    const Matrix32 direct = ref_conv2d(image, 7, 6, 3, weights);
    const Matrix32 viaim  = ref_matmul(im2col(image, 7, 6, 3), weights);
    CHECK(direct.values == viaim.values);
}
