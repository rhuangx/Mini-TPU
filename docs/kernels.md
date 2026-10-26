# The compiler and the reference ops

`src/compiler/` turns neural-network layers into TPU programs. `src/ref/` computes the same results as plain loops, and the tests require the two to agree bit for bit.

## A layer

```cpp
Layer l;
l.weights  = random_matrix(300, 280, 24);   // K inputs x N outputs, int8
l.shift    = 10;                             // Activate reads each sum as sum / 2^10
l.function = ActivationFunction::Relu;
Compiled c = compile_layers(input, {l});     // input: M rows (examples or pixels) x K columns
```

`compile_layers` chains any number of layers. Each layer's output stays in the Unified Buffer and becomes the next layer's input, and only the last layer's output goes back to the host. `read_result(tpu.host(), c)` reads it back.

## Memory layout

- **Column blocks.** A matrix is stored in 256-wide column blocks, each block holding every row. Block b covers columns 256·b to 256·b + 255, zero-padded at the edge. This matches the machine: one row of a block is one MXU input vector.
- **Host memory.** The input's blocks go at host row 0, one after another, and the result's blocks follow right after.
- **Unified Buffer.** The input and each layer's output get their own region, one after another.
- **Weight Memory.** A K × N weight matrix is cut into 256 × 256 tiles, zero-padded. Tiles are stored in the order the multiplies use them.

## The instruction schedule

For each layer, each output block gets one multiply per input block, all adding into the same accumulator rows, then one `Activate`:

```
Read_Weights ×4                          start the slow DDR3 fetches at once, alongside the PCIe input
Read_Host_Memory (each input block)
for each layer, for each output block nb:
    for each input block kb:
        MatrixMultiply  ub=input block kb, acc=buffer, accumulate=(kb > 0), new_weights=1
        Read_Weights    (the tile 4 multiplies ahead)
    Activate            acc=buffer → ub=output block nb
Write_Host_Memory (each result block)
Halt
```

Three choices make it overlap:
- **Prefetching 4 tiles ahead.** The FIFO holds 4 tiles, so each new `Read_Weights` goes in right after the multiply that frees a slot. The DDR3 channel never waits on an instruction.
- **`accumulate` chains the input blocks.** An input wider than 256 becomes several multiplies adding into the same accumulator rows, and one `Activate` reads the finished sum.
- **Double-buffered accumulators.** When a layer has at most 2,048 rows, consecutive output blocks alternate between the two halves of the accumulators. The next block's multiplies then fill one half while `Activate` drains the other. The machine's interlocks keep this correct without any `Sync`.

A layer can have at most 4,096 rows (all of the accumulators). The compiler reports shapes that don't fit, such as `layer 0: its weights have 65 rows but its input has 64 columns`.

## Convolutions

`compile_conv(image, height, width, kernel, layer)` runs a stride-1, unpadded convolution:
1. **im2col on the host.** The image, stored one pixel per row with channels across, is rearranged into one row per output pixel. Each row holds that pixel's kernel × kernel window, ordered as column (dy·kernel + dx)·channels + c.
2. **One layer on the TPU.** The rearranged matrix is multiplied by the filters, which is exactly the convolution.
3. **Pooling in the Activate.** With `layer.pool` set, the compiler fills in the output map's width, and the activation unit pools as it writes.

## The reference ops

`ref_matmul`, `ref_activate`, `ref_pool` and `ref_conv2d` are plain nested loops with no timing. They're written apart from the simulator so that agreement means something:
- **Rounding:** `ref_activate` rounds with `std::llround` on a double, where the activation unit uses integer shifts.
- **Convolution:** `ref_conv2d` convolves directly, with no im2col.

The tests check each one by hand on small cases, and sweep `ref_activate` against the activation unit across about 3,800 values × 13 shifts × 4 functions. The compiler tests then run whole programs on the simulator:
- a dense layer spread over 2×2 tiles with padding
- a 3-layer MLP (ReLU → tanh → identity)
- convolutions with max and average pooling
- a round trip through `.s` text

Each result must equal the reference exactly.

## What the schedule achieves

Prefetching hides the weight fetches behind each other, but not behind the arithmetic. A multiply over B rows keeps the array busy for B + 511 cycles, while the next tile needs 1,366 cycles to arrive and 256 to shift in. So a program with small batches waits on DDR3 most of the time:

| Program | Cycles | MXU busy | Waiting for weights |
|---|---|---|---|
| dense.s (16 rows, 1 tile) | 2,352 | 527 (22%) | 1,619 |
| mlp.s (16 rows, 3 tiles) | 5,084 | 1,581 (31%) | 3,293 |
| conv.s (36 pixels, 1 tile) | 2,310 | 547 (24%) | 1,619 |

To keep the array busy, each tile has to be used on over a thousand rows. That happens with large batches, and with convolutions over big images, where every weight is reused at every pixel. For inference on MLPs and LSTMs, where batches are small, TPUv1 is bound by weight bandwidth rather than by arithmetic.

## ResNet-18

`compile_resnet(net, image)` in `src/compiler/resnet.cpp` turns a whole ResNet into one program. `src/ref/ref_resnet.cpp` builds the network and runs it as plain loops, and `tools/resnet18` runs both and compares every map.

### The network

- **Shape:**
  - a 7×7 stride-2 stem, then a 3×3 stride-2 max pool;
  - four stages of two basic blocks, with 64, 128, 256 and 512 channels. Stages 2–4 open with a stride-2 block whose shortcut is a 1×1 stride-2 conv;
  - a global average pool and a 1,000-way fully connected layer.
- **Weights:** random int8 from −7 to 7. BatchNorm counts as folded into the weights, and there are no biases.
- **Shifts:** each conv's shift is chosen on the input image, so that the 99.9th percentile of its sums lands at 127. An identity shortcut adds x · 2^min(shift, 6), so x passes through the block's shift about whole.

### What stays on the chip

Only the image comes from the host. Every activation map stays in the Unified Buffer until the logits go back. A map uses one of three layouts, with each 256-channel block on its own:

| Layout | Feeds | Pixel (y, x) is at |
|---|---|---|
| frame | a stride-1 block | `base + (y + 1)·pitch + (x + 1)`, with pitch = side + 2 and a zero border |
| phases | a stride-2 block | the bordered map split by row and column parity into four phases. Pixel (y, x) is in phase ((y + 1) mod 2, (x + 1) mod 2), at row ⌊(y + 1) / 2⌋, column ⌊(x + 1) / 2⌋, with pitch = side / 2 + 2 |
| vector | the classifier | one row per block |

### Convolutions as shifted multiplies

- **Stride 1:** output pixel (y, x) goes to accumulator row y·pitch + x. Tap (dy, dx) reads input pixel (y + dy − 1, x + dx − 1), which is frame row (y·pitch + x) + dy·pitch + dx. Every tap reads the same run of rows, just shifted.
  - So a 3×3 conv is one `MatrixMultiply` per tap and input block, all adding into the same accumulator rows, each with its tap's 256×256 weight tile.
  - The zero border supplies the padding.
  - The two rows between image rows (x = side and side + 1) compute garbage that nothing reads.
- **Stride 2:** the block before writes its output as phases. Tap (dy, dx) then reads phase (dy mod 2, dx mod 2), shifted by ⌊dy / 2⌋·pitch + ⌊dx / 2⌋, which is again one run of rows.
- **Writing maps:**
  - A frame is written by one `Activate` per image row, which leaves the border zero.
  - Phases need one `Activate` per pixel, because neighbouring pixels go to different phases. That is 4,116 one-row `Activate`s in all.

### Residuals in the accumulators

The shortcut is one more multiply into the conv's accumulator rows, before the block's `Activate`:
- **Identity:** the block's input frame read at its centre tap, times a tile with 2^min(shift, 6) on the diagonal.
- **Downsample:** phase (1, 1) of the block's input, which is exactly its stride-2 pixels, times the 1×1 conv's weights.

Either way, relu((conv + shortcut) / 2^shift) is a single `Activate`, and no add instruction is needed.

### The stem, the pools and the classifier

- **Stem:** the host im2cols the image, giving one row per conv output pixel that holds its 7×7×3 window (147 values). The grid gets an extra zero row on top and an extra zero column on the left.
  - The rows arrive over PCIe in bands of 8 pooled rows (17 conv rows).
  - Each band is one multiply, which runs while the next band transfers.
- **Max pool:** one `Activate` per pooled row reads its three conv rows, with `pool=max size=3 stride=2`. The extra zero row and column are its padding, which is exact because every value is at least zero after relu.
- **Global average pool:** the last block's `Activate` pools its 7×7 map directly, with `pool=avg size=7 stride=7` over the 9-wide frame. The two garbage columns fall outside the window.
- **Classifier:** 2 input blocks × 4 output blocks of single-row multiplies. The logits then go back to the host.

### What one image costs

| | |
|---|---|
| Program | 5,035 instructions, 253 weight tiles (15.8 MiB) |
| Cycles | 599,132 (0.86 ms at 700 MHz) |
| MXU busy | 53% of cycles |
| Arithmetic | 1.81 G MACs, 3,028 per cycle: 4.6% of the array's peak |

| Stall cause | Share of cycles |
|---|---|
| MXU busy | 45.0% |
| weights not ready | 21.4% |
| host interface busy | 17.3% |
| accumulator rows not ready | 7.9% |
| activation unit busy | 4.8% |
| Unified Buffer rows not ready | 2.7% |

The array is busy half the time but does under 5% of its peak work:
- **Narrow layers:** a 64-channel conv uses 64 of a tile's 256 rows and 64 of its 256 columns, so only 1/16 of the PEs. A 128-channel conv uses 1/4.
- **Fill and drain:** every multiply pays 511 cycles on top of its rows (see "One multiply at a time" in [microarch.md](microarch.md)). Stage 4's maps are 7×7, which is 61 rows per multiply.
- **Weight bandwidth:** a stage-4 conv needs 36 tiles at 1,366 cycles each from DDR3, against multiplies of 572 cycles. That is most of the 21% spent waiting for weights.
- **PCIe:** the im2col'd image is 12,769 rows at 12 cycles each, about 153,000 cycles. The bands overlap it with the stem's arithmetic.

### Limits

- The final map must fit the pool window (at most 7×7), and a stage-1 conv's map must fit the 4,096 accumulator rows. Together these allow inputs up to 224×224.
- One image per program (batch 1).
- The tests run two small networks bit for bit:
  - a 32×32 input through all four stages;
  - a stage of 264 channels with 300 classes, so every multiply, shortcut and the classifier spans two 256-wide blocks.
