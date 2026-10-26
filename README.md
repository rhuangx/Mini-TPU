# Mini-TPU

A cycle-accurate simulator of Google's **TPUv1**, the inference chip, written in plain C++17. You drive it from a terminal. Every unit is modelled at its real size and speed:
- a 256×256 systolic array
- a 24 MiB Unified Buffer
- 4 MiB of accumulators
- a 4-tile Weight FIFO fed by DDR3
- a PCIe host link
- an activation unit with pooling

A small compiler turns dense layers, MLPs, convolutions and a whole ResNet-18 into TPU programs, and every result is checked bit for bit against plain reference loops.

```
./tpu
```

`./tpu` rebuilds anything that changed and starts the simulator.

## The machine

| Unit | Size | Speed |
|---|---|---|
| Matrix Multiply Unit | 256 × 256 int8 multiply-adds, two weight planes | a multiply over B rows takes B + 511 cycles |
| Unified Buffer | 24 MiB: 98,304 rows of 256 int8 | written and read a row at a time |
| Accumulators | 4 MiB: 4,096 rows of 256 int32 | the MXU writes, Activate reads |
| Weight FIFO | 4 tiles of 64 KiB | filled from DDR3 at 48 B/cycle: 1,366 cycles per tile |
| Weight Memory | 8 GiB DDR3 | 34 GB/s at 700 MHz |
| Host interface | PCIe Gen3 x16 | 22 B/cycle: 12 cycles per 256-byte row |
| Activation unit | int32 → int8, ReLU / sigmoid / tanh, max or average pooling | one row per cycle |

At 700 MHz the array peaks at 91.75 TOPS (65,536 multiply-adds per cycle, 2 operations each).

Instructions are 12 bytes each and issue in order, one per cycle. The units then run side by side, and an interlock holds an instruction back until what it needs is ready. [docs/microarch.md](docs/microarch.md) covers the timing and the interlocks, and [docs/isa.md](docs/isa.md) the instruction set.

## Using it

The terminal takes two commands:

| Command | Does |
|---|---|
| `load FILE` | reset the machine, assemble the file, load its code and data |
| `quit` | exit (Ctrl-C and Ctrl-D too) |

`./tpu FILE` loads the file as it starts. Enter on an empty line repeats the last command, so after editing a `.s` file, Enter reloads it. A `.s` file holds its data (`.host ROW`, `.weights TILE [ROW]`, then int8 values) and its instructions (`Op key=value …`); [docs/isa.md](docs/isa.md) has the full syntax.

## Running ResNet-18

```
cmake -S . -B build && cmake --build build --target resnet18 && ./build/resnet18
```

This builds ResNet-18 at 224×224 with random int8 weights, compiles the whole network into one TPU program, runs it on the simulator, and checks every intermediate map and all 1,000 logits against a plain-loop int8 reference. It prints the cycle count, the MXU's utilization and where the stalled cycles went. One image is about 600,000 cycles (0.86 ms at 700 MHz), and simulating it takes about 11 seconds. `--seed N` picks other weights; [docs/kernels.md](docs/kernels.md#resnet-18) explains how the network is laid out on the chip.

