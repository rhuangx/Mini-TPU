# Mini-TPU

A cycle-accurate simulator of Google's **TPUv1**, the inference chip, written in plain C++17. You drive it from a terminal. Every unit is modelled at its real size and speed:
- a 256×256 systolic array
- a 24 MiB Unified Buffer
- 4 MiB of accumulators
- a 4-tile Weight FIFO fed by DDR3
- a PCIe host link
- an activation unit with pooling

A small compiler turns dense layers, MLPs, convolutions and a whole ResNet-50 into TPU programs, and every result is checked bit for bit against plain reference loops. A pretrained ResNet-50 runs on it in int8 and classifies real ImageNet photos.

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

## Running ResNet-50

```
uv run tools/export_resnet50.py
cmake -S . -B build && cmake --build build --target resnet
./build/resnet --weights models/resnet50.bin --image models/images/golden_retriever.bin
```
