# Instruction set

TPUv1 has a small set of CISC instructions. Each one moves or transforms a whole block of rows, not a single value. The host sends them to the chip, and the chip runs them in order.

## Encoding

Every instruction is **12 bytes**:

| Byte | Holds |
|---|---|
| 0 | opcode |
| 1 | flags (per instruction) |
| 2 | unused, zero |
| 3–11 | operands, little-endian (lowest byte first) |

Addresses count in the unit each memory is read in:
- **Unified Buffer:** rows of 256 bytes.
- **Accumulators:** rows of 256 int32.
- **Host memory:** rows of 256 bytes.
- **Weight Memory:** tiles of 64 KiB.

| Opcode | Instruction | Operands | Bytes |
|---|---|---|---|
| `0x00` | `Nop` | none | |
| `0x01` | `Halt` | none | |
| `0x02` | `Sync` | none | |
| `0x10` | `Read_Host_Memory` | `host` → `ub`, `rows` | ub 3–5, host 6–9, rows 10–11 |
| `0x11` | `Write_Host_Memory` | `ub` → `host`, `rows` | ub 3–5, host 6–9, rows 10–11 |
| `0x20` | `Read_Weights` | `tile` | tile 3–6 |
| `0x30` | `MatrixMultiply` | `ub`, `acc`, `rows`, `accumulate`, `new_weights` | ub 3–5, acc 6–7, rows 8–11; flags: bit 0 accumulate, bit 1 new_weights |
| `0x40` | `Activate` | `acc` → `ub`, `rows`, `shift`, `function`, optional `pool`, `size`, `width`, `stride` | ub 3–5, acc 6–7, rows 8–9, shift 10, width 11, stride byte 2 bits 0–2; flags: bits 0–1 function, 2–3 pool, 4–6 size |

Field widths limit the values:
- `ub` is 24 bits and `acc` 16.
- `rows` is 16 bits for host transfers and `Activate`, and 32 bits for `MatrixMultiply`.
- `tile` and `host` are 32 bits.
- `shift` is 0–31.
- `size` is 1–7, `width` 1–255 and `stride` 0–7.

The memories are smaller than their fields can address: the Unified Buffer has 98,304 rows, the accumulators 4,096, and Weight Memory 131,072 tiles. The range is checked when an instruction issues, and an out-of-range instruction stops the machine with an error naming its PC.

An unknown opcode byte also stops the machine, with `pc N: unknown opcode 0x..`.

## What each instruction does

**`Nop`** does nothing and takes one issue cycle.

**`Halt`** waits until every unit is idle, then stops the machine. The PC stays on the `Halt`.

**`Sync`** waits until every unit is idle, then lets the next instruction issue.

**`Read_Host_Memory host=H ub=U rows=R`** copies host rows H… into Unified Buffer rows U… over PCIe at 22 bytes per cycle, so it takes ⌈256·R / 22⌉ cycles. **`Write_Host_Memory ub=U host=H rows=R`** is the same in the other direction. Only one host transfer runs at a time.

**`Read_Weights tile=T`** reserves a Weight FIFO slot and starts fetching Weight Memory tile T. It issues as soon as a slot is free and doesn't wait for the data. The tile then arrives over 1,366 cycles at 48 bytes per cycle. Tiles arrive in the order they were read, over one DDR3 channel.

**`MatrixMultiply ub=U acc=A rows=B accumulate=0|1 new_weights=0|1`** streams Unified Buffer rows U… through the systolic array. Each output lands in accumulator row A + r: `acc[A + r][n] = Σₖ ub[U + r][k] · W[k][n]`.
- **`accumulate`:** with `accumulate=1` the sum is added to what is already there (wrapping in 32 bits), which is how an input wider than 256 is split across several multiplies. With `0` it overwrites.
- **`new_weights`:** with `new_weights=1` the array first switches to its shadow weight plane, which needs a whole tile in it. With `0` it reuses the current weights.
- **Timing:** the array is busy for B + 511 cycles.

**`Activate acc=A ub=U rows=R shift=S function=F`** reads accumulator rows A… one per cycle and writes int8 Unified Buffer rows U…. Each value becomes x = value / 2^S, and then:

| `function` | Result |
|---|---|
| `identity` | x rounded to the nearest integer (halves away from zero), saturated to −128…127 |
| `relu` | the same, with negatives set to 0 |
| `sigmoid` | round(sigmoid(x) · 128), saturated: an int8 read as a fraction of 128 (Q0.7) |
| `tanh` | round(tanh(x) · 128), saturated, also Q0.7 |

With **`pool=max|avg size=K width=W`**, the R input rows are read as a feature map W pixels wide, one pixel per row. The unit pools non-overlapping K×K windows:
- **Output:** R / K² rows.
- **Rounding:** `avg` rounds halves away from zero.
- **Shape rules:** W must be a multiple of K, and R a whole number of K-row bands of W pixels.
- **When results land:** each band's pooled rows reach the Unified Buffer on the cycle its last pixel is read.

Adding **`stride=S`** (1–7) places the windows S pixels apart instead, so they can overlap (`size=3 stride=2` is a CNN's 3×3/2 max pool) or skip pixels:
- **Output:** one row per window that fits wholly inside the map, ⌊(H − K) / S⌋ + 1 by ⌊(W − K) / S⌋ + 1 of them for a map H = R / W rows high, in reading order. Pixels past the last window are read but go into no window.
- **Shape rules:** R must be whole image rows of W pixels, and the map at least K×K. There is no padding: a compiler that needs it lays out zero pixels around the map.
- **When results land:** each pooled row reaches the Unified Buffer on the cycle the last pixel of its windows' bottom image row is read, so pooling still adds no cycles.
- **Encoding:** `stride` sits in byte 2, which is zero in every other instruction. `stride=0` is the default and means side by side, so programs without it encode exactly as before.

This is an extension of this simulator's ISA: TPUv1's pooling hardware is not documented at this level.

## Assembler syntax

A `.s` file has one statement per line, and `#` starts a comment.

```
# a comment
.host 2                                  data for host row 2 (host memory is addressed in rows)
1 -2 0x7F 0xFF                           int8 values: decimal -128..127, or hex 0x00..0xFF as raw bits (0xFF is -1)
.weights 1                               data for Weight Memory tile 1, starting at its row 0
.weights 1 3                             ... or at row 3 of tile 1 (each tile is 256 rows of 256 bytes)

Read_Host_Memory host=0x2 ub=0x10 rows=4
Read_Weights tile=1
MatrixMultiply ub=0x10 acc=0 rows=4 accumulate=0 new_weights=1
Activate acc=0 ub=0x20 rows=4 shift=8 function=relu
Activate acc=0 ub=0x40 rows=16 shift=8 function=relu pool=max size=2 width=4
Activate acc=0 ub=0x50 rows=25 shift=8 function=relu pool=max size=3 width=5 stride=2
Write_Host_Memory ub=0x20 host=0x8 rows=4
Halt
```

- **Data** fills from where its directive points, row after row, and continues across lines until the next directive or instruction.
- **Operands** are `key=value` in any order. Numbers are decimal or `0x` hex. `function=` and `pool=` take names.
- **Errors** stop at the first problem and give the file, line and column: `prog.s:3:18: error: Read_Host_Memory has no operand 'hots' (expects host, ub, rows)`.
- **Disassembly:** the disassembler prints exactly this syntax, so any instruction assembles back to the same 12 bytes.
- **`program_text`** writes a whole compiled program back out as assembler text, which is how `gen_programs` produces its `.s` files.
