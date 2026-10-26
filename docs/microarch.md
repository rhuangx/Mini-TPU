# Microarchitecture

## One cycle

`Tpu::tick()` is one clock cycle at 700 MHz. It always goes in this order:

1. **Decode.** The 12 bytes at the PC become an instruction. If the PC is past the program's end, the machine stops with `past the end of the program; is a Halt missing?`.
2. **Check.** `why_blocked` asks whether the instruction can issue this cycle. If it can't, it names exactly one stall cause, from the table below.
3. **Deadlock check.** If the instruction is blocked and no unit is working, nothing can ever unblock it. The machine stops with, for example, `pc 0: MatrixMultiply …: deadlock: weights not ready, but no unit is working`.
4. **Issue.** The instruction starts its unit, and the PC moves on (except on `Halt`). At most one instruction issues per cycle, always in program order.
5. **Advance.** Each unit takes one step, in the order host interface → Weight FIFO → MXU (weight shifter, then the array) → activation unit.
6. **Count.** The cycle counter goes up, and a blocked cycle is charged to its stall cause.

An error at any step leaves the machine exactly as it was before the cycle.

The machine is deterministic, and `load` is the only way data enters it. That makes going back simple: `Tpu::back_cycles` and `Tpu::prev_instructions` reload the program and replay to the earlier point, and they land on the identical state.

## The units

**Host interface (PCIe).** It moves whole 256-byte rows between host memory and the Unified Buffer, 22 bytes per cycle. One transfer runs at a time. A transfer of n bytes takes ⌈n / 22⌉ cycles, which is 12 for one row. Both ranges are checked before any byte moves.

**Weight FIFO (DDR3).** Four slots of 64 KiB. `Read_Weights` reserves a slot, and one DDR3 channel fills the slots in order at 48 bytes per cycle, so a tile takes 1,366 cycles (1,365 full cycles and one of 16 bytes). This is decoupled access: the instruction issues as soon as a slot is free, without waiting for the data.

**Weight shifter and the two weight planes.** Each PE holds two weights: an active one, and a shadow one being loaded.
- **Filling the shadow plane:** when the shadow plane is empty and the oldest FIFO tile has fully arrived, the shifter copies that tile in one 256-byte row per cycle, straight from its FIFO slot, 256 cycles in all. The slot frees after the last row.
- **Switching planes:** `MatrixMultiply new_weights=1` makes the full shadow plane active at no cost, and the old active plane becomes the next shadow.
- **Overlap:** the next tile shifts in while the array computes with the current one.

**Systolic array (the MXU).** 256 × 256 processing elements. PE (k, n) holds weight W[k][n].
- **Skew:** Unified Buffer row r enters array row k at step r + k. Row k starts k cycles late.
- **Movement:** activations move one PE to the right per cycle. Partial sums move one PE down per cycle, and each PE adds its product.
- **Output:** the sum for input row r, column n, leaves the bottom at step r + 255 + n and goes into the accumulators.
- **Duration:** a multiply over B rows therefore takes **B + 511 cycles**, B cycles of input plus 2·255 + 1 cycles to fill and drain.

Each PE's registers are its activation, its partial sum, and which input row that data belongs to. Every cycle the whole grid's next state is computed from the previous one and then swapped in, so the order PEs are visited in can't affect the result.

**Accumulators.** 4,096 rows of 256 int32. A multiply either overwrites its rows or, with `accumulate=1`, adds into them with 32-bit wrap-around.

**Activation unit.** It reads one accumulator row per cycle, applies shift, round, saturate and the function, and writes an int8 Unified Buffer row.
- **Pooling:** with pooling, each row is one pixel of a feature map. The unit keeps a line buffer holding one band of pooled pixels: a running maximum or a running sum.
- **Output timing:** it writes the band's pooled rows on the cycle the band's last pixel is read, so pooling adds no cycles.

## Interlocks

An instruction waits until nothing it depends on is still in flight. The check is at the level of **rows**: two units can work on different rows of the same memory at once.

| Instruction | Waits while | Stall cause |
|---|---|---|
| `Read_Host_Memory` | a host transfer is running | host interface busy |
| | the MXU is still reading any of its UB rows | Unified Buffer rows not ready |
| | an `Activate` is still writing any of its UB rows | Unified Buffer rows not ready |
| `Write_Host_Memory` | a host transfer is running | host interface busy |
| | an `Activate` is still writing any of its UB rows | Unified Buffer rows not ready |
| `Read_Weights` | all four FIFO slots are taken | weight FIFO full |
| `MatrixMultiply` | the MXU is busy | MXU busy |
| | `new_weights=1` and the shadow plane isn't full | weights not ready |
| | a host read or an `Activate` is still writing its input rows | Unified Buffer rows not ready |
| | an `Activate` hasn't finished reading its accumulator rows | accumulator rows not ready |
| `Activate` | an `Activate` is running | activation unit busy |
| | the MXU is still writing its accumulator rows | accumulator rows not ready |
| | a host transfer or the MXU is touching its output rows | Unified Buffer rows not ready |
| `Sync`, `Halt` | any unit is working | waiting for units to finish |

These are the eight stall causes. Every cycle that issues nothing is charged to exactly one of them, and the counts are kept in `Stats::stall_cycles`.

## Simplifications

- **One multiply at a time:** a `MatrixMultiply` occupies the array until its last sum drains. The next one starts afterwards rather than streaming in behind it.
- **Signed int8 only:** the 16-bit and unsigned operand modes aren't modelled.
- **No instruction transfer:** instructions are already in the machine when it starts. The host driver and the instruction buffer's PCIe transfer aren't modelled.
- **Host-side im2col:** convolutions are rearranged into rows by the compiler on the host (see [kernels.md](kernels.md)).
