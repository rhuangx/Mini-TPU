# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "torchvision", "pillow", "numpy"]
# ///
"""export_resnet50: torchvision's pretrained ResNet-50 as int8 for the TPU, and a few images to run it on.

BatchNorm is folded into each conv. Activation scales are calibrated on 100 ImageNet images (every tenth class), and
each layer gets int8 weights, int32 biases and a power-of-two shift, the only rescaling Activate can do. The int8
network is then run here, in exact integer arithmetic matching the simulator's, on the other 900 images, against float.

    uv run tools/export_resnet50.py            # writes models/resnet50.bin, imagenet_classes.txt and images/*.bin
"""

import argparse
import io
import math
import struct
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

SAMPLES_URL = "https://github.com/EliSchwartz/imagenet-sample-images/archive/refs/heads/master.zip"
DEMO = ["golden_retriever", "pug", "tiger", "goldfish", "pizza", "school_bus", "espresso", "electric_guitar"]
INPUT_SCALE = 2.64 / 127   # the normalized image spans about -2.12 to 2.64
BIAS_LIMIT = 255 * 127 * 127 + 63   # what a bias tile adds at most from one constant row
STEM_BIAS_LIMIT = (256 - 147 - 1) * 127 * 127 + 63   # the stem's im2col row has 108 columns past its 147 taps


def sample_images(models: Path) -> list[Path]:
    """One ImageNet image per class, in class order: the files are named by WordNet id, which sorts like the classes."""
    folder = models / "imagenet-sample-images-master"
    if not folder.exists():
        print(f"downloading {SAMPLES_URL}")
        with urllib.request.urlopen(SAMPLES_URL) as response:
            zipfile.ZipFile(io.BytesIO(response.read())).extractall(models)
    images = sorted(folder.glob("n*.JPEG"))
    assert len(images) == 1000, f"expected 1000 sample images in {folder}, found {len(images)}"
    return images


# --- The float network, with BatchNorm folded in --------------------------------------------------------------------

def fold(conv, bn) -> dict:
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return {"w": (conv.weight * scale[:, None, None, None]).double().detach(),
            "b": (bn.bias - bn.running_mean * scale).double().detach(),
            "stride": conv.stride[0], "pad": conv.padding[0]}


def folded_network(model) -> dict:
    blocks = []
    for layer in (model.layer1, model.layer2, model.layer3, model.layer4):
        for block in layer:
            blocks.append({"convs": [fold(block.conv1, block.bn1), fold(block.conv2, block.bn2), fold(block.conv3, block.bn3)],
                           "down": fold(block.downsample[0], block.downsample[1]) if block.downsample else None})
    return {"stem": fold(model.conv1, model.bn1), "blocks": blocks,
            "fc": {"w": model.fc.weight.double().detach(), "b": model.fc.bias.double().detach()}}


def conv(x, layer):
    return F.conv2d(x, layer["w"], layer["b"], stride=layer["stride"], padding=layer["pad"])


def float_forward(net, x, record=None):
    """The folded network in float64; `record` collects every map the TPU stores as int8, in network order."""
    keep = record.append if record is not None else (lambda _: None)
    x = F.max_pool2d(F.relu(conv(x, net["stem"])), 3, 2, 1)
    keep(x)
    for block in net["blocks"]:
        h = x
        for layer in block["convs"][:-1]:
            h = F.relu(conv(h, layer))
            keep(h)
        shortcut = conv(x, block["down"]) if block["down"] else x
        x = F.relu(conv(h, block["convs"][-1]) + shortcut)
        keep(x)
    logits = x.mean((2, 3)) @ net["fc"]["w"].T + net["fc"]["b"]
    keep(logits)
    return logits


# --- Quantization ---------------------------------------------------------------------------------------------------

def quantize_weights(w, unit):
    return torch.clamp(torch.round(w / unit), -127, 127)


def quantize_bias(b, unit, limit):
    q = torch.round(b / unit)
    clipped = int((q.abs() > limit).sum())
    if clipped:
        print(f"  warning: {clipped} biases clipped to +-{limit}")
    return torch.clamp(q, -limit, limit)


def shift_for(ratio):
    """The largest shift with 2^shift <= ratio: weights then fill at least half of int8."""
    shift = math.floor(math.log2(ratio))
    assert 0 <= shift <= 31, f"shift {shift} is out of Activate's range"
    return shift


def quantize(net, scales):
    """Every layer as int8 weights, int32 biases and a shift. A map's scale is the real value of one int8 step, so a
    conv from scale s_in to s_out sums in steps of s_out / 2^shift, and its weights are in steps of that over s_in."""
    scales = iter(scales)
    s_x = INPUT_SCALE
    q = {}

    def plain(layer, s_in, s_out, limit=BIAS_LIMIT):
        shift = shift_for(127 * s_out / (s_in * layer["w"].abs().max().item()))
        unit = s_out / 2**shift
        return {"w": quantize_weights(layer["w"], unit / s_in), "b": quantize_bias(layer["b"], unit, limit),
                "shift": shift, "stride": layer["stride"], "pad": layer["pad"]}

    s_out = next(scales)
    q["stem"] = plain(net["stem"], s_x, s_out, STEM_BIAS_LIMIT)
    s_x = s_out
    q["blocks"] = []
    for block in net["blocks"]:
        convs, s_in = [], s_x
        for layer in block["convs"][:-1]:
            s_out = next(scales)
            convs.append(plain(layer, s_in, s_out))
            s_in = s_out

        last, target = block["convs"][-1], next(scales)
        down = block["down"]
        if down:
            # Both convs sum into the same accumulators, so they share a step: the shift is the one both weights fit.
            shift = shift_for(min(127 * target / (s_in * last["w"].abs().max().item()),
                                  127 * target / (s_x * down["w"].abs().max().item())))
            unit = target / 2**shift
            residual_scale, s_out = 0, target
            q_down = {"w": quantize_weights(down["w"], unit / s_x), "b": None, "shift": 0, "stride": down["stride"],
                      "pad": down["pad"]}
        else:
            # x adds in as x * r, so one accumulator step is s_x / r: the largest r up to 127 that the weights fit.
            residual_scale = max(1, min(127, math.floor(127 * s_x / (s_in * last["w"].abs().max().item()))))
            unit = s_x / residual_scale
            shift = round(math.log2(target / unit))
            s_out = unit * 2**shift
            q_down = None
        bias = last["b"] + (down["b"] if down else 0)
        q_last = {"w": quantize_weights(last["w"], unit / s_in), "b": quantize_bias(bias, unit, BIAS_LIMIT),
                  "shift": shift, "stride": last["stride"], "pad": last["pad"]}
        q["blocks"].append({"convs": [*convs, q_last], "down": q_down, "residual_scale": residual_scale})
        s_x = s_out

    fc = {"w": net["fc"]["w"][:, :, None, None], "b": net["fc"]["b"], "stride": 1, "pad": 0}
    q["fc"] = plain(fc, s_x, next(scales))
    return q


# --- The int8 network, exactly as the simulator computes it ---------------------------------------------------------

def rounded(v):
    """llround: halves away from zero."""
    return torch.sign(v) * torch.floor(torch.abs(v) + 0.5)


def activate(sums, shift, relu=True):
    v = rounded(sums / 2**shift)
    return torch.clamp(v, 0 if relu else -128, 127)


def int_conv(x, layer):
    return F.conv2d(x, layer["w"], layer["b"], stride=layer["stride"], padding=layer["pad"])


def int_forward(q, x):
    x = F.max_pool2d(activate(int_conv(x, q["stem"]), q["stem"]["shift"]), 3, 2, 1)
    for block in q["blocks"]:
        h = x
        for layer in block["convs"][:-1]:
            h = activate(int_conv(h, layer), layer["shift"])
        last = block["convs"][-1]
        shortcut = int_conv(x, block["down"]) if block["down"] else x * block["residual_scale"]
        x = activate(int_conv(h, last) + shortcut, last["shift"])
    pooled = torch.clamp(rounded(x.sum((2, 3)) / (x.shape[2] * x.shape[3])), -128, 127)
    fc = q["fc"]
    return activate(pooled @ fc["w"][:, :, 0, 0].T + fc["b"], fc["shift"], relu=False)


def quantize_image(x):
    return torch.clamp(rounded(x / INPUT_SCALE), -128, 127)


# --- The file the simulator reads -----------------------------------------------------------------------------------

def conv_record(layer) -> bytes:
    w = layer["w"]
    out_channels, in_channels, kernel, _ = w.shape
    # Row (dy * kernel + dx) * in_channels + c, column o.
    rows = w.permute(2, 3, 1, 0).reshape(kernel * kernel * in_channels, out_channels)
    has_bias = layer["b"] is not None
    record = struct.pack("<7I", kernel, layer["stride"], layer["pad"], in_channels, out_channels, layer["shift"], has_bias)
    record += rows.numpy().astype(np.int8).tobytes()
    if has_bias:
        record += layer["b"].numpy().astype("<i4").tobytes()
    return record


def write_network(q, path: Path):
    data = b"MTPURES1" + struct.pack("<3I", 224, 1000, 4) + struct.pack("<8I", 64, 128, 256, 512, 3, 4, 6, 3)
    data += conv_record(q["stem"])
    for block in q["blocks"]:
        data += struct.pack("<I", len(block["convs"]))
        for layer in block["convs"]:
            data += conv_record(layer)
        data += struct.pack("<I", block["down"] is not None)
        if block["down"] is not None:
            data += conv_record(block["down"])
        data += struct.pack("<I", block["residual_scale"])
    data += conv_record(q["fc"])
    path.write_bytes(data)


# --- Main -----------------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "models")
    parser.add_argument("--percentile", type=float, default=99.999, help="of each map's values, mapped to 127")
    parser.add_argument("--eval", type=int, default=900, help="images to evaluate on, of the 900 not calibrated on")
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)
    torch.set_grad_enabled(False)

    weights = ResNet50_Weights.IMAGENET1K_V1
    model = resnet50(weights=weights).eval()
    net = folded_network(model)
    preprocess = weights.transforms()
    labels = weights.meta["categories"]
    files = sample_images(args.out)

    def load(indices):
        return torch.stack([preprocess(Image.open(files[i]).convert("RGB")) for i in indices]).double()

    # Each map's scale: the mean over the calibration images of its percentile, as 127 steps.
    calibration = list(range(0, 1000, 10))
    per_image = []
    for start in range(0, len(calibration), 10):
        for image in load(calibration[start:start + 10]):
            maps = []
            float_forward(net, image[None], maps)
            per_image.append([torch.quantile(m.abs().flatten()[::max(1, m.numel() // 1_000_000)].float(),
                                             args.percentile / 100).item() for m in maps[:-1]] + [maps[-1].max().item()])
    # The logits are the exception: only the largest matter, so the largest seen maps to 127 and none saturate.
    per_image = np.array(per_image)
    scales = [v / 127 for v in [*per_image[:, :-1].mean(0), per_image[:, -1].max()]]
    q = quantize(net, scales)
    print(f"calibrated {len(scales)} maps on {len(calibration)} images at the {args.percentile}th percentile")

    held_out = [i for i in range(1000) if i % 10 != 0][: args.eval]
    hits = {"float top-1": 0, "float top-5": 0, "int8 top-1": 0, "int8 top-5": 0, "int8 = float top-1": 0}
    for start in range(0, len(held_out), 25):
        batch = held_out[start:start + 25]
        x = load(batch)
        f = float_forward(net, x)
        i = int_forward(q, quantize_image(x))
        target = torch.tensor(batch)
        hits["float top-1"] += int((f.argmax(1) == target).sum())
        hits["float top-5"] += int((f.topk(5).indices == target[:, None]).any(1).sum())
        hits["int8 top-1"] += int((i.argmax(1) == target).sum())
        hits["int8 top-5"] += int((i.topk(5).indices == target[:, None]).any(1).sum())
        hits["int8 = float top-1"] += int((i.argmax(1) == f.argmax(1)).sum())
    print(f"on {len(held_out)} held-out images:")
    for name, count in hits.items():
        print(f"  {name:20s} {100 * count / len(held_out):5.1f}%")

    write_network(q, args.out / "resnet50.bin")
    (args.out / "imagenet_classes.txt").write_text("\n".join(labels) + "\n")
    (args.out / "images").mkdir(exist_ok=True)
    print("demo images (int8 top-1 here; the simulator must print the same):")
    for name in DEMO:
        index = next(i for i, f in enumerate(files) if f.stem.split("_", 1)[1] == name)
        assert index % 10 != 0, f"{name} is a calibration image"
        x = quantize_image(load([index]))
        (args.out / "images" / f"{name}.bin").write_bytes(x[0].permute(1, 2, 0).numpy().astype(np.int8).tobytes())
        logits = int_forward(q, x)[0]
        top = logits.argmax().item()
        float_top = float_forward(net, load([index]))[0].argmax().item()
        print(f"  {name:18s} -> {labels[top]} ({int(logits[top])}); float: {labels[float_top]}")
    print(f"wrote {args.out / 'resnet50.bin'}")


if __name__ == "__main__":
    main()
