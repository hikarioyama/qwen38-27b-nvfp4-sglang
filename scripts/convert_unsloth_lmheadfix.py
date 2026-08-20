#!/usr/bin/env python3
"""Dequantize Unsloth's FP8 lm_head -> BF16, drop lm_head.weight_scale,
and exclude lm_head from the fp8 config group. Everything else is byte-identical.

Streams model.safetensors tensor-by-tensor (no full-file RAM load).

Paths are required via --src/--dst or SRC_DIR/DST_DIR. Example (not required):
  --src $HOME/models/Qwen3.8-27B-NVFP4-unsloth
  --dst $HOME/models/Qwen3.8-27B-NVFP4-unsloth-lmheadfix
"""
import argparse
import json
import os
import shutil
import struct
import sys

import torch
from safetensors import safe_open

SRC = os.environ.get("SRC_DIR", "")
DST = os.environ.get("DST_DIR", "")

# safetensors dtype -> bytes per element
DTYPE_SIZE = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1,
    "U8": 1, "I8": 1, "U16": 2, "I16": 2, "U32": 4, "I32": 4, "U64": 8, "I64": 8,
    "BOOL": 1,
}

LM_HEAD_WEIGHT = "lm_head.weight"
LM_HEAD_SCALE = "lm_head.weight_scale"


def parse_dtype_str(dtype) -> str:
    """safe_open get_dtype -> safetensors JSON dtype string."""
    s = str(dtype)
    m = {
        "torch.float8_e4m3fn": "F8_E4M3",
        "torch.float8_e5m2": "F8_E5M2",
        "torch.bfloat16": "BF16",
        "torch.float32": "F32",
        "torch.float16": "F16",
        "torch.float64": "F64",
        "torch.uint8": "U8",
        "torch.int8": "I8",
        "torch.int16": "I16",
        "torch.int32": "I32",
        "torch.int64": "I64",
        "torch.bool": "BOOL",
    }
    return m.get(s, s)


def build_new_model_safetensors():
    src_path = os.path.join(SRC, "model.safetensors")
    dst_path = os.path.join(DST, "model.safetensors")

    with open(src_path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        src_header = json.loads(fh.read(n))
        data_start = 8 + n

    f = safe_open(src_path, framework="pt")

    # Build new header (drop lm_head.weight_scale; lm_head.weight -> BF16).
    tensors = []  # (name, dtype_str, shape_list, nbytes)
    for name, meta in src_header.items():
        if name == "__metadata__":
            continue
        if name == LM_HEAD_SCALE:
            continue
        dtype = meta["dtype"]
        shape = meta["shape"]
        if name == LM_HEAD_WEIGHT:
            dtype = "BF16"
        nbytes = 1
        for d in shape:
            nbytes *= d
        nbytes *= DTYPE_SIZE[dtype]
        tensors.append((name, dtype, shape, nbytes))

    # offsets
    header_dict = {}
    offset = 0
    for name, dtype, shape, nbytes in tensors:
        header_dict[name] = {"dtype": dtype, "shape": shape,
                             "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    total_data = offset

    header_json = json.dumps(header_dict, separators=(",", ":"))
    header_bytes = header_json.encode("utf-8")

    # Dequant lm_head into a temp bytes-on-demand (do it first, it's the first tensor).
    lm_bytes = None
    if any(name == LM_HEAD_WEIGHT for name, *_ in tensors):
        w = f.get_tensor(LM_HEAD_WEIGHT)            # torch.float8_e4m3fn [out, in]
        scale = f.get_tensor(LM_HEAD_SCALE)          # torch.bfloat16 [out, 1]
        w_bf16 = (w.float() * scale.float()).to(torch.bfloat16).contiguous()
        lm_bytes = w_bf16.view(torch.uint16).numpy().tobytes()
        del w, scale, w_bf16
        torch.cuda.empty_cache()

    with open(dst_path, "wb") as out:
        out.write(struct.pack("<Q", len(header_bytes)))
        out.write(header_bytes)
        for name, dtype, shape, nbytes in tensors:
            if name == LM_HEAD_WEIGHT:
                wrote = out.write(lm_bytes)
                assert wrote == nbytes, (name, wrote, nbytes)
            else:
                begin, end = src_header[name]["data_offsets"]
                with open(src_path, "rb") as sfil:
                    sfil.seek(data_start + begin)
                    remaining = end - begin
                    while remaining > 0:
                        chunk = sfil.read(min(1 << 26, remaining))  # 64 MiB
                        out.write(chunk)
                        remaining -= len(chunk)
            if nbytes >= (1 << 30):
                print(f"wrote {name} dtype={dtype} shape={shape} ({nbytes>>20} MiB)")
    return total_data


def main():
    global SRC, DST
    ap = argparse.ArgumentParser(
        description="Dequantize Unsloth FP8 lm_head to BF16; copy everything else byte-identical."
    )
    ap.add_argument(
        "--src",
        default=os.environ.get("SRC_DIR", ""),
        help="Unsloth NVFP4 checkpoint dir (or SRC_DIR). Example: $HOME/models/Qwen3.8-27B-NVFP4-unsloth",
    )
    ap.add_argument(
        "--dst",
        default=os.environ.get("DST_DIR", ""),
        help="Output dir (or DST_DIR). Example: $HOME/models/Qwen3.8-27B-NVFP4-unsloth-lmheadfix",
    )
    args = ap.parse_args()
    SRC = args.src
    DST = args.dst
    if not SRC or not DST:
        ap.error("set --src and --dst (or SRC_DIR and DST_DIR)")
    if os.path.abspath(DST) == os.path.abspath(SRC):
        raise RuntimeError("refusing to overwrite source")

    os.makedirs(DST, exist_ok=True)

    src_index = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
    wm = dict(src_index["weight_map"])
    # model.safetensors loses lm_head.weight_scale; mtp untouched.
    total_data = build_new_model_safetensors()
    wm.pop(LM_HEAD_SCALE, None)

    # copy mtp byte-identical
    shutil.copy2(os.path.join(SRC, "model_mtp.safetensors"), os.path.join(DST, "model_mtp.safetensors"))
    mtp_size = os.path.getsize(os.path.join(DST, "model_mtp.safetensors"))

    # new index.json
    new_index = {
        "metadata": {"total_size": total_data + mtp_size},
        "weight_map": {k: v for k, v in wm.items()},
    }
    with open(os.path.join(DST, "model.safetensors.index.json"), "w") as fh:
        json.dump(new_index, fh, indent=2)

    # config.json: drop lm_head from fp8 targets, add to ignore.
    cfg = json.load(open(os.path.join(SRC, "config.json")))
    qc = cfg["quantization_config"]
    g0 = qc["config_groups"]["group_0"]
    g0["targets"] = [t for t in g0["targets"] if "lm_head" not in t]
    ignore = qc["ignore"]
    if "lm_head" not in ignore:
        ignore.insert(0, "lm_head")
    with open(os.path.join(DST, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)

    # copy tokenizer + template + other files
    for fn in [
        "tokenizer.json", "vocab.json", "chat_template.jinja",
        "generation_config.json", "tokenizer_config.json",
        "video_preprocessor_config.json", "preprocessor_config.json",
        "README.md", ".gitattributes",
    ]:
        s = os.path.join(SRC, fn)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(DST, fn))

    print(f"done. total_data={total_data} mtp_size={mtp_size}")
    print("config group_0 targets now:", g0["targets"])
    print("ignore[0:3]:", ignore[:3])


if __name__ == "__main__":
    main()
