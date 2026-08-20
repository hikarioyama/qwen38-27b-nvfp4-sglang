#!/usr/bin/env python3
"""Rename+reciprocal Unsloth compressed-tensors NVFP4 -> ModelOpt NVFP4.

No GPU. No repack. Streams model.safetensors tensor-by-tensor
(never loads the 27B checkpoint at once).

Unsloth:  W = e2m1(weight_packed) * fp8(weight_scale) / weight_global_scale
ModelOpt: W = e2m1(weight)        * fp8(weight_scale) * weight_scale_2

  weight_packed       -> weight             (bytes untouched)
  weight_scale        -> weight_scale       (bytes untouched)
  weight_global_scale -> weight_scale_2 = 1/x, shape []
  input_global_scale  -> input_scale    = 1/x, shape []

Copies lm_head / norms / conv / visual / mtp as-is.
Writes hf_quant_config.json (NVFP4, group_size 16, exclude lm_head).

Paths via --src/--dst or SRC_DIR/DST_DIR. Example (not required):
  --src $HOME/models/Qwen3.8-27B-NVFP4-unsloth-w4a4attn
  --dst $HOME/models/Qwen3.8-27B-NVFP4-unsloth-modelopt
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""

SRC = os.environ.get("SRC_DIR", "")
DST = os.environ.get("DST_DIR", "")

DTYPE_SIZE = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1,
    "U8": 1, "I8": 1, "U16": 2, "I16": 2, "U32": 4, "I32": 4, "U64": 8, "I64": 8,
    "BOOL": 1,
}

COPY_FILES = [
    "tokenizer.json", "vocab.json", "chat_template.jinja",
    "generation_config.json", "tokenizer_config.json",
    "video_preprocessor_config.json", "preprocessor_config.json",
    "README.md", ".gitattributes",
]

CHUNK = 1 << 26
PRODUCER = {"name": "modelopt", "version": "0.43.0"}


def nbytes_of(dtype: str, shape: list) -> int:
    n = DTYPE_SIZE[dtype]
    for d in shape:
        n *= d
    return n


def read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        return header, 8 + n


def pad_header(header_dict: dict) -> bytes:
    raw = json.dumps(header_dict, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(raw) % 8)) % 8
    if pad:
        raw = raw + (b" " * pad)
    return raw


def invert_f32(blob: bytes, name: str) -> bytes:
    if len(blob) != 4:
        raise RuntimeError(f"{name}: expected 4-byte F32, got {len(blob)}")
    (x,) = struct.unpack("<f", blob)
    if not (x > 0.0) or x != x:
        raise RuntimeError(f"{name}: bad global scale {x}")
    return struct.pack("<f", 1.0 / x)


def rename_tensor(name: str) -> tuple[str, str]:
    if name.endswith(".weight_packed"):
        return name[: -len(".weight_packed")] + ".weight", "packed"
    if name.endswith(".weight_global_scale"):
        return name[: -len(".weight_global_scale")] + ".weight_scale_2", "wgs"
    if name.endswith(".input_global_scale"):
        return name[: -len(".input_global_scale")] + ".input_scale", "igs"
    return name, "copy"


def module_of(name: str) -> str:
    for suf in (
        ".weight_packed",
        ".weight_global_scale",
        ".input_global_scale",
        ".weight_scale_2",
        ".input_scale",
        ".weight_scale",
        ".weight",
        ".bias",
    ):
        if name.endswith(suf):
            return name[: -len(suf)]
    if "." not in name:
        return name
    return name.rsplit(".", 1)[0]


def classify(src_header: dict) -> tuple[list[tuple], int, int]:
    leftover_fp8 = []
    leftover_unsloth = []
    seen_dst: dict[str, str] = {}
    out: list[tuple] = []
    n_quads = 0
    n_copy = 0
    for name, meta in src_header.items():
        if name == "__metadata__":
            continue
        dtype = meta["dtype"]
        shape = list(meta["shape"])
        if dtype == "F8_E4M3" and name.endswith(".weight"):
            leftover_fp8.append(name)
        dst_name, kind = rename_tensor(name)
        if kind != "copy":
            leftover_unsloth.append(name)
        if dst_name in seen_dst:
            raise RuntimeError(
                f"name collision {seen_dst[dst_name]!r} and {name!r} -> {dst_name!r}"
            )
        seen_dst[dst_name] = name
        if kind in ("wgs", "igs"):
            if dtype != "F32":
                raise RuntimeError(f"{name}: expected F32, got {dtype}")
            if shape not in ([1], []):
                raise RuntimeError(f"{name}: expected shape [1] or [], got {shape}")
            shape = []
            nbytes = 4
        else:
            nbytes = nbytes_of(dtype, shape)
            begin, end = meta["data_offsets"]
            if end - begin != nbytes:
                raise RuntimeError(
                    f"{name}: data_offsets size {end - begin} != {nbytes}"
                )
        if kind == "packed":
            if dtype != "U8":
                raise RuntimeError(f"{name}: expected U8, got {dtype}")
            n_quads += 1
        elif kind == "copy":
            n_copy += 1
        out.append((name, dst_name, kind, dtype, shape, nbytes))
    if leftover_fp8:
        raise RuntimeError(
            f"leftover FP8 .weight tensors ({len(leftover_fp8)}); "
            "run pack_unsloth_w4a4attn.py first. "
            f"e.g. {leftover_fp8[:3]}"
        )
    if n_quads == 0:
        raise RuntimeError("no *.weight_packed tensors; source is not Unsloth NVFP4")
    return out, n_quads, n_copy


def conv1d_excludes(header: dict) -> list[str]:
    mods = set()
    for name in header:
        if name == "__metadata__":
            continue
        mod = module_of(name)
        if mod.endswith(".linear_attn.conv1d"):
            mods.add(mod)
    return sorted(mods)


def mtp_excludes(mtp_header: dict | None) -> list[str]:
    if not mtp_header:
        return []
    mods = set()
    for name in mtp_header:
        if name == "__metadata__":
            continue
        mods.add(module_of(name))
    return sorted(mods)


def write_hf_quant_config(conv1d: list[str]) -> None:
    exclude = ["lm_head", *conv1d, "model.visual*"]
    cfg = {
        "producer": dict(PRODUCER),
        "quantization": {
            "quant_algo": "NVFP4",
            "kv_cache_quant_algo": None,
            "group_size": 16,
            "exclude_modules": exclude,
        },
    }
    path = os.path.join(DST, "hf_quant_config.json")
    with open(path, "w") as fh:
        json.dump(cfg, fh, indent=4)
        fh.write("\n")
    return exclude


def patch_config(conv1d: list[str], mtp_mods: list[str]) -> list[str]:
    src_cfg_path = os.path.join(SRC, "config.json")
    if not os.path.isfile(src_cfg_path):
        raise RuntimeError(f"missing {src_cfg_path}")
    cfg = json.load(open(src_cfg_path))
    ignore = ["lm_head", *conv1d, "model.visual*", *mtp_mods]
    cfg["quantization_config"] = {
        "config_groups": {
            "group_0": {
                "input_activations": {
                    "dynamic": False,
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16,
                },
                "weights": {
                    "dynamic": False,
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16,
                },
                "targets": ["Linear"],
            }
        },
        "ignore": ignore,
        "quant_algo": "NVFP4",
        "producer": dict(PRODUCER),
        "quant_method": "modelopt",
    }
    with open(os.path.join(DST, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
        fh.write("\n")
    return ignore


def copy_sidecars() -> None:
    mtp_src = os.path.join(SRC, "model_mtp.safetensors")
    if os.path.isfile(mtp_src):
        shutil.copy2(mtp_src, os.path.join(DST, "model_mtp.safetensors"))
    for fn in COPY_FILES:
        s = os.path.join(SRC, fn)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(DST, fn))


def write_index(out_tensors: list[tuple], total_data: int) -> int:
    src_index_path = os.path.join(SRC, "model.safetensors.index.json")
    wm: dict[str, str] = {}
    if os.path.isfile(src_index_path):
        src_index = json.load(open(src_index_path))
        for k, v in src_index.get("weight_map", {}).items():
            dst_k, _kind = rename_tensor(k)
            wm[dst_k] = v
    else:
        for _src, dst_name, _kind, _dt, _sh, _nb in out_tensors:
            wm[dst_name] = "model.safetensors"
    mtp_path = os.path.join(DST, "model_mtp.safetensors")
    mtp_size = os.path.getsize(mtp_path) if os.path.isfile(mtp_path) else 0
    if os.path.isfile(mtp_path):
        mtp_header, _ = read_header(mtp_path)
        for name in mtp_header:
            if name == "__metadata__":
                continue
            wm.setdefault(name, "model_mtp.safetensors")
    new_index = {
        "metadata": {"total_size": total_data + mtp_size},
        "weight_map": wm,
    }
    with open(os.path.join(DST, "model.safetensors.index.json"), "w") as fh:
        json.dump(new_index, fh, indent=2)
        fh.write("\n")
    return len(wm)


def stream_copy(sfil, out, data_start: int, begin: int, nbytes: int, name: str) -> None:
    sfil.seek(data_start + begin)
    remaining = nbytes
    while remaining > 0:
        chunk = sfil.read(min(CHUNK, remaining))
        if not chunk:
            raise RuntimeError(f"short read {name}")
        out.write(chunk)
        remaining -= len(chunk)


def write_model(src_path: str, dst_path: str, src_header: dict, data_start: int, out_tensors: list[tuple]) -> int:
    header_dict = {}
    offset = 0
    for _src, dst_name, _kind, dtype, shape, nbytes in out_tensors:
        header_dict[dst_name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    total_data = offset
    header_bytes = pad_header(header_dict)
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    with open(src_path, "rb") as sfil, open(dst_path, "wb") as out:
        out.write(struct.pack("<Q", len(header_bytes)))
        out.write(header_bytes)
        for src_name, dst_name, kind, _dtype, _shape, nbytes in out_tensors:
            begin, end = src_header[src_name]["data_offsets"]
            src_n = end - begin
            if kind in ("wgs", "igs"):
                sfil.seek(data_start + begin)
                blob = sfil.read(src_n)
                out_blob = invert_f32(blob, src_name)
                if len(out_blob) != nbytes:
                    raise RuntimeError(f"{dst_name}: invert size {len(out_blob)} != {nbytes}")
                out.write(out_blob)
            else:
                if src_n != nbytes:
                    raise RuntimeError(f"{src_name}: copy size {src_n} != {nbytes}")
                stream_copy(sfil, out, data_start, begin, nbytes, src_name)
    return total_data


def convert() -> dict:
    src_path = os.path.join(SRC, "model.safetensors")
    dst_path = os.path.join(DST, "model.safetensors")
    if not os.path.isfile(src_path):
        raise RuntimeError(f"missing {src_path}")
    if os.path.abspath(DST) == os.path.abspath(SRC):
        raise RuntimeError("refusing to overwrite source")
    os.makedirs(DST, exist_ok=True)

    src_header, data_start = read_header(src_path)
    out_tensors, n_quads, n_copy = classify(src_header)
    conv1d = conv1d_excludes(src_header)

    mtp_src = os.path.join(SRC, "model_mtp.safetensors")
    mtp_header = read_header(mtp_src)[0] if os.path.isfile(mtp_src) else None
    mtp_mods = mtp_excludes(mtp_header)

    total_data = write_model(src_path, dst_path, src_header, data_start, out_tensors)
    copy_sidecars()
    n_wm = write_index(out_tensors, total_data)
    hf_exclude = write_hf_quant_config(conv1d)
    ignore = patch_config(conv1d, mtp_mods)

    leftover = [
        src for src, _dst, kind, *_ in out_tensors if kind == "packed"
    ]
    report = {
        "n_quads": n_quads,
        "n_copy": n_copy,
        "n_tensors_model": len(out_tensors),
        "n_weight_map": n_wm,
        "total_data": total_data,
        "hf_exclude_head": hf_exclude[:3],
        "ignore_head": ignore[:3],
        "n_packed_renames": len(leftover),
    }
    print(
        f"done. n_quads={n_quads} n_copy={n_copy} "
        f"n_tensors={len(out_tensors)} total_data={total_data}",
        flush=True,
    )
    return report


def write_st(path: str, tensors: list[tuple[str, str, list, bytes]]) -> None:
    header = {}
    offset = 0
    for name, dtype, shape, blob in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        offset += len(blob)
    hb = pad_header(header)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hb)))
        fh.write(hb)
        for _n, _d, _s, blob in tensors:
            fh.write(blob)


def load_st(path: str) -> dict:
    header, data_start = read_header(path)
    out = {}
    with open(path, "rb") as fh:
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            b, e = meta["data_offsets"]
            fh.seek(data_start + b)
            out[name] = (meta["dtype"], list(meta["shape"]), fh.read(e - b))
    return out


def self_test() -> int:
    global SRC, DST
    packed = bytes(range(32))
    scale = b"\x00\x11\x22\x33"
    wgs = struct.pack("<f", 6400.0)
    igs = struct.pack("<f", 836.0)
    lm = os.urandom(16)
    conv = struct.pack("<ff", 1.0, 2.0)
    norm = os.urandom(8)
    mtp_norm = os.urandom(8)

    with tempfile.TemporaryDirectory(prefix="unsloth-modelopt-") as td:
        src = os.path.join(td, "src")
        dst = os.path.join(td, "dst")
        os.makedirs(src)
        write_st(
            os.path.join(src, "model.safetensors"),
            [
                ("foo.weight_packed", "U8", [4, 8], packed),
                ("foo.weight_scale", "F8_E4M3", [4, 1], scale),
                ("foo.weight_global_scale", "F32", [1], wgs),
                ("foo.input_global_scale", "F32", [1], igs),
                ("lm_head.weight", "BF16", [2, 4], lm),
                (
                    "model.language_model.layers.0.linear_attn.conv1d.weight",
                    "F32",
                    [2],
                    conv,
                ),
                ("some.norm.weight", "BF16", [4], norm),
            ],
        )
        write_st(
            os.path.join(src, "model_mtp.safetensors"),
            [("mtp.norm.weight", "BF16", [4], mtp_norm)],
        )
        with open(os.path.join(src, "config.json"), "w") as fh:
            json.dump(
                {
                    "model_type": "qwen3_5",
                    "quantization_config": {
                        "quant_method": "compressed-tensors",
                        "config_groups": {},
                        "ignore": ["lm_head"],
                    },
                },
                fh,
            )
        with open(os.path.join(src, "model.safetensors.index.json"), "w") as fh:
            json.dump(
                {
                    "metadata": {"total_size": 0},
                    "weight_map": {
                        "foo.weight_packed": "model.safetensors",
                        "foo.weight_scale": "model.safetensors",
                        "foo.weight_global_scale": "model.safetensors",
                        "foo.input_global_scale": "model.safetensors",
                        "lm_head.weight": "model.safetensors",
                        "mtp.norm.weight": "model_mtp.safetensors",
                    },
                },
                fh,
            )
        open(os.path.join(src, "tokenizer.json"), "w").write("{}\n")

        SRC, DST = src, dst
        report = convert()
        if report["n_quads"] != 1:
            raise RuntimeError(report)
        got = load_st(os.path.join(dst, "model.safetensors"))
        if "foo.weight_packed" in got or "foo.weight_global_scale" in got:
            raise RuntimeError("unsloth names leaked")
        if got["foo.weight"] != ("U8", [4, 8], packed):
            raise RuntimeError("packed bytes/shape mismatch")
        if got["foo.weight_scale"] != ("F8_E4M3", [4, 1], scale):
            raise RuntimeError("weight_scale mismatch")
        ws2_dt, ws2_sh, ws2_b = got["foo.weight_scale_2"]
        ins_dt, ins_sh, ins_b = got["foo.input_scale"]
        if ws2_dt != "F32" or ws2_sh != [] or ins_dt != "F32" or ins_sh != []:
            raise RuntimeError((ws2_dt, ws2_sh, ins_dt, ins_sh))
        ws2 = struct.unpack("<f", ws2_b)[0]
        ins = struct.unpack("<f", ins_b)[0]
        if abs(6400.0 * ws2 - 1.0) >= 1e-6:
            raise RuntimeError(f"wgs reciprocal {6400.0 * ws2}")
        if abs(836.0 * ins - 1.0) >= 1e-6:
            raise RuntimeError(f"igs reciprocal {836.0 * ins}")
        if got["lm_head.weight"][2] != lm:
            raise RuntimeError("lm_head not copied")
        if got["some.norm.weight"][2] != norm:
            raise RuntimeError("norm not copied")
        hf = json.load(open(os.path.join(dst, "hf_quant_config.json")))
        ex = hf["quantization"]["exclude_modules"]
        if hf["quantization"]["quant_algo"] != "NVFP4":
            raise RuntimeError(hf)
        if hf["quantization"]["group_size"] != 16:
            raise RuntimeError(hf)
        if "lm_head" not in ex:
            raise RuntimeError(ex)
        if "model.language_model.layers.0.linear_attn.conv1d" not in ex:
            raise RuntimeError(ex)
        cfg = json.load(open(os.path.join(dst, "config.json")))
        if cfg["quantization_config"]["quant_method"] != "modelopt":
            raise RuntimeError(cfg["quantization_config"])
        if "mtp.norm" not in cfg["quantization_config"]["ignore"]:
            raise RuntimeError(cfg["quantization_config"]["ignore"])
        mtp_dst = load_st(os.path.join(dst, "model_mtp.safetensors"))
        if mtp_dst["mtp.norm.weight"][2] != mtp_norm:
            raise RuntimeError("mtp not copied")
        idx = json.load(open(os.path.join(dst, "model.safetensors.index.json")))
        if "foo.weight" not in idx["weight_map"]:
            raise RuntimeError(idx)
        if "foo.weight_packed" in idx["weight_map"]:
            raise RuntimeError(idx)

        SRC, DST = src, src
        try:
            convert()
        except RuntimeError as exc:
            if "overwrite" not in str(exc):
                raise
        else:
            raise RuntimeError("overwrite not refused")

        bad = os.path.join(td, "fp8")
        os.makedirs(bad)
        write_st(
            os.path.join(bad, "model.safetensors"),
            [
                ("foo.weight_packed", "U8", [4, 8], packed),
                ("foo.weight_scale", "F8_E4M3", [4, 1], scale),
                ("foo.weight_global_scale", "F32", [1], wgs),
                ("foo.input_global_scale", "F32", [1], igs),
                ("bar.weight", "F8_E4M3", [2, 16], bytes(32)),
            ],
        )
        with open(os.path.join(bad, "config.json"), "w") as fh:
            json.dump({"quantization_config": {"ignore": []}}, fh)
        SRC, DST = bad, os.path.join(td, "fp8-dst")
        try:
            convert()
        except RuntimeError as exc:
            if "leftover FP8" not in str(exc):
                raise
        else:
            raise RuntimeError("leftover FP8 not refused")

    print("self-test ok", flush=True)
    return 0


def main() -> int:
    global SRC, DST
    ap = argparse.ArgumentParser(
        description=(
            "Rename+reciprocal Unsloth NVFP4 (weight_packed, divisor globals) "
            "to ModelOpt NVFP4 (weight, reciprocal globals). No GPU, no repack."
        )
    )
    ap.add_argument(
        "--src",
        default=os.environ.get("SRC_DIR", ""),
        help="packed Unsloth NVFP4 dir (or SRC_DIR). Example: $HOME/models/Qwen3.8-27B-NVFP4-unsloth-w4a4attn",
    )
    ap.add_argument(
        "--dst",
        default=os.environ.get("DST_DIR", ""),
        help="Output dir (or DST_DIR). Example: $HOME/models/Qwen3.8-27B-NVFP4-unsloth-modelopt",
    )
    ap.add_argument("--self-test", action="store_true", help="run synthetic convert checks and exit")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    SRC = args.src
    DST = args.dst
    if not SRC or not DST:
        ap.error("set --src and --dst (or SRC_DIR and DST_DIR)")
    convert()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
