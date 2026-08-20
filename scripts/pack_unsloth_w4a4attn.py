#!/usr/bin/env python3
"""Stream leftover Unsloth FP8 Linears -> compressed-tensors NVFP4.

Source is never overwritten. GPU is never used.

Paths are required via --src/--dst/--huihui or SRC_DIR/DST_DIR/HUIHUI_DIR.
Example (not required):
  --src $HOME/models/Qwen3.8-27B-NVFP4-unsloth-lmheadfix
  --dst $HOME/models/Qwen3.8-27B-NVFP4-unsloth-w4a4attn
  --huihui $HOME/models/Huihui-Qwen3.8-27B-abliterated-thinking-cut-k2-nvfp4-w4a4

  - 232 leftover FP8 weights packed e2m1+block16 (Unsloth divisor globals)
  - 96 in_proj_a/b BF16 packed the same way (Huihui already NVFP4'd these)
  - MLP 0-55 / norms / conv / mtp / lm_head copied byte-identical
  - input_global_scale = 1 / Huihui same-module input_scale (read-only steal)
  - config: drop group_0, expand group_1.targets to attn+mlp(+in_proj_a/b),
    keep lm_head in ignore, stay compressed-tensors
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from safetensors import safe_open

SRC = os.environ.get("SRC_DIR", "")
DST = os.environ.get("DST_DIR", "")
HUI = os.environ.get("HUIHUI_DIR", "")

DTYPE_SIZE = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1,
    "U8": 1, "I8": 1, "U16": 2, "I16": 2, "U32": 4, "I32": 4, "U64": 8, "I64": 8,
    "BOOL": 1,
}

FP4_MAX = 6.0
FP8_MAX = 448.0
GROUP = 16
# Midpoints between e2m1 magnitudes. torch.bucketize(right=False) + these
# matches argmin |x - level| with ties -> smaller magnitude (compressed_tensors).
_MIDPOINTS = torch.tensor(
    [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32
)
_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)

VALIDATE_MODULES = [
    "model.language_model.layers.11.self_attn.k_proj",
    "model.language_model.layers.11.self_attn.q_proj",
    "model.language_model.layers.0.linear_attn.in_proj_qkv",
    "model.language_model.layers.56.mlp.gate_proj",
]
BYTE_EXACT_PACKED = "model.language_model.layers.0.mlp.gate_proj.weight_packed"
BYTE_EXACT_LMHEAD = "lm_head.weight"
INPROJ_A_MOD = "model.language_model.layers.0.linear_attn.in_proj_a"

COPY_FILES = [
    "tokenizer.json", "vocab.json", "chat_template.jinja",
    "generation_config.json", "tokenizer_config.json",
    "video_preprocessor_config.json", "preprocessor_config.json",
    "README.md", ".gitattributes",
]


def nbytes_of(dtype: str, shape: list[int]) -> int:
    n = DTYPE_SIZE[dtype]
    for d in shape:
        n *= d
    return n


def pack_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Pack [N, K] floats to U8 [N, K/2]; low nibble = even K."""
    n, k = x.shape
    if k % 2:
        raise ValueError(f"K={k} not even")
    idx = torch.bucketize(x.abs(), _MIDPOINTS, right=False)  # 0..7
    nibbles = idx.to(torch.uint8) | (torch.signbit(x).to(torch.uint8) << 3)
    return nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    n, half = packed.shape
    lo = (packed & 0x0F).long()
    hi = (packed >> 4).long()
    codes = torch.empty((n, half * 2), dtype=torch.long)
    codes[:, 0::2] = lo
    codes[:, 1::2] = hi
    mag = _E2M1[codes & 7]
    sign = torch.where((codes & 8) != 0, -1.0, 1.0)
    return sign * mag


def pack_nvfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-form NVFP4 pack. Unsloth divisor: W ≈ e2m1 * e4m3_scale / global."""
    if w.dtype != torch.float32:
        w = w.float()
    w = w.contiguous()
    n, k = w.shape
    if k % GROUP:
        raise ValueError(f"K={k} not divisible by {GROUP}")
    amax = w.abs().max().clamp_min(torch.finfo(torch.float32).tiny)
    weight_global_scale = (FP8_MAX * FP4_MAX / amax).reshape(1).to(torch.float32)
    scale_2 = amax / (FP4_MAX * FP8_MAX)
    block_amax = w.view(n, k // GROUP, GROUP).abs().amax(dim=-1)
    weight_scale = (block_amax / (FP4_MAX * scale_2)).to(torch.float8_e4m3fn)
    sf = weight_scale.float()
    safe = torch.where(sf == 0, torch.ones_like(sf), sf)
    # pack against the stored e4m3 scale so reconstruct is consistent
    inv = 1.0 / (safe * scale_2)
    w_scaled = w * inv.repeat_interleave(GROUP, dim=1)
    w_scaled = torch.where(sf.repeat_interleave(GROUP, dim=1) == 0, torch.zeros_like(w_scaled), w_scaled)
    packed = pack_e2m1(w_scaled)
    return packed.contiguous(), weight_scale.contiguous(), weight_global_scale.contiguous()


def reconstruct(packed: torch.Tensor, scale: torch.Tensor, wgs: torch.Tensor) -> torch.Tensor:
    e2m1 = unpack_e2m1(packed)
    sf = scale.float().repeat_interleave(GROUP, dim=1)
    return e2m1 * sf / float(wgs.item())


def err_stats(ref: torch.Tensor, hat: torch.Tensor) -> dict:
    d = (ref.float() - hat.float()).abs()
    finite = torch.isfinite(d)
    n = int(d.numel())
    if n == 0:
        p99 = 0.0
    elif n <= 16_000_000:
        p99 = float(torch.quantile(d.reshape(-1), 0.99).item())
    else:
        # torch.quantile refuses > 2^24 elements; numpy quantile is exact enough
        p99 = float(__import__("numpy").quantile(d.detach().cpu().numpy().reshape(-1), 0.99))
    return {
        "max_abs": float(d.max().item()) if n else 0.0,
        "mean_abs": float(d.mean().item()) if n else 0.0,
        "p99_abs": p99,
        "nan": int((~torch.isfinite(ref.float())).sum().item() + (~torch.isfinite(hat.float())).sum().item()),
        "n": n,
        "all_finite": bool(finite.all().item()) if n else True,
    }



def tensor_bytes(t: torch.Tensor, dtype_str: str) -> bytes:
    t = t.contiguous().cpu()
    if dtype_str in ("U8", "F8_E4M3"):
        return t.view(torch.uint8).numpy().tobytes()
    if dtype_str == "F32":
        return t.to(torch.float32).numpy().tobytes()
    raise ValueError(dtype_str)


def read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        return header, 8 + n


def classify(src_header: dict) -> tuple[set[str], set[str], set[str]]:
    leftover_fp8: set[str] = set()
    leftover_scale: set[str] = set()
    inproj_ab: set[str] = set()
    for name, meta in src_header.items():
        if name == "__metadata__":
            continue
        if meta["dtype"] == "F8_E4M3" and name.endswith(".weight"):
            mod = name[: -len(".weight")]
            leftover_fp8.add(mod)
            leftover_scale.add(mod + ".weight_scale")
        if meta["dtype"] == "BF16" and (
            name.endswith(".linear_attn.in_proj_a.weight")
            or name.endswith(".linear_attn.in_proj_b.weight")
        ):
            inproj_ab.add(name[: -len(".weight")])
    return leftover_fp8, leftover_scale, inproj_ab


def nvfp4_quad(mod: str, n: int, k: int, kind: str) -> list[tuple]:
    """(name, dtype, shape, nbytes, kind, mod) x 4."""
    rows = [
        (f"{mod}.weight_packed", "U8", [n, k // 2], kind, mod),
        (f"{mod}.weight_scale", "F8_E4M3", [n, k // GROUP], kind, mod),
        (f"{mod}.weight_global_scale", "F32", [1], kind, mod),
        (f"{mod}.input_global_scale", "F32", [1], kind, mod),
    ]
    return [(nm, dt, sh, nbytes_of(dt, sh), kd, m) for nm, dt, sh, kd, m in rows]


def steal_input_scales(mods: set[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    missing = []
    with safe_open(os.path.join(HUI, "model.safetensors"), framework="pt") as hf:
        keys = set(hf.keys())
        for mod in mods:
            key = mod + ".input_scale"
            if key not in keys:
                missing.append(key)
                continue
            val = float(hf.get_tensor(key).item())
            if not (val > 0.0) or val != val:
                raise RuntimeError(f"bad Huihui input_scale {key}={val}")
            out[mod] = val
    if missing:
        raise RuntimeError(f"Huihui missing {len(missing)} input_scale e.g. {missing[:5]}")
    return out


def build_out_list(src_header, leftover_fp8, leftover_scale, inproj_ab):
    out = []
    packed_fp8 = packed_ab = copied = 0
    for name, meta in src_header.items():
        if name == "__metadata__":
            continue
        if name in leftover_scale:
            continue
        if name.endswith(".weight") and name[: -len(".weight")] in leftover_fp8:
            mod = name[: -len(".weight")]
            n, k = meta["shape"]
            out.extend(nvfp4_quad(mod, n, k, "pack_fp8"))
            packed_fp8 += 1
            continue
        if name.endswith(".weight") and name[: -len(".weight")] in inproj_ab:
            mod = name[: -len(".weight")]
            n, k = meta["shape"]
            out.extend(nvfp4_quad(mod, n, k, "pack_bf16"))
            packed_ab += 1
            continue
        dtype = meta["dtype"]
        shape = meta["shape"]
        out.append((name, dtype, shape, nbytes_of(dtype, shape), "copy", name))
        copied += 1
    return out, packed_fp8, packed_ab, copied


def inc0_kproj(src_f) -> dict:
    mod = "model.language_model.layers.11.self_attn.k_proj"
    w = src_f.get_tensor(mod + ".weight")
    sc = src_f.get_tensor(mod + ".weight_scale")
    w_ref = w.float() * sc.float()
    packed, scale, wgs = pack_nvfp4(w_ref)
    hat = reconstruct(packed, scale, wgs)
    stats = err_stats(w_ref, hat)
    stats.update({
        "module": mod,
        "shape": list(w_ref.shape),
        "amax": float(w_ref.abs().max().item()),
        "weight_global_scale": float(wgs.item()),
        "packed_shape": list(packed.shape),
        "scale_shape": list(scale.shape),
        "src_bytes": int(w.numel() + sc.numel() * 2),
        "dst_bytes": int(packed.numel() + scale.numel() + 8),
    })
    print("INC-0", json.dumps(stats, indent=2), flush=True)
    if stats["nan"] or not stats["all_finite"]:
        raise RuntimeError("INC-0 produced NaN/Inf")
    if stats["max_abs"] > 0.05:
        raise RuntimeError(f"INC-0 max_abs {stats['max_abs']} > 0.05 gate")
    # plan: amax=0.2837, wgs=9475.08, max_abs=0.0150
    if not (0.2 < stats["amax"] < 0.4):
        raise RuntimeError(f"INC-0 amax unexpected {stats['amax']}")
    if not (8000 < stats["weight_global_scale"] < 11000):
        raise RuntimeError(f"INC-0 wgs unexpected {stats['weight_global_scale']}")
    return stats


def write_model(src_path, dst_path, src_header, data_start, out_tensors, leftover_fp8, inproj_ab, hui_input, src_f):
    header_dict = {}
    offset = 0
    for name, dtype, shape, nbytes, kind, extra in out_tensors:
        header_dict[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    total_data = offset
    header_json = json.dumps(header_dict, separators=(",", ":"))
    header_bytes = header_json.encode("utf-8")
    # pad to 8-byte like official safetensors (source was unpadded; readers accept both)
    pad = (8 - (len(header_bytes) % 8)) % 8
    if pad:
        header_bytes = header_bytes + (b" " * pad)

    pack_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    t0 = time.time()
    n_pack = 0
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(src_path, "rb") as sfil, open(dst_path, "wb") as out:
        out.write(struct.pack("<Q", len(header_bytes)))
        out.write(header_bytes)
        written = 0
        for name, dtype, shape, nbytes, kind, extra in out_tensors:
            if kind == "copy":
                begin, end = src_header[extra]["data_offsets"]
                sfil.seek(data_start + begin)
                remaining = end - begin
                if remaining != nbytes:
                    raise RuntimeError(f"copy size mismatch {name}: {remaining} != {nbytes}")
                while remaining > 0:
                    chunk = sfil.read(min(1 << 26, remaining))
                    if not chunk:
                        raise RuntimeError(f"short read {name}")
                    out.write(chunk)
                    remaining -= len(chunk)
            else:
                mod = extra
                if mod not in pack_cache:
                    if kind == "pack_fp8":
                        w = src_f.get_tensor(mod + ".weight")
                        sc = src_f.get_tensor(mod + ".weight_scale")
                        w_ref = w.float() * sc.float()
                        del w, sc
                    elif kind == "pack_bf16":
                        w_ref = src_f.get_tensor(mod + ".weight").float()
                    else:
                        raise RuntimeError(kind)
                    packed, scale, wgs = pack_nvfp4(w_ref)
                    del w_ref
                    pack_cache[mod] = (packed, scale, wgs)
                    n_pack += 1
                    if n_pack % 10 == 0 or n_pack <= 3:
                        print(
                            f"  packed {n_pack}/{len(leftover_fp8)+len(inproj_ab)} "
                            f"{mod} wgs={float(wgs.item()):.4g} "
                            f"elapsed={time.time()-t0:.1f}s",
                            flush=True,
                        )
                packed, scale, wgs = pack_cache[mod]
                igs = torch.tensor([1.0 / hui_input[mod]], dtype=torch.float32)
                if name.endswith(".weight_packed"):
                    blob = tensor_bytes(packed, "U8")
                elif name.endswith(".weight_scale"):
                    blob = tensor_bytes(scale, "F8_E4M3")
                elif name.endswith(".weight_global_scale"):
                    blob = tensor_bytes(wgs, "F32")
                elif name.endswith(".input_global_scale"):
                    blob = tensor_bytes(igs, "F32")
                    # last of the quad — free cache
                    del pack_cache[mod]
                else:
                    raise RuntimeError(name)
                if len(blob) != nbytes:
                    raise RuntimeError(f"pack size mismatch {name}: {len(blob)} != {nbytes}")
                out.write(blob)
            written += nbytes
    print(f"wrote {dst_path} data={total_data} packed_modules={n_pack} wall={time.time()-t0:.1f}s", flush=True)
    return total_data


def patch_config():
    cfg = json.load(open(os.path.join(SRC, "config.json")))
    qc = cfg["quantization_config"]
    groups = qc["config_groups"]
    if "group_0" not in groups:
        raise RuntimeError("group_0 already absent")
    del groups["group_0"]
    g1 = groups["group_1"]
    g1["targets"] = [
        r"re:.*self_attn\.(q|k|v|o)_proj$",
        r"re:.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj|in_proj_a|in_proj_b)$",
        r"re:.*mlp\.(gate|up|down)_proj$",
    ]
    ignore = qc["ignore"]
    ignore = [
        x for x in ignore
        if not x.endswith(".in_proj_a") and not x.endswith(".in_proj_b")
    ]
    if "lm_head" not in ignore:
        ignore.insert(0, "lm_head")
    elif ignore[0] != "lm_head":
        ignore = ["lm_head"] + [x for x in ignore if x != "lm_head"]
    qc["ignore"] = ignore
    if qc.get("quant_method") != "compressed-tensors":
        raise RuntimeError(qc.get("quant_method"))
    with open(os.path.join(DST, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
        fh.write("\n")
    return g1["targets"], ignore[:5], len(ignore)


def copy_sidecars():
    shutil.copy2(os.path.join(SRC, "model_mtp.safetensors"), os.path.join(DST, "model_mtp.safetensors"))
    for fn in COPY_FILES:
        s = os.path.join(SRC, fn)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(DST, fn))


def write_index(total_data: int, out_tensors):
    src_index = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
    mtp_size = os.path.getsize(os.path.join(DST, "model_mtp.safetensors"))
    wm = {name: "model.safetensors" for name, *_ in out_tensors}
    for k, v in src_index["weight_map"].items():
        if v == "model_mtp.safetensors":
            wm[k] = v
    new_index = {
        "metadata": {"total_size": total_data + mtp_size},
        "weight_map": wm,
    }
    with open(os.path.join(DST, "model.safetensors.index.json"), "w") as fh:
        json.dump(new_index, fh, indent=2)
        fh.write("\n")
    return len(wm), total_data + mtp_size


def validate(src_f) -> dict:
    dst_path = os.path.join(DST, "model.safetensors")
    src_path = os.path.join(SRC, "model.safetensors")
    results = {"reconstruct": {}, "byte_exact": {}, "in_proj_a": {}}
    with safe_open(dst_path, framework="pt") as dst, safe_open(src_path, framework="pt") as src:
        for mod in VALIDATE_MODULES:
            w = src.get_tensor(mod + ".weight")
            sc = src.get_tensor(mod + ".weight_scale")
            w_ref = w.float() * sc.float()
            packed = dst.get_tensor(mod + ".weight_packed")
            scale = dst.get_tensor(mod + ".weight_scale")
            wgs = dst.get_tensor(mod + ".weight_global_scale")
            igs = dst.get_tensor(mod + ".input_global_scale")
            hat = reconstruct(packed, scale, wgs)
            st = err_stats(w_ref, hat)
            st.update({
                "weight_global_scale": float(wgs.item()),
                "input_global_scale": float(igs.item()),
                "amax_fp8": float(w_ref.abs().max().item()),
                "packed_dtype": str(packed.dtype),
                "scale_dtype": str(scale.dtype),
                "packed_shape": list(packed.shape),
            })
            results["reconstruct"][mod] = st
            print(f"VALIDATE {mod}: {st}", flush=True)
            if st["nan"] or not st["all_finite"]:
                raise RuntimeError(f"{mod} NaN/Inf")
            if st["max_abs"] > 0.05:
                raise RuntimeError(f"{mod} max_abs {st['max_abs']} > 0.05")

        a = src.get_tensor(BYTE_EXACT_PACKED)
        b = dst.get_tensor(BYTE_EXACT_PACKED)
        results["byte_exact"]["mlp0_gate_packed"] = bool(torch.equal(a, b))
        for suffix in (".weight_scale", ".weight_global_scale", ".input_global_scale"):
            k = BYTE_EXACT_PACKED.replace(".weight_packed", suffix)
            results["byte_exact"]["mlp0_gate" + suffix] = bool(
                torch.equal(src.get_tensor(k), dst.get_tensor(k))
            )
        results["byte_exact"]["lm_head"] = bool(
            torch.equal(src.get_tensor(BYTE_EXACT_LMHEAD), dst.get_tensor(BYTE_EXACT_LMHEAD))
        )
        print("BYTE_EXACT", results["byte_exact"], flush=True)
        if not all(results["byte_exact"].values()):
            raise RuntimeError(f"byte-exact failed {results['byte_exact']}")

        w = src.get_tensor(INPROJ_A_MOD + ".weight").float()
        packed = dst.get_tensor(INPROJ_A_MOD + ".weight_packed")
        scale = dst.get_tensor(INPROJ_A_MOD + ".weight_scale")
        wgs = dst.get_tensor(INPROJ_A_MOD + ".weight_global_scale")
        hat = reconstruct(packed, scale, wgs)
        st = err_stats(w, hat)
        st["weight_global_scale"] = float(wgs.item())
        results["in_proj_a"] = st
        print("VALIDATE in_proj_a", st, flush=True)
        if st["nan"] or st["max_abs"] > 0.05:
            raise RuntimeError(f"in_proj_a reconstruct failed {st}")

        # leftover FP8 must be gone; packed must exist
        keys = set(dst.keys())
        if "model.language_model.layers.11.self_attn.k_proj.weight" in keys:
            raise RuntimeError("leftover FP8 weight still present")
        if "model.language_model.layers.11.self_attn.k_proj.weight_packed" not in keys:
            raise RuntimeError("k_proj packed missing")
        results["dst_n_tensors"] = len(keys)
    return results


def main() -> int:
    global SRC, DST, HUI
    ap = argparse.ArgumentParser(
        description="Pack leftover Unsloth FP8 Linears to compressed-tensors NVFP4."
    )
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument(
        "--src",
        default=os.environ.get("SRC_DIR", ""),
        help="lmheadfix checkpoint dir (or SRC_DIR). Example: $HOME/models/Qwen3.8-27B-NVFP4-unsloth-lmheadfix",
    )
    ap.add_argument(
        "--dst",
        default=os.environ.get("DST_DIR", ""),
        help="Output dir (or DST_DIR). Example: $HOME/models/Qwen3.8-27B-NVFP4-unsloth-w4a4attn",
    )
    ap.add_argument(
        "--huihui",
        default=os.environ.get("HUIHUI_DIR", ""),
        help="Huihui NVFP4 dir for read-only input_scale steal (or HUIHUI_DIR)",
    )
    args = ap.parse_args()
    SRC = args.src
    DST = args.dst
    HUI = args.huihui
    if not SRC or not DST or not HUI:
        ap.error("set --src, --dst, and --huihui (or SRC_DIR, DST_DIR, HUIHUI_DIR)")

    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, os.cpu_count() or 8))
    src_path = os.path.join(SRC, "model.safetensors")
    dst_path = os.path.join(DST, "model.safetensors")
    if os.path.abspath(DST) == os.path.abspath(SRC):
        raise RuntimeError("refusing to overwrite source")
    os.makedirs(DST, exist_ok=True)

    src_header, data_start = read_header(src_path)
    leftover_fp8, leftover_scale, inproj_ab = classify(src_header)
    print(
        f"classify leftover_fp8={len(leftover_fp8)} scales_drop={len(leftover_scale)} "
        f"in_proj_a/b={len(inproj_ab)}",
        flush=True,
    )
    if len(leftover_fp8) != 232:
        raise RuntimeError(f"expected 232 leftover FP8, got {len(leftover_fp8)}")
    if len(inproj_ab) != 96:
        raise RuntimeError(f"expected 96 in_proj_a/b, got {len(inproj_ab)}")

    out_tensors, n_fp8, n_ab, n_copy = build_out_list(
        src_header, leftover_fp8, leftover_scale, inproj_ab
    )
    print(f"out tensors={len(out_tensors)} packed_fp8={n_fp8} packed_ab={n_ab} copied={n_copy}", flush=True)

    src_f = safe_open(src_path, framework="pt")
    inc0 = None
    if not args.validate_only:
        print("stealing Huihui input_scale (read-only)...", flush=True)
        hui_input = steal_input_scales(leftover_fp8 | inproj_ab)
        print(f"stolen {len(hui_input)} input_scale values", flush=True)
        print("running INC-0 k_proj pack...", flush=True)
        inc0 = inc0_kproj(src_f)
        print("streaming dest model.safetensors...", flush=True)
        total_data = write_model(
            src_path, dst_path, src_header, data_start, out_tensors,
            leftover_fp8, inproj_ab, hui_input, src_f,
        )
        copy_sidecars()
        n_wm, total_size = write_index(total_data, out_tensors)
        targets, ignore_head, n_ignore = patch_config()
        print("config targets", targets, "ignore_head", ignore_head, "n_ignore", n_ignore, flush=True)
    else:
        if not os.path.isfile(dst_path):
            raise RuntimeError(f"missing dest {dst_path}")
        total_data = os.path.getsize(dst_path)
        n_wm = len(json.load(open(os.path.join(DST, "model.safetensors.index.json")))["weight_map"])
        total_size = json.load(open(os.path.join(DST, "model.safetensors.index.json")))["metadata"]["total_size"]
        cfg = json.load(open(os.path.join(DST, "config.json")))
        g1 = cfg["quantization_config"]["config_groups"]["group_1"]
        targets = g1["targets"]
        ignore_head = cfg["quantization_config"]["ignore"][:5]
        n_ignore = len(cfg["quantization_config"]["ignore"])
        print("running INC-0 k_proj pack (validate-only, in-memory)...", flush=True)
        inc0 = inc0_kproj(src_f)

    print("validating reconstruct / byte-exact...", flush=True)
    val = validate(src_f)
    report = {
        "src": SRC,
        "dst": DST,
        "inc0": inc0,
        "n_leftover_fp8": n_fp8,
        "n_inproj_ab": n_ab,
        "n_copied": n_copy,
        "n_out_tensors": len(out_tensors),
        "n_weight_map": n_wm,
        "total_data": total_data,
        "total_size": total_size,
        "dst_bytes": os.path.getsize(dst_path),
        "src_bytes": os.path.getsize(src_path),
        "config_targets": targets,
        "ignore_head": ignore_head,
        "n_ignore": n_ignore,
        "validate": val,
    }
    with open(os.path.join(DST, "REQUANT.json"), "w") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
    print("DONE", json.dumps({k: report[k] for k in ("n_leftover_fp8", "n_inproj_ab", "dst_bytes", "src_bytes")}, indent=2), flush=True)
    return 0



if __name__ == "__main__":
    sys.exit(main())
