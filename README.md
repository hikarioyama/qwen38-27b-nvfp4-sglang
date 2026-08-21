# Qwen3.8-27B NVFP4 × SGLang reproduction recipe

This tree is a reproduction kit for launching **Qwen3.8-27B NVFP4** on SGLang (`lmsysorg/sglang:dev-cu13`) and measuring single-stream decode under the same conditions.

- **Weights are not included.** Provide your own checkpoints.
- Paths, GPU UUID, and port are all environment variables. This repository does not require a host-specific absolute path.

## Layout

| Path | Contents |
|---|---|
| `overlay/` | The **7 files** of the DFlash2 overlay bind-mounted at production launch |
| `scripts/convert_unsloth_lmheadfix.py` | Unsloth FP8 `lm_head` → BF16 |
| `scripts/pack_unsloth_w4a4attn.py` | Pack remaining FP8 Linears to compressed-tensors NVFP4 |
| `scripts/convert_unsloth_modelopt.py` | `weight_packed` → ModelOpt names, reciprocal global scale |
| `recipes/*.sh.template` | docker launch templates (`docker run -d`, no `--gpus`) |
| `bench/bench_single_stream.py` | C=1 / n=3 median / TTFT / accept harness |
| `results/` | C=1 @1024 short-prompt matrix and REAL-text long-context ctx×C matrix (n=3 median, same-condition) |
| `env.example` | Example environment variables (comments only) |

## Overlay (7 files)

In-container path is `/sgl-workspace/sglang/python/sglang/<rel>`. Do not mount the whole tree.

| overlay relative path | sha256 |
|---|---|
| `srt/models/dflash.py` | `ad1e7957ce59751266c9359bfe65f51668fd81716744e34f55f7efd42e08d0f6` |
| `srt/speculative/dflash_utils.py` | `88e77363b33acfd8ac0e5b6bc899df1cfad80ce09d21c333c8afe605e1643a62` |
| `srt/speculative/dflash_worker_v2.py` | `ad83d694982de1e019443d07e31c4d6b975e0e4999a920663b78ff5d212262a5` |
| `kernels/ops/speculative/dflash.py` | `21f11619531f493bbe9f71d465912ca707f2d722991832e446ff65aecb40528c` |
| `srt/model_executor/model_runner_components/spec_aux_hidden_state.py` | `a4da95cfd059699ff1d11ed18f9338a02b553d2c81a89616c24827eab142ff56` |
| `srt/speculative/spec_utils.py` | `747d8de9ef7f80a30c04550b83a90387d2351d461654c9dc1f078d492bbf0ee6` |
| `srt/mem_cache/kv_cache_configurator.py` | `10004c14282291bd42bb967d23d01deffca0e17be9a5651eaf5e53e27d2da30b` |

`dflash_info.py` is not one of the production 7 files, so it is not bundled.

## Convert

Steps to build the derived checkpoint used at launch from the official Unsloth NVFP4. Does not use a GPU.

```bash
# 1. lm_head FP8 → BF16
python3 scripts/convert_unsloth_lmheadfix.py \
  --src "$SRC_DIR" \
  --dst "$LMHEADFIX_DIR"

# 2. remaining attention FP8 → NVFP4 (compressed-tensors)
python3 scripts/pack_unsloth_w4a4attn.py \
  --src "$LMHEADFIX_DIR" \
  --dst "$W4A4_DIR" \
  --huihui "$HUIHUI_DIR"

# 3. compressed-tensors names → ModelOpt names (no repack; reciprocal global scale)
#    Unsloth:  W = e2m1(weight_packed) * fp8(weight_scale) / weight_global_scale
#    ModelOpt: W = e2m1(weight)        * fp8(weight_scale) * weight_scale_2
python3 scripts/convert_unsloth_modelopt.py \
  --src "$W4A4_DIR" \
  --dst "$MODELOPT_DIR"
```

Step 3 is rename + `weight_scale_2 = 1/weight_global_scale`, `input_scale = 1/input_global_scale` only. Copy `lm_head` / norm / MTP. `hf_quant_config.json` is NVFP4 / `group_size` 16 / `exclude_modules` includes `lm_head`. The `unsloth-dflash2` recipe `MODEL_DIR` is this ModelOpt checkpoint.

## Launch

Pin with `NVIDIA_VISIBLE_DEVICES=$GPU_UUID`. Do not use `--gpus`. `docker run -d`.

```bash
set -a
# source a filled-in copy of env.example
set +a
bash recipes/huihui-dflash2.sh.template
# or
bash recipes/unsloth-dflash2.sh.template
bash recipes/unsloth-nextn.sh.template
```

| Recipe | Quantization | spec | overlay | `--max-running-requests` |
|---|---|---|---|---|
| `huihui-dflash2` | `modelopt_fp4` | DFLASH K=8 + ReplaySSM, overlap off | 7 files | omit |
| `unsloth-dflash2` | `modelopt_fp4` | same as above | 7 files | omit |
| `unsloth-nextn` | (no flag / checkpoint-dependent) | NEXTN steps=3 topk=1 draft=4 | none | default 64 |

Required: `MODEL_DIR`, `GPU_UUID`. DFlash recipes also need `DRAFT_DIR`. Optional: `PORT`, `HOST`, `OVERLAY_DIR`, `FLASHINFER_CACHE`, `IMAGE`.

## Bench conditions (fixed)

- C=1, `max_tokens=1024`, T=0, seed=0, thinking off
- `POST /v1/chat/completions`
- discard 1 warmup, **n=3 median, same-condition**
- token count = `usage.completion_tokens` (do not count SSE chunks)
- accept = `choices[0].meta_info.spec_accept_length` (`return_meta_info=true`)
- TTFT = first content delta of a separate stream, n=3

```bash
python3 bench/bench_single_stream.py \
  --base "http://${HOST:-127.0.0.1}:${PORT:-8040}" \
  --recipe huihui-dflash2 \
  --prompt LRU
```

Prompts: `LRU` / `JA-code` / `EN-prose` / `JA-prose`

Output line:

```
<recipe> chat <prompt> thinking-off @1024 : <tok/s> tok/s (n=3, same-condition) | TTFT <s> | accept <len>
```

## Results table

Conditions: C=1, `max_tokens=1024`, thinking off, T=0, seed=0, n=3 median, same-condition. Blank TTFT (—) was not measured in this run.

### tok/s (n=3 median, same-condition)

| recipe \ prompt | LRU | JA-code | EN-prose | JA-prose |
|---|---|---|---|---|
| huihui-dflash2 | 270.08 | 241.05 | 135.43 | 91.41 |
| unsloth-dflash2 | 265.75 | 242.71 | 136.11 | 95.08 |
| unsloth-nextn | 120.32 | 114.87 | 88.37 | 79.89 |

### TTFT (seconds, n=3 median)

| recipe \ prompt | LRU | JA-code | EN-prose | JA-prose |
|---|---|---|---|---|
| huihui-dflash2 | 0.026 | — | — | — |
| unsloth-dflash2 | — | — | — | — |
| unsloth-nextn | — | — | — | — |

### accept length (n=3 median)

| recipe \ prompt | LRU | JA-code | EN-prose | JA-prose |
|---|---|---|---|---|
| huihui-dflash2 | 5.278 | 4.719 | 2.672 | 1.795 |
| unsloth-dflash2 | 5.198 | 4.741 | 2.661 | 1.850 |
| unsloth-nextn | 3.507 | 3.346 | 2.583 | 2.339 |


## Long-context matrix (REAL text)

Same Unsloth NVFP4 DFlash2 (K=8, ReplaySSM, KV FP8) serve, **input length × concurrency**, on repeated natural English technical prose (last paragraph = EN-code LRU). Filler is not pad tokens.

Tables: [`results/long-context.md`](results/long-context.md). JSON: [`results/long-context.json`](results/long-context.json).

A prior pad-token filler matrix is **invalid** and is **not published**. The short-prompt C=1 table above is unchanged.

## Notes

- This kit does not start or stop a server. GPU job ownership stays with the caller.
- Do not write internal hostnames or machine-specific absolute paths as facts.
- Formatters have not been applied.
