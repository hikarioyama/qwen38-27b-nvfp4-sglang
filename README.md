# Qwen3.8-27B NVFP4 × SGLang 再現レシピ

このツリーは **Qwen3.8-27B NVFP4** を SGLang (`lmsysorg/sglang:dev-cu13`) で起動し、単一ストリーム decode を同じ条件で測るための再現キットです。

- **重みは含みません。** チェックポイントは自分で用意してください。
- パス・GPU UUID・ポートはすべて環境変数です。このリポジトリは特定ホストの絶対パスを要求しません。

## レイアウト

| パス | 内容 |
|---|---|
| `overlay/` | 本番起動で bind-mount している DFlash2 overlay **7 ファイル** |
| `scripts/convert_unsloth_lmheadfix.py` | Unsloth FP8 `lm_head` → BF16 |
| `scripts/pack_unsloth_w4a4attn.py` | 残り FP8 Linear を compressed-tensors NVFP4 に pack |
| `scripts/convert_unsloth_modelopt.py` | `weight_packed` → ModelOpt 名、global scale を reciprocal |
| `recipes/*.sh.template` | docker 起動テンプレート（`docker run -d`、`--gpus` なし） |
| `bench/bench_single_stream.py` | C=1 / n=3 median / TTFT / accept harness |
| `results/` | C=1 @1024 実測マトリクス（n=3 median, same-condition） |
| `env.example` | 環境変数の例（コメントのみ） |

## Overlay（7 ファイル）

コンテナ内パスは `/sgl-workspace/sglang/python/sglang/<rel>`。ツリー全体をマウントしないこと。

| overlay 相対パス | sha256 |
|---|---|
| `srt/models/dflash.py` | `ad1e7957ce59751266c9359bfe65f51668fd81716744e34f55f7efd42e08d0f6` |
| `srt/speculative/dflash_utils.py` | `88e77363b33acfd8ac0e5b6bc899df1cfad80ce09d21c333c8afe605e1643a62` |
| `srt/speculative/dflash_worker_v2.py` | `ad83d694982de1e019443d07e31c4d6b975e0e4999a920663b78ff5d212262a5` |
| `kernels/ops/speculative/dflash.py` | `21f11619531f493bbe9f71d465912ca707f2d722991832e446ff65aecb40528c` |
| `srt/model_executor/model_runner_components/spec_aux_hidden_state.py` | `a4da95cfd059699ff1d11ed18f9338a02b553d2c81a89616c24827eab142ff56` |
| `srt/speculative/spec_utils.py` | `747d8de9ef7f80a30c04550b83a90387d2351d461654c9dc1f078d492bbf0ee6` |
| `srt/mem_cache/kv_cache_configurator.py` | `10004c14282291bd42bb967d23d01deffca0e17be9a5651eaf5e53e27d2da30b` |

`dflash_info.py` は本番 7 ファイルに含まれないため未同梱。

## 変換

Unsloth 公式 NVFP4 から、起動に使う派生チェックポイントを作る手順。GPU は使いません。

```bash
# 1. lm_head FP8 → BF16
python3 scripts/convert_unsloth_lmheadfix.py \
  --src "$SRC_DIR" \
  --dst "$LMHEADFIX_DIR"

# 2. 残り attention FP8 → NVFP4 (compressed-tensors)
python3 scripts/pack_unsloth_w4a4attn.py \
  --src "$LMHEADFIX_DIR" \
  --dst "$W4A4_DIR" \
  --huihui "$HUIHUI_DIR"

# 3. compressed-tensors 名 → ModelOpt 名（repack なし。global scale を reciprocal）
#    Unsloth:  W = e2m1(weight_packed) * fp8(weight_scale) / weight_global_scale
#    ModelOpt: W = e2m1(weight)        * fp8(weight_scale) * weight_scale_2
python3 scripts/convert_unsloth_modelopt.py \
  --src "$W4A4_DIR" \
  --dst "$MODELOPT_DIR"
```

ステップ 3 は rename + `weight_scale_2 = 1/weight_global_scale`、`input_scale = 1/input_global_scale` だけ。`lm_head` / norm / MTP はコピー。`hf_quant_config.json` は NVFP4 / `group_size` 16 / `exclude_modules` に `lm_head`。`unsloth-dflash2` レシピの `MODEL_DIR` はこの ModelOpt チェックポイント。

## 起動

`NVIDIA_VISIBLE_DEVICES=$GPU_UUID` でピンする。`--gpus` は使わない。`docker run -d`。

```bash
set -a
# source env.example を埋めたファイル
set +a
bash recipes/huihui-dflash2.sh.template
# または
bash recipes/unsloth-dflash2.sh.template
bash recipes/unsloth-nextn.sh.template
```

| レシピ | 量子化 | spec | overlay | `--max-running-requests` |
|---|---|---|---|---|
| `huihui-dflash2` | `modelopt_fp4` | DFLASH K=8 + ReplaySSM、overlap off | 7 files | 付けない |
| `unsloth-dflash2` | `modelopt_fp4` | 同上 | 7 files | 付けない |
| `unsloth-nextn` | （フラグなし / チェックポイント依存） | NEXTN steps=3 topk=1 draft=4 | なし | 既定 64 |

必須: `MODEL_DIR`, `GPU_UUID`。DFlash 系はさらに `DRAFT_DIR`。任意: `PORT`, `HOST`, `OVERLAY_DIR`, `FLASHINFER_CACHE`, `IMAGE`。

## Bench 条件（固定）

- C=1, `max_tokens=1024`, T=0, seed=0, thinking off
- `POST /v1/chat/completions`
- warmup 1 本破棄、**n=3 median, same-condition**
- トークン数 = `usage.completion_tokens`（SSE chunk は数えない）
- accept = `choices[0].meta_info.spec_accept_length`（`return_meta_info=true`）
- TTFT = 別ストリームの first content delta、n=3

```bash
python3 bench/bench_single_stream.py \
  --base "http://${HOST:-127.0.0.1}:${PORT:-8040}" \
  --recipe huihui-dflash2 \
  --prompt LRU
```

プロンプト: `LRU` / `JA-code` / `EN-prose` / `JA-prose`

出力行:

```
<recipe> chat <prompt> thinking-off @1024 : <tok/s> tok/s (n=3, same-condition) | TTFT <s> | accept <len>
```

## 結果テーブル

条件: C=1, `max_tokens=1024`, thinking off, T=0, seed=0, n=3 median, same-condition。空欄の TTFT（—）はこのランでは未測定。

### tok/s（n=3 median, same-condition）

| recipe \ prompt | LRU | JA-code | EN-prose | JA-prose |
|---|---|---|---|---|
| huihui-dflash2 | 270.08 | 241.05 | 135.43 | 91.41 |
| unsloth-dflash2 | 265.75 | 242.71 | 136.11 | 95.08 |
| unsloth-nextn | 120.32 | 114.87 | 88.37 | 79.89 |

### TTFT（秒, n=3 median）

| recipe \ prompt | LRU | JA-code | EN-prose | JA-prose |
|---|---|---|---|---|
| huihui-dflash2 | 0.026 | — | — | — |
| unsloth-dflash2 | — | — | — | — |
| unsloth-nextn | — | — | — | — |

### accept length（n=3 median）

| recipe \ prompt | LRU | JA-code | EN-prose | JA-prose |
|---|---|---|---|---|
| huihui-dflash2 | 5.278 | 4.719 | 2.672 | 1.795 |
| unsloth-dflash2 | 5.198 | 4.741 | 2.661 | 1.850 |
| unsloth-nextn | 3.507 | 3.346 | 2.583 | 2.339 |

## 注意

- このキットはサーバを起動・停止しない。GPU ジョブの所有権は呼び出し側。
- 内部ホスト名や特定マシンの絶対パスを事実として書かない。
- フォーマッタはかけていない。
