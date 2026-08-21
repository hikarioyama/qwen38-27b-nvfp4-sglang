#!/usr/bin/env python3
"""REAL-text long-context ctx × concurrency bench for a served SGLang endpoint.

Does not start or stop a server. Does not touch GPUs.

Filler is repeated natural English technical prose. The last paragraph is the
EN-code LRU task. Filler is not pad tokens and not the word "padding".

Protocol:
  thinking off, T=0, seed=0, /v1/chat/completions
  warmup 1 discarded, then median of 3 timed batches (same-condition)
  C concurrent requests launched together
  tok count = usage.completion_tokens (never SSE chunks)
  TTFT = stream first content delta (no return_meta_info on the stream)
  decode tok/s after first token = (completion_tokens - 1) / (wall - TTFT)
  accept = spec_accept_length from a separate non-stream return_meta_info=true
  C>1: median across streams in the batch, then median across timed batches

Requires a local tokenizer dir (transformers AutoTokenizer). Point it at the
served checkpoint (or any matching tokenizer). Paths are flags / env, not baked in.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

BASE = os.environ.get("BENCH_BASE", "http://127.0.0.1:8040")
MODEL = os.environ.get("BENCH_MODEL", "/model")
MAX_NEW = 1024
REQ_TIMEOUT = 1800.0
CS = [1, 2, 4, 8]
TARGETS = [8192, 16384, 31744]
OUT_JSON = Path("long-context-run.json")
OUT_MD = Path("long-context-run.md")

EN_TASK = (
    "Write a complete, production-quality Python LRU cache with per-key TTL "
    "and a pytest suite."
)
REAL_UNIT = (
    "A least-recently-used cache in Python keeps a hash map from keys to nodes "
    "and a doubly linked list that records recency order. A successful get moves "
    "that node to the most-recent end so later eviction still removes the oldest "
    "entry. When the bounded capacity is exceeded the least-recent node is unlinked "
    "and its key is deleted from the map. Per-key time-to-live stores an absolute "
    "expiry on each node; after expiry a lookup must miss even if the key has not "
    "yet been evicted. OrderedDict can encode recency by moving keys to the end on "
    "access. Thread safety is optional and must be documented. Tests should cover "
    "hit, miss, eviction order, overwrite, a capacity of one, and expiry on both "
    "get and set. "
)
REAL_WORDS = [
    " cache",
    " recency",
    " eviction",
    " dictionary",
    " mapping",
    " expiry",
    " lookup",
    " capacity",
    " node",
    " key",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def classify(err: str) -> str:
    s = err.lower()
    if "timed out" in s or "timeout" in s:
        return "timeout"
    if "out of memory" in s or "oom" in s or "cuda out of memory" in s:
        return "OOM"
    if "kv cache is full" in s or "kvcache is full" in s or "prefix cache is full" in s:
        return "KV full"
    if "no space" in s and "kv" in s:
        return "KV full"
    if "503" in s or "queue" in s or "not admitted" in s or "too many requests" in s:
        return "queue not admitted"
    if "500" in s or "internal server error" in s or "internal_server_error" in s:
        return "500"
    return err.strip().replace("\n", " ")[:240]


def http_json(url: str, body: dict | None = None, timeout: float = 30.0) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        raw = ""
        try:
            raw = e.read().decode("utf-8", "replace")[:4000]
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code}: {raw or e}") from None
    except TimeoutError as e:
        raise RuntimeError(f"timeout: {e}") from None
    except urllib.error.URLError as e:
        reason = str(getattr(e, "reason", e))
        if "timed out" in reason.lower() or "timeout" in reason.lower():
            raise RuntimeError(f"timeout: {reason}") from None
        raise RuntimeError(f"URLError: {reason}") from None


def templated_count(tok, user: str) -> int:
    s = tok.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return len(tok.encode(s, add_special_tokens=False))


def make_real(tok, target: int) -> tuple[str, int]:
    prefix = "Background for a production cache library. "
    suffix = "\n" + EN_TASK
    blob = prefix + REAL_UNIT + suffix
    if "padding" in blob.lower() or "padding" in " ".join(REAL_WORDS).lower():
        raise RuntimeError("REAL prompt contains the word padding")

    def count_of(n: int, extra: str) -> tuple[str, int]:
        user = prefix + (REAL_UNIT * n) + extra + suffix
        return user, templated_count(tok, user)

    base = templated_count(tok, prefix + suffix)
    one = templated_count(tok, prefix + REAL_UNIT + suffix)
    unit_toks = max(1, one - base)
    lo, hi = 0, max(0, (target - base) // unit_toks + 8)
    best_n = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        _, c = count_of(mid, "")
        if c <= target:
            best_n = mid
            lo = mid + 1
        else:
            hi = mid - 1
    extra = ""
    user, c = count_of(best_n, extra)
    wi = 0
    guard = 0
    while c < target and guard < 4096:
        trial = extra + REAL_WORDS[wi % len(REAL_WORDS)]
        tuser, tc = count_of(best_n, trial)
        if tc > target:
            break
        extra = trial
        user, c = tuser, tc
        wi += 1
        guard += 1
    if c != target:
        raise RuntimeError(
            f"REAL could not hit exact {target}, last={c} n={best_n} extra={extra!r}"
        )
    if "padding" in user.lower():
        raise RuntimeError("REAL user contains the word padding")
    if not user.rstrip().endswith(EN_TASK):
        raise RuntimeError("REAL last paragraph is not the EN-code LRU task")
    return user, c


def chat_body(prompt: str, *, stream: bool, return_meta: bool) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_NEW,
        "temperature": 0,
        "seed": 0,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    if return_meta:
        body["return_meta_info"] = True
    return body


def stream_one(prompt: str) -> dict:
    data = json.dumps(chat_body(prompt, stream=True, return_meta=False)).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    t0 = time.perf_counter()
    first = None
    usage = None
    finish = None
    try:
        with urllib.request.urlopen(req, timeout=REQ_TIMEOUT) as resp:
            while True:
                raw = resp.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                choice = (obj.get("choices") or [{}])[0]
                if choice.get("finish_reason"):
                    finish = choice.get("finish_reason")
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content and first is None:
                    first = time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        raw = ""
        try:
            raw = e.read().decode("utf-8", "replace")[:4000]
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code}: {raw or e}") from None
    except TimeoutError as e:
        raise RuntimeError(f"timeout: {e}") from None
    except urllib.error.URLError as e:
        reason = str(getattr(e, "reason", e))
        if "timed out" in reason.lower() or "timeout" in reason.lower():
            raise RuntimeError(f"timeout: {reason}") from None
        raise RuntimeError(f"URLError: {reason}") from None
    wall = time.perf_counter() - t0
    if first is None:
        raise RuntimeError("no content delta (TTFT missing)")
    if not usage or usage.get("completion_tokens") is None:
        raise RuntimeError("missing usage.completion_tokens on stream")
    toks = int(usage["completion_tokens"])
    decode_s = wall - first
    if toks < 2 or decode_s <= 0:
        raise RuntimeError(f"cannot compute decode tok/s toks={toks} decode_s={decode_s}")
    return {
        "ok": True,
        "ttft_s": first,
        "wall_s": wall,
        "decode_s": decode_s,
        "completion_tokens": toks,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "decode_tok_s": (toks - 1) / decode_s,
        "finish_reason": finish,
        "reasoning_tokens": (
            (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
            if isinstance(usage.get("completion_tokens_details"), dict)
            else usage.get("reasoning_tokens")
        ),
    }


def accept_one(prompt: str) -> dict:
    t0 = time.perf_counter()
    r = http_json(
        BASE + "/v1/chat/completions",
        body=chat_body(prompt, stream=False, return_meta=True),
        timeout=REQ_TIMEOUT,
    )
    dt = time.perf_counter() - t0
    usage = r.get("usage") or {}
    toks = usage.get("completion_tokens")
    if toks is None:
        raise RuntimeError("missing usage.completion_tokens")
    choice = (r.get("choices") or [{}])[0]
    meta = choice.get("meta_info") or r.get("meta_info") or {}
    acc = meta.get("spec_accept_length")
    if acc is None:
        raise RuntimeError("missing spec_accept_length")
    return {
        "ok": True,
        "completion_tokens": int(toks),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "seconds": dt,
        "finish_reason": choice.get("finish_reason"),
        "spec_accept_length": float(acc),
        "spec_accept_rate": meta.get("spec_accept_rate"),
        "spec_verify_ct": meta.get("spec_verify_ct"),
    }


def run_concurrent(fn, prompt: str, c: int) -> dict:
    barrier = threading.Barrier(c)

    def worker(_i: int) -> dict:
        try:
            barrier.wait(timeout=30)
            return fn(prompt)
        except Exception as e:
            return {"ok": False, "error": classify(str(e)), "raw": str(e)[:500]}

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:
        futs = [ex.submit(worker, i) for i in range(c)]
        rows = [f.result() for f in futs]
    wall = time.perf_counter() - t0
    fails = [r for r in rows if not r.get("ok")]
    if fails:
        reasons = sorted({str(r.get("error") or "unknown") for r in fails})
        return {
            "ok": False,
            "skip_reason": reasons[0] if len(reasons) == 1 else "; ".join(reasons),
            "wall_s": wall,
            "n_ok": c - len(fails),
            "n_fail": len(fails),
            "requests": rows,
        }
    return {"ok": True, "wall_s": wall, "requests": rows}


def summarize_stream_batch(batch: dict, c: int) -> dict:
    if not batch.get("ok"):
        return batch
    rows = batch["requests"]
    ttfts = [float(r["ttft_s"]) for r in rows]
    decodes = [float(r["decode_tok_s"]) for r in rows]
    toks = [int(r["completion_tokens"]) for r in rows]
    pts = [int(r["prompt_tokens"]) for r in rows]
    return {
        "ok": True,
        "wall_s": batch["wall_s"],
        "ttft_s": statistics.median(ttfts),
        "decode_tok_s": statistics.median(decodes),
        "ttft_s_streams": ttfts,
        "decode_tok_s_streams": decodes,
        "completion_tokens": toks,
        "prompt_tokens": pts,
        "finish_reasons": [r.get("finish_reason") for r in rows],
        "requests": rows,
    }


def summarize_accept_batch(batch: dict, c: int) -> dict:
    if not batch.get("ok"):
        return batch
    rows = batch["requests"]
    accs = [float(r["spec_accept_length"]) for r in rows]
    return {
        "ok": True,
        "wall_s": batch["wall_s"],
        "accept": statistics.median(accs),
        "accept_streams": accs,
        "completion_tokens": [int(r["completion_tokens"]) for r in rows],
        "prompt_tokens": [int(r["prompt_tokens"]) for r in rows],
        "finish_reasons": [r.get("finish_reason") for r in rows],
        "spec_accept_rate": [r.get("spec_accept_rate") for r in rows],
        "requests": rows,
    }


def spread_frac(xs: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    med = statistics.median(xs)
    if med == 0:
        return None
    return (max(xs) - min(xs)) / abs(med)


def messy_note(xs: list[float], label: str) -> str | None:
    sp = spread_frac(xs)
    if sp is None or sp <= 0.20:
        return None
    raw = " / ".join(f"{x:.4g}" for x in xs)
    return f"messy {label} spread {sp*100:.1f}% raw {raw}"


def fmt_cell(cell: dict) -> str:
    if cell.get("skip_reason"):
        return f"skip: {cell['skip_reason']}"
    ttft = cell.get("median_ttft_s")
    dec = cell.get("median_decode_tok_s")
    acc = cell.get("median_accept")
    if ttft is None or dec is None:
        return "skip: no median"
    acc_s = f"{acc:.3f}" if acc is not None else "NA"
    bits = [f"TTFT {ttft:.3f}s", f"decode {dec:.2f} tok/s", f"accept {acc_s}"]
    notes = cell.get("messy") or []
    return " / ".join(bits[:3]) + ((" (" + "; ".join(notes) + ")") if notes else "")


def write_md(doc: dict) -> str:
    cells = {(c["target_label"], c["C"]): c for c in doc["cells"]}
    rows_ctx = doc["row_labels"]
    lines = [
        "# REAL-text long-context ctx × C",
        "",
        "Does not start or stop a server. Filler is repeated natural English "
        "technical prose about Python LRU caches / TTL / eviction. The word "
        "`padding` does not appear. Last paragraph is: *Write a complete, "
        "production-quality Python LRU cache with per-key TTL and a pytest suite.*",
        "",
        "Same-condition: chat completions, thinking off, T=0, seed=0, "
        f"max_new_tokens={doc.get('max_new_tokens')}. "
        "Launch C requests at the same time, wait for all. "
        "TTFT = stream first *content* delta (no `return_meta_info` on the stream). "
        "Decode tok/s after first token = (`usage.completion_tokens` − 1) / (wall − TTFT), "
        "per stream; for C>1 median across streams in the batch, then median across "
        "timed batches. Token count = `usage.completion_tokens` from "
        "`stream_options.include_usage` (never SSE chunk counts). "
        "accept = `spec_accept_length` from a separate non-stream call with "
        "`return_meta_info=true`; for C>1 median across streams in the batch, then "
        "median across timed batches. median of 3 timed batches, 1 warmup discarded.",
        "",
        "Prompts are filled to exact tokenizer length. A target that cannot fit "
        "`input + max_new` in the server context window is clamped.",
        "",
        f"Server: context_length={doc['server']['context_length']}, "
        f"kv_cache_dtype={doc['server']['kv_cache_dtype']}, "
        f"quantization={doc['server']['quantization']}, "
        f"speculative_algorithm={doc['server']['speculative_algorithm']}, "
        f"speculative_num_draft_tokens={doc['server']['speculative_num_draft_tokens']}, "
        f"enable_linear_replayssm_spec={doc['server']['enable_linear_replayssm_spec']}, "
        f"disable_overlap_schedule={doc['server'].get('disable_overlap_schedule')}, "
        f"max_running_requests={doc['server']['max_running_requests']}, "
        f"max_total_num_tokens={doc['server']['max_total_num_tokens']}.",
        "",
        f"JSON: `{OUT_JSON}`",
        "",
        "## Table",
        "",
        "| context | " + " | ".join(f"C={c}" for c in CS) + " |",
        "|" + "|".join(["---"] * (1 + len(CS))) + "|",
    ]
    for ctx in rows_ctx:
        cols = []
        for c in CS:
            cell = cells.get((ctx, c))
            cols.append(fmt_cell(cell) if cell else "pending")
        lines.append("| " + " | ".join([str(ctx)] + cols) + " |")
    lines += ["", "## Notes", ""]
    skipped = [c for c in doc["cells"] if c.get("skip_reason")]
    if skipped:
        for c in skipped:
            lines.append(f"- SKIP ctx={c['target_label']} C={c['C']}: {c['skip_reason']}")
    else:
        lines.append("- No cells skipped (no OOM / 500 / queue / KV-full / timeout).")
    for c in doc["cells"]:
        if c.get("messy"):
            raw_d = c.get("raw_decode_tok_s")
            raw_t = c.get("raw_ttft_s")
            raw_a = c.get("raw_accept")
            lines.append(
                f"- ctx={c['target_label']} C={c['C']} messy: {'; '.join(c['messy'])} "
                f"| raw decode {raw_d} | raw TTFT {raw_t} | raw accept {raw_a}"
            )
        pts = None
        for batch in c.get("timed_stream") or []:
            if batch.get("prompt_tokens"):
                pts = batch["prompt_tokens"]
                break
        if pts:
            lines.append(
                f"- ctx={c['target_label']} C={c['C']} server prompt_tokens={pts} "
                f"(tokenizer {c.get('tokenizer_tokens')})"
            )
    lines.append("")
    return "\n".join(lines) + "\n"


def save(doc: dict) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(write_md(doc), encoding="utf-8")


def slim_batch(batch: dict) -> dict:
    out = {k: v for k, v in batch.items() if k != "requests"}
    reqs = []
    for r in batch.get("requests") or []:
        reqs.append({k: v for k, v in r.items() if k != "raw" or not r.get("ok")})
    out["requests"] = reqs
    return out


def _csv_ints(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise argparse.ArgumentTypeError("need at least one integer")
    return out


def main() -> int:
    global BASE, MODEL, MAX_NEW, REQ_TIMEOUT, CS, TARGETS, OUT_JSON, OUT_MD
    ap = argparse.ArgumentParser(
        description="REAL-text long-context ctx × C bench (does not start/stop a server)"
    )
    ap.add_argument("--base", default=os.environ.get("BENCH_BASE", "http://127.0.0.1:8040"))
    ap.add_argument("--model", default=os.environ.get("BENCH_MODEL", "/model"))
    ap.add_argument(
        "--tokenizer",
        default=os.environ.get("TOKENIZER_DIR", ""),
        help="local tokenizer / checkpoint dir (or TOKENIZER_DIR). Required.",
    )
    ap.add_argument(
        "--out-json",
        default=os.environ.get("BENCH_OUT_JSON", "long-context-run.json"),
    )
    ap.add_argument(
        "--out-md",
        default=os.environ.get("BENCH_OUT_MD", "long-context-run.md"),
    )
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--targets", type=_csv_ints, default="8192,16384,31744")
    ap.add_argument("--concurrency", type=_csv_ints, default="1,2,4,8")
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args()
    if not args.tokenizer:
        ap.error("set --tokenizer or TOKENIZER_DIR to a local checkpoint/tokenizer dir")

    BASE = args.base.rstrip("/")
    MODEL = args.model
    MAX_NEW = args.max_new
    REQ_TIMEOUT = args.timeout
    TARGETS = args.targets if isinstance(args.targets, list) else _csv_ints(args.targets)
    CS = args.concurrency if isinstance(args.concurrency, list) else _csv_ints(args.concurrency)
    OUT_JSON = Path(args.out_json)
    OUT_MD = Path(args.out_md)

    info = http_json(BASE + "/get_server_info", timeout=15)
    ctx_len = int(info["context_length"])
    server = {
        "context_length": ctx_len,
        "kv_cache_dtype": info.get("kv_cache_dtype"),
        "quantization": info.get("quantization"),
        "speculative_algorithm": info.get("speculative_algorithm"),
        "speculative_num_draft_tokens": info.get("speculative_num_draft_tokens"),
        "enable_linear_replayssm_spec": info.get("enable_linear_replayssm_spec"),
        "max_running_requests": info.get("max_running_requests"),
        "max_total_num_tokens": info.get("max_total_num_tokens"),
        "max_req_input_len": info.get("max_req_input_len"),
        "max_prefill_tokens": info.get("max_prefill_tokens"),
        "chunked_prefill_size": info.get("chunked_prefill_size"),
        "model_path": info.get("model_path"),
        "disable_overlap_schedule": info.get("disable_overlap_schedule"),
    }
    log(
        f"server context_length={ctx_len} kv={server['kv_cache_dtype']} "
        f"spec={server['speculative_algorithm']} K={server['speculative_num_draft_tokens']}"
    )
    if ctx_len < max(TARGETS) + MAX_NEW:
        log("WARN context_length cannot fit the largest target + max_new; will clamp")

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    tok.model_max_length = 10**9

    prompts: dict[int, dict] = {}
    row_labels: list[int] = []
    skip_contexts: dict[int, str] = {}
    for target in TARGETS:
        actual = target
        clamped = False
        if ctx_len - MAX_NEW < target:
            actual = ctx_len - MAX_NEW
            clamped = True
        if actual < 1:
            skip_contexts[target] = (
                f"even C=1 cannot fit (context_length={ctx_len}, max_new={MAX_NEW})"
            )
            log(f"SKIP ctx {target}: {skip_contexts[target]}")
            continue
        log(f"REAL prompt target={target} actual={actual} clamped={clamped}")
        user, n = make_real(tok, actual)
        prompts[actual] = {
            "user": user,
            "tokenizer_tokens": n,
            "requested_target": target,
            "clamped": clamped,
            "label": actual,
            "chars": len(user),
        }
        row_labels.append(actual)
        log(f"  tokenizer_tokens={n} chars={len(user)}")

    doc = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base": BASE,
        "model": MODEL,
        "tokenizer": args.tokenizer,
        "max_new_tokens": MAX_NEW,
        "temperature": 0,
        "seed": 0,
        "thinking": "off",
        "filler": "REAL English LRU prose (no padding unit)",
        "kv_cache_dtype": server["kv_cache_dtype"],
        "server": server,
        "same_condition": True,
        "timing": "median of 3 timed batches, 1 warmup discarded",
        "metric": (
            "TTFT=stream first content delta (no return_meta_info); "
            "decode tok/s=(usage.completion_tokens-1)/(wall-TTFT) per stream; "
            "accept=spec_accept_length from separate non-stream return_meta_info=true; "
            "C>1: median across streams in the batch, then median across timed batches"
        ),
        "row_labels": row_labels,
        "skip_contexts": skip_contexts,
        "prompt_meta": {
            str(k): {kk: vv for kk, vv in v.items() if kk != "user"}
            for k, v in prompts.items()
        },
        "cells": [],
        "aborted": None,
    }
    save(doc)

    for actual in row_labels:
        meta = prompts[actual]
        user = meta["user"]
        for c in CS:
            log(f"=== ctx={actual} C={c} warmup stream ===")
            cell = {
                "target_label": actual,
                "requested_target": meta["requested_target"],
                "clamped": meta["clamped"],
                "tokenizer_tokens": meta["tokenizer_tokens"],
                "C": c,
                "skip_reason": None,
                "warmup_stream": None,
                "warmup_accept": None,
                "timed_stream": [],
                "timed_accept": [],
                "median_ttft_s": None,
                "median_decode_tok_s": None,
                "median_accept": None,
            }
            try:
                warm_s = summarize_stream_batch(run_concurrent(stream_one, user, c), c)
            except Exception as e:
                cell["skip_reason"] = classify(str(e))
                cell["warmup_stream"] = {"ok": False, "error": cell["skip_reason"]}
                doc["cells"].append(cell)
                save(doc)
                log(f"SKIP ctx={actual} C={c}: {cell['skip_reason']}")
                continue
            cell["warmup_stream"] = slim_batch(warm_s)
            if not warm_s.get("ok"):
                cell["skip_reason"] = warm_s.get("skip_reason") or "warmup stream failed"
                doc["cells"].append(cell)
                save(doc)
                log(f"SKIP ctx={actual} C={c}: {cell['skip_reason']}")
                continue
            pts = warm_s.get("prompt_tokens") or []
            if pts and any(p != actual for p in pts):
                log(f"WARN prompt_tokens {pts} != tokenizer {actual}")
            log(
                f"  warmup stream TTFT={warm_s['ttft_s']:.3f}s decode={warm_s['decode_tok_s']:.2f} "
                f"toks={warm_s['completion_tokens']} wall={warm_s['wall_s']:.3f}s pts={pts}"
            )

            log(f"=== ctx={actual} C={c} warmup accept ===")
            try:
                warm_a = summarize_accept_batch(run_concurrent(accept_one, user, c), c)
            except Exception as e:
                cell["skip_reason"] = classify(str(e))
                cell["warmup_accept"] = {"ok": False, "error": cell["skip_reason"]}
                doc["cells"].append(cell)
                save(doc)
                log(f"SKIP ctx={actual} C={c}: {cell['skip_reason']}")
                continue
            cell["warmup_accept"] = slim_batch(warm_a)
            if not warm_a.get("ok"):
                cell["skip_reason"] = warm_a.get("skip_reason") or "warmup accept failed"
                doc["cells"].append(cell)
                save(doc)
                log(f"SKIP ctx={actual} C={c}: {cell['skip_reason']}")
                continue
            log(f"  warmup accept={warm_a['accept']:.3f} wall={warm_a['wall_s']:.3f}s")

            timed_s = []
            timed_a = []
            skipped = None
            for i in range(3):
                log(f"=== ctx={actual} C={c} timed stream {i+1}/3 ===")
                try:
                    rec_s = summarize_stream_batch(run_concurrent(stream_one, user, c), c)
                except Exception as e:
                    skipped = classify(str(e))
                    break
                timed_s.append(slim_batch(rec_s))
                if not rec_s.get("ok"):
                    skipped = rec_s.get("skip_reason") or "timed stream failed"
                    break
                log(
                    f"  TTFT={rec_s['ttft_s']:.3f}s decode={rec_s['decode_tok_s']:.2f} "
                    f"toks={rec_s['completion_tokens']} wall={rec_s['wall_s']:.3f}s"
                )
                log(f"=== ctx={actual} C={c} timed accept {i+1}/3 ===")
                try:
                    rec_a = summarize_accept_batch(run_concurrent(accept_one, user, c), c)
                except Exception as e:
                    skipped = classify(str(e))
                    break
                timed_a.append(slim_batch(rec_a))
                if not rec_a.get("ok"):
                    skipped = rec_a.get("skip_reason") or "timed accept failed"
                    break
                log(f"  accept={rec_a['accept']:.3f} wall={rec_a['wall_s']:.3f}s")

            cell["timed_stream"] = timed_s
            cell["timed_accept"] = timed_a
            if skipped:
                cell["skip_reason"] = skipped
                doc["cells"].append(cell)
                save(doc)
                log(f"SKIP ctx={actual} C={c}: {skipped}")
                continue
            if len(timed_s) != 3 or len(timed_a) != 3:
                cell["skip_reason"] = (
                    f"incomplete timed batches stream={len(timed_s)} accept={len(timed_a)}"
                )
                doc["cells"].append(cell)
                save(doc)
                log(f"SKIP ctx={actual} C={c}: {cell['skip_reason']}")
                continue

            ttfts = [t["ttft_s"] for t in timed_s]
            decodes = [t["decode_tok_s"] for t in timed_s]
            accs = [t["accept"] for t in timed_a]
            cell["raw_ttft_s"] = ttfts
            cell["raw_decode_tok_s"] = decodes
            cell["raw_accept"] = accs
            cell["median_ttft_s"] = statistics.median(ttfts)
            cell["median_decode_tok_s"] = statistics.median(decodes)
            cell["median_accept"] = statistics.median(accs)
            notes = []
            for xs, lab in ((decodes, "decode"), (ttfts, "TTFT"), (accs, "accept")):
                n = messy_note(xs, lab)
                if n:
                    notes.append(n)
            cell["messy"] = notes
            doc["cells"].append(cell)
            save(doc)
            log(
                f"MEDIAN ctx={actual} C={c}: TTFT {cell['median_ttft_s']:.3f}s / "
                f"decode {cell['median_decode_tok_s']:.2f} tok/s / accept {cell['median_accept']:.3f}"
                + (f" ({'; '.join(notes)})" if notes else "")
            )

    doc["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(doc)
    log("WROTE " + str(OUT_JSON))
    log("WROTE " + str(OUT_MD))
    log(write_md(doc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
