#!/usr/bin/env python3
"""Same-condition single-stream decode bench for a served SGLang endpoint.

Does not start or stop a server. Does not touch GPUs.

Protocol:
  C=1, max_tokens=1024, T=0, seed=0, thinking off, /v1/chat/completions
  warmup 1 discarded, then n=3 median
  tok count = usage.completion_tokens (never SSE chunks)
  accept = choices[0].meta_info.spec_accept_length (return_meta_info=true)
  TTFT = separate stream, first content delta, n=3

Line format:
  <recipe> chat <prompt> thinking-off @1024 : <tok/s> tok/s (n=3, same-condition) | TTFT <s> | accept <len>
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.request

PROMPTS = {
    "LRU": "Write a complete, production-quality Python LRU cache with per-key TTL and a pytest suite.",
    "JA-code": "PythonでTTL付きLRUキャッシュをクラスとして書いて。pytestも同じファイルに含めて。",
    "EN-prose": "Explain the difference between NVFP4 and BF16 quantization in three paragraphs.",
    "JA-prose": "量子化NVFP4とBF16の違いを3段落で説明して。",
}


def _post(url: str, body: dict, timeout: float = 600):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def chat_body(model: str, prompt: str, max_tokens: int, stream: bool, return_meta: bool) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 0,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if return_meta:
        body["return_meta_info"] = True
    return body


def nonstream(base: str, model: str, prompt: str, max_tokens: int) -> dict:
    t0 = time.perf_counter()
    r = _post(base + "/v1/chat/completions", chat_body(model, prompt, max_tokens, False, True))
    dt = time.perf_counter() - t0
    toks = r["usage"]["completion_tokens"]
    choice = (r.get("choices") or [{}])[0]
    meta = choice.get("meta_info") or {}
    accept = meta.get("spec_accept_length")
    return {
        "completion_tokens": toks,
        "seconds": dt,
        "wall_tok_s": (toks / dt) if dt else None,
        "spec_accept_length": accept,
        "finish_reason": choice.get("finish_reason"),
        "usage": r.get("usage"),
    }


def ttft(base: str, model: str, prompt: str, max_tokens: int) -> float | None:
    body = chat_body(model, prompt, max_tokens, True, False)
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    first = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
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
            delta = (obj.get("choices") or [{}])[0].get("delta") or {}
            content = delta.get("content") or delta.get("reasoning_content")
            if content and first is None:
                first = time.perf_counter() - t0
                break
    return first


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Same-condition single-stream SGLang bench")
    ap.add_argument("--base", default=os.environ.get("BENCH_BASE", "http://127.0.0.1:8040"))
    ap.add_argument("--model", default=os.environ.get("BENCH_MODEL", "/model"))
    ap.add_argument("--recipe", default=os.environ.get("BENCH_RECIPE", "recipe"))
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default="LRU")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    prompt = PROMPTS[args.prompt]
    out = {
        "recipe": args.recipe,
        "prompt": args.prompt,
        "base": args.base,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "n": args.n,
        "same_condition": True,
        "thinking": "off",
        "warmup": None,
        "runs": [],
        "ttft_s": [],
    }

    print("warmup...", flush=True)
    out["warmup"] = nonstream(args.base, args.model, prompt, args.max_tokens)
    print("WARMUP", json.dumps(out["warmup"], default=str), flush=True)

    for i in range(args.n):
        rec = nonstream(args.base, args.model, prompt, args.max_tokens)
        rec["i"] = i + 1
        out["runs"].append(rec)
        print("RUN", json.dumps(rec, default=str), flush=True)

    for i in range(args.n):
        if i:
            time.sleep(2)
        val = ttft(args.base, args.model, prompt, args.max_tokens)
        out["ttft_s"].append(val)
        print(f"TTFT {i + 1} {val}", flush=True)

    tok_s = [r["wall_tok_s"] for r in out["runs"]]
    accepts = [r["spec_accept_length"] for r in out["runs"]]
    med_tok = median(tok_s)
    med_ttft = median(out["ttft_s"])
    med_acc = median(accepts)
    out["median_tok_s"] = med_tok
    out["median_ttft_s"] = med_ttft
    out["median_accept"] = med_acc

    tok_s_s = f"{med_tok:.2f}" if med_tok is not None else "TBD"
    ttft_s = f"{med_ttft:.4f}" if med_ttft is not None else "TBD"
    acc_s = f"{med_acc:.3f}" if med_acc is not None else "TBD"
    line = (
        f"{args.recipe} chat {args.prompt} thinking-off @{args.max_tokens} : "
        f"{tok_s_s} tok/s (n={args.n}, same-condition) | TTFT {ttft_s}s | accept {acc_s}"
    )
    out["line"] = line
    print(line, flush=True)

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(out, fh, indent=2)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
