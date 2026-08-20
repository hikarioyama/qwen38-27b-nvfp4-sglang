# Qwen3.8-27B C=1 @1024 matrix

Same-condition: chat completions, thinking off, T=0, seed=0, max_tokens=1024, C=1, n=3 median after 1 warmup discard.
Token count = `usage.completion_tokens`. Accept = `meta_info.spec_accept_length` (`return_meta_info` on non-stream only).
TTFT = separate stream first content delta, **without** `return_meta_info`. Only `huihui-dflash2` LRU measured (n=3) on the live restored server; other cells TTFT null (no 3-recipe recyle).
`docker run -d`. SIGTERM between recipes.

Restore target: Huihui ReplaySSM DFLASH K=8, `--disable-overlap-schedule`, **without** `--max-running-requests 64`.

## Median table

| recipe | prompt | tok/s | TTFT s | accept |
|---|---|---:|---:|---:|
| huihui-dflash2 | LRU | 270.08 | 0.026 | 5.278 |
| huihui-dflash2 | JA-code | 241.05 | | 4.719 |
| huihui-dflash2 | EN-prose | 135.43 | | 2.672 |
| huihui-dflash2 | JA-prose | 91.41 | | 1.795 |
| unsloth-dflash2 | LRU | 265.75 | | 5.198 |
| unsloth-dflash2 | JA-code | 242.71 | | 4.741 |
| unsloth-dflash2 | EN-prose | 136.11 | | 2.661 |
| unsloth-dflash2 | JA-prose | 95.08 | | 1.850 |
| unsloth-nextn | LRU | 120.32 | | 3.507 |
| unsloth-nextn | JA-code | 114.87 | | 3.346 |
| unsloth-nextn | EN-prose | 88.37 | | 2.583 |
| unsloth-nextn | JA-prose | 79.89 | | 2.339 |

## Lines

huihui-dflash2 chat LRU thinking-off @1024 : 270.08 tok/s (n=3, same-condition) | TTFT 0.026s | accept 5.278
huihui-dflash2 chat JA-code thinking-off @1024 : 241.05 tok/s (n=3, same-condition) | TTFT NA | accept 4.719
huihui-dflash2 chat EN-prose thinking-off @1024 : 135.43 tok/s (n=3, same-condition) | TTFT NA | accept 2.672
huihui-dflash2 chat JA-prose thinking-off @1024 : 91.41 tok/s (n=3, same-condition) | TTFT NA | accept 1.795
unsloth-dflash2 chat LRU thinking-off @1024 : 265.75 tok/s (n=3, same-condition) | TTFT NA | accept 5.198
unsloth-dflash2 chat JA-code thinking-off @1024 : 242.71 tok/s (n=3, same-condition) | TTFT NA | accept 4.741
unsloth-dflash2 chat EN-prose thinking-off @1024 : 136.11 tok/s (n=3, same-condition) | TTFT NA | accept 2.661
unsloth-dflash2 chat JA-prose thinking-off @1024 : 95.08 tok/s (n=3, same-condition) | TTFT NA | accept 1.850
unsloth-nextn chat LRU thinking-off @1024 : 120.32 tok/s (n=3, same-condition) | TTFT NA | accept 3.507
unsloth-nextn chat JA-code thinking-off @1024 : 114.87 tok/s (n=3, same-condition) | TTFT NA | accept 3.346
unsloth-nextn chat EN-prose thinking-off @1024 : 88.37 tok/s (n=3, same-condition) | TTFT NA | accept 2.583
unsloth-nextn chat JA-prose thinking-off @1024 : 79.89 tok/s (n=3, same-condition) | TTFT NA | accept 2.339
