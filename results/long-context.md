# Long-context REAL-text matrix

Unsloth NVFP4 DFlash2, K=8, ReplaySSM, KV FP8. `max_new_tokens=1024`, thinking off, T=0, seed=0.

Each cell is the **median of 3 timed batches** after 1 warmup discard (same-condition):

**decode tok/s after first token** / **prefill TTFT (s)** / **accept length**

Prompt filler is **repeated natural English technical prose**. The last paragraph is the EN-code LRU task. Filler is **not** pad tokens and **not** the word `padding`.

A prior pad-token filler matrix is **invalid** and is **not published**.

The short-prompt C=1 matrix in [`MATRIX.md`](MATRIX.md) is unchanged.

`31744` input tokens is clamped so `input + 1024` fits a 32768 context window.

Machine-readable copy: [`long-context.json`](long-context.json).

## Decode tok/s (after first token, n=3 median)

| input tokens \ C | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| 8192 | 241.57 | 232.63 | 186.81 | 131.78 |
| 16384 | 262.08 | 213.14 | 179.09 | 127.04 |
| 31744 | 196.86 | 185.08 | 160.21 | 119.51 |

## Prefill TTFT (seconds, n=3 median)

| input tokens \ C | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| 8192 | 0.606 | 0.917 | 1.535 | 2.769 |
| 16384 | 0.759 | 1.120 | 1.852 | 3.316 |
| 31744 | 0.918 | 1.329 | 1.553 | 2.514 |

## Accept length (n=3 median)

| input tokens \ C | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| 8192 | 4.655 | 5.020 | 4.876 | 4.592 |
| 16384 | 5.146 | 4.676 | 4.923 | 4.900 |
| 31744 | 3.946 | 4.206 | 4.104 | 4.189 |

## Combined cells

| input tokens | C | decode tok/s | TTFT s | accept |
|---|---:|---:|---:|---:|
| 8192 | 1 | 241.57 | 0.606 | 4.655 |
| 8192 | 2 | 232.63 | 0.917 | 5.020 |
| 8192 | 4 | 186.81 | 1.535 | 4.876 |
| 8192 | 8 | 131.78 | 2.769 | 4.592 |
| 16384 | 1 | 262.08 | 0.759 | 5.146 |
| 16384 | 2 | 213.14 | 1.120 | 4.676 |
| 16384 | 4 | 179.09 | 1.852 | 4.923 |
| 16384 | 8 | 127.04 | 3.316 | 4.900 |
| 31744 | 1 | 196.86 | 0.918 | 3.946 |
| 31744 | 2 | 185.08 | 1.329 | 4.206 |
| 31744 | 4 | 160.21 | 1.553 | 4.104 |
| 31744 | 8 | 119.51 | 2.514 | 4.189 |
