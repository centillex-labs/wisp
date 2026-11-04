# Bonsai strawberry repair

From the repository root:

```bash
cargo build --release
./target/release/wisp examples/strawberry.wisp
```

[`strawberry.wisp`](../examples/strawberry.wisp) downloads and caches the pinned Bonsai-1.7B
checkpoint, runs native CPU inference, fits an output-row edit, saves and reloads
it, and asserts that generation returns `3`. No Python, task file, activation
preparation, or Clang is needed at runtime.

Allow about **8 GiB of available RAM** and **3.5 GB of cache storage**. The measured
cached run took 36.18 seconds and peaked at 6.54 GiB RSS on this machine. The
backend uses float32 CPU computation with four worker threads.

## Recorded result

Greedy generation of **one new token from the full 151,669-token vocabulary**:

| Repair prompt | Before | After |
| --- | --- | --- |
| Return only the number of occurrences of r in strawberry | `To` | `3` |
| Count r in cherry | `3` | `2` |
| Count r in banana | `3` | `0` |
| Count r in orange | `3` | `1` |

Both numerical-comparison controls remained `True` and `False`, respectively.
A fresh process loaded the cached model and saved edit, without fitting or saved
activations, and reproduced all six outputs. The completed program also ran with
an empty `PATH`, no `HF_TOKEN`, and `HF_HUB_OFFLINE=1`.

These exact prompts are in the program. **All six are fitting inputs**, including
the two zero-drift controls, not held-out validation. Some other strawberry prompts
already return `3` before editing. This demonstrates a targeted repair, not general
counting competence; the original strawberry failure here was the first token `To`,
not an incorrect numerical answer `2`.

## Edit and artifacts

The Wisp program solves a small regularized linear system using captured final
hidden states. Each desired digit receives a target logit one unit above the
original maximum; the controls request zero change. The `1`-logit margin and
`0.0001` ridge are engineering choices, not calibrated safety guarantees.

The saved edit changes **8,168 FP16 entries in four output rows** (`0`, `1`, `2`,
`3`), with relative four-row change **0.13912153**. The decoder and original input
embeddings remain unchanged. Output rows are overridden independently of the tied
input embeddings; edited rows are no longer constrained to the original 1-bit
structure.

Artifacts are written beside the cached checkpoint, outside the project:

- `strawberry_base.safetensors`: original output-row bundle.
- `strawberry_head.safetensors` and `.wisp.json`: edited rows and model binding.
- `strawberry_head.patch.json`: hash-checked `output_rows` patch, reconstructible
  against the **base row bundle**, not the full checkpoint.

`with_head(model, wload(path))` installs the saved rows. The binding checks checkpoint,
configuration, tokenizer, token IDs, and original-row hashes. This is a small
readout artifact, not a full modified checkpoint or a GGUF/MLX export.

[Recorded measurements](../examples/results/strawberry.json).

## Native model API

- `hf_model(repo, 'revision', commit [, 'sha256', hash] [, 'offline', true])`
  fetches `config.json`, `tokenizer.json`, and `model.safetensors` through the
  [verified HF cache](huggingface.md). A full commit hash is required.
- `capture(model, prompts)` returns `hidden`, full-vocabulary `logits`, greedy
  next-token `texts`, and `prompt_tokens`. Numeric observations carry `Act`
  provenance. It costs one observation pass per batch.
- `run(model, prompts)` returns final-position logits, at one pass per batch.
- `gen(model, prompts, steps, 'temperature', 0)` performs greedy autoregressive
  generation, stopping at EOS or the requested limit. It costs one pass per
  512 prompts, rounded up. Generated token arrays carry `Act` provenance.
- `token_head(model, {'0','1',...})` exposes selected output rows as an editable
  linear network, with `W1`, zero `b1`, and `token_ids`. Tokens must each encode
  as one distinct token. Inspection is not an observation charge.
- `with_head(model, head)` returns a model with that readout installed, without
  mutating the original. It replaces any previously installed readout.

Prompts are a cell of strings or a dataset name declared with `dataset`.
The current backend supports single-file, dense Qwen3 checkpoints with full
attention, SiLU, Q/K RMSNorm, GQA, and default or explicit YaRN rotary positions.
It uses Qwen3's non-thinking, single-user chat format. Unsupported architecture
settings are rejected before downloading weights; remote model code is never run.

Native requests are limited to 1–1,024 prompts, at most 2,048 context tokens
(or the model's smaller limit), and 1–64 generated tokens. Generation recomputes
the prefix rather than keeping a KV cache. Native `gen` currently supports only
greedy decoding; sampling and grader options are rejected.

The demonstrated program uses **3 logical passes and 18 decoder executions**:
six before-generation calls, six captures, and six after-generation calls.
Downloads, weight inspection, the solve, and saving spend no observation passes.
These counters are per invocation, not persistent exploration limits.
