# Kernel mode

Drive Wisp statement by statement against one pinned model with one observation
budget. This is the harness a reasoning agent works in: inspect, think, probe,
edit, verify, revert, without reloading the model between steps.

```bash
wisp kernel setup.wisp [--log FILE]
```

```bash
wisp kernel setup.wisp [--log FILE] [--transcript FILE] [--resume FILE]
```

The setup program runs once, exactly like a normal program. It typically loads a
pinned checkpoint, sets `budget(n)`, and defines fixture variables. Then the
kernel reads statements from stdin:

- Statements are submitted as chunks ended by a blank line. Newlines inside a
  chunk are ordinary statement separators, and `...` continuations work.
- One JSON line comes back per chunk: `ok`, `output` (captured `disp` and
  `fprintf` text), `error` (rendered diagnostic, on failure), `budget_used`, and
  `budget_remaining`.
- The environment, defined functions, and observation accounting persist across
  chunks. Parse and runtime errors are reported and do not end the session.
- `exit` or end of input ends the session.
- `--transcript FILE` appends the setup source and every chunk and response as
  JSON lines. `--resume FILE` replays the recorded chunks against a fresh
  session before reading stdin, restoring state and budget; chunks are
  re-executed, so replay requires deterministic programs, and a replay failure
  aborts with the recorded chunk's diagnostic.
- Scoring with `score` charges passes but consumes no freshness, so a keep-if-
  better loop may rescore the same set; fit-purpose calls (`capture`, `run`,
  `gen`) reject held-out datasets with `E-HELDOUT`.

Example session:

```json
{"ok": true, "budget_used": 0, "budget_remaining": 20, "output": "", "event": "ready"}
{"ok": true, "budget_used": 1, "budget_remaining": 19, "output": "12.\n"}
{"ok": false, "budget_used": 2, "budget_remaining": 18, "output": "", "error": "error[E-BUDGET]: ..."}
```

An agent can therefore spend a budget deliberately: read behavior with `gen` and
`capture`, fit or edit through `token_head` and `with_head`, verify from the
saved FP16 rows via `wload`, and revert by reloading the baseline bundle. Every
saved edit still produces a hash-checked patch file, and every observation is
logged and charged. The kernel adds no capability beyond a normal program; it
changes who presses the keys and when.

## Causal loop, scoring, and provenance reads

Fix 1 to 6 of the WSP gap analysis are implemented:

- `read_weights(model, {'model.layers.23.mlp.down_proj.weight', ...})` reads
  checkpoint tensors bound to the model's identity.
- `with_writers(model, parts)` installs edited writer tensors; shapes, stored
  dtype, and binding are verified; `edited_parts` lists what changed.
- `capture(model, prompts, 'layers', [23, 26], 'attn', [0])` returns per-layer
  residual states and last-position attention views (`layers`, `layer_ids`,
  `attn`, `attn_layer_ids`) in one charged pass.
- `score(model, data, steps, grader)` scores greedy generations with a Wisp
  grader function and returns `scores` and `mean`; it may observe held-out data
  that `capture`/`run`/`gen` must not touch.
- `dataset(name, inputs, 'heldout', true)` marks an evaluation-only set.
- `sum`/`mean`/`max`/`min` dimension arguments now follow the MATLAB convention
  (dim 1 reduces rows); programs written for the old convention need their
  dimension arguments swapped.

The recorded causal test in [the refusal study](bonsai-refusal.md) used all of
these and falsified the single-column localization: removing the direction from
the three standout writer columns left every response unchanged.

## Recorded kernel session

`docs/bonsai-refusal.md` records the kernel sessions over Bonsai: baseline
generation, a two-prompt refusal fit, verification of all six candidates,
evaluation on the two reserved prompts, control checks, a revert, the writer-
direction trace, and the causal column-removal test that falsified the single-
column localization. Session output was captured as one JSON line per step.
