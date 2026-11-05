# Kernel mode

Drive Wisp statement by statement against one pinned model with one observation
budget. This is the harness a reasoning agent works in: inspect, think, probe,
edit, verify, revert, without reloading the model between steps.

```bash
wisp kernel setup.wisp [--log FILE]
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

## Recorded kernel session

`docs/bonsai-refusal.md` records a kernel session over Bonsai: baseline
generation, a two-prompt refusal fit, verification of all six candidates,
evaluation on the two reserved prompts, control checks, and a revert. Session
output was captured as one JSON line per step.
