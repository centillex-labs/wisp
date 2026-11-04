# Budgets

Wisp has two independent budgets:

| Budget | Limits | Purpose |
| --- | --- | --- |
| Observation passes | How often a program measures model behavior | Bound evaluation cost and feedback available while choosing an edit |
| Changed-weight fraction | How many loaded weight entries an edit changes | Keep a repair scoped and prevent accidental broad rewrites |

Neither budget guarantees improved quality or prevents regressions.

## Observation budget

```wisp
budget(10);
dataset('repair_cases', [1 0; 0 1]);
```

Run with `wisp experiment.wisp`. `budget()` returns remaining passes.
`budget(n)` sets or tightens the total limit; it never resets usage or raises an
existing limit. A limit below already-used passes is rejected.

Optionally, `wisp experiment.wisp --budget 5` imposes a stricter ceiling that the
program cannot raise.

```wisp
scores = run(net, 'repair_cases'); // one pass
assert(all(scores > 0));          // no additional charge
```

The observation costs a pass, not the assertion that checks its result.

| Operation | Current cost |
| --- | ---: |
| `run(net, inputs)` | 1 per call |
| `activations('cache')` | 1 per cache read |
| `capture(model, prompts)` | 1 per batch |
| `tf(net, texts)` | `ceil(number of sequences / 1024)` |
| `attn(net, texts)` | 1 per sequence |
| `gen(net, prompts, ...)` | `ceil(number of prompts / 512)` |
| `pair(...)` / `confirm(...)` | 1 outer pass covering nested metric observations |
| Weight-only inspection, editing, and assertions | 0 |
| `hf_download(...)`, `hf_model(...)`, `wload(...)`, and `token_head(...)` | 0 |

Recognized weight computations that act as behavioral probes can also be charged.
See [observation APIs](observations.md) for exact contracts.

An operation exceeding the limit fails with `E-BUDGET` before the observation runs.
A budget refusal does not consume passes or mark data observed. A failure after
charging can still consume the pass and expose the dataset.

### What it prevents—and what it does not

An observation limit constrains repeated trial-and-error tuning and makes feedback
budgets comparable between editing methods. It helps control evaluation expense
and limits opportunities to overfit known examples.

It does **not** prevent overfitting by itself. Freeze the candidate before testing
on unused confirmation data, and do not retune against that data afterward.

A pass is a logical accounting unit, not necessarily one physical forward execution.
It does not measure money, tokens, elapsed time, RAM, or network traffic.
`--budget 0` permits weight-only work; it does not make that work costless.
External inference must be accounted for separately unless integrated with the
harness. Cached activations do not erase their collection cost.

## Changed-weight budget

```wisp
wsave(candidate, 'edited.safetensors', 'budget', 0.01);
```

Here `0.01` means **at most 1% of the loaded weight entries may differ from their
base values**. It does not mean one observation pass.

The check compares values in their stored dtype, such as FP16 or BF16. Repeated
writes to the same entry count as one changed entry if its final stored value
differs from the base. An edit that rounds back to the original value does not count.
Exceeding the limit fails with `E-EDIT-BUDGET` before writing the model.

For example, eight changed entries out of 1,000 loaded weights use a fraction of
`0.008`, or 0.8%. If only a tensor bundle was loaded, the denominator is that bundle,
**not the full checkpoint**.

### What it prevents—and what it does not

This limit guards against accidentally rewriting a large part of the loaded model.
It counts changed entries, **not the magnitude or behavioral impact of changes**.
One changed weight can still cause a major regression.

Use `protect('part_name')` to preserve a part when saving. Add assertions for change
magnitude, repair behavior, and regressions. Validate the saved-dtype artifact too.

## Scope

Observation history belongs to the current harness; edit comparisons use the
current network's base snapshot. These are not automatic lifetime limits across
restarted sessions or repeatedly reloaded edited models. The host must preserve
experiment state when enforcing a cumulative policy.

Run trusted programs. For untrusted code, enforce process isolation
and dataset access controls independently of the observation budget.
