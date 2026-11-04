# Runtime observation APIs

See [budgets](budgets.md) for the distinction between observation passes and
changed-weight limits and their intended use.

Use `wisp program.wisp` to observe networks created with `mlp(...)` and `transformer(...)`,
or compatible MLP weight files. Architecture-specific checkpoint evaluation uses
an external inference host connected through trusted activation caches.

```bash
cargo build --release
./target/release/wisp examples/observations.wisp --log observations.jsonl
```

The example exercises teacher forcing, attention maps, generation with grading,
paired gain, and confirmation on fresh data, using exactly six passes.

## Self-contained programs

Declare inputs, limits, and protected weights in the program:

```wisp
budget(4);
dataset('practice', [0; 1; 2]);
dataset('confirmation', [3; 4; 5]);
protect('b1');
```

Run it with `wisp program.wisp`. No configuration file is needed.

- `dataset(name, inputs)` registers an array of rows or a cell of sequences.
  Names must be unique. Inputs are captured by value.
- `budget(n)` sets or tightens the total observation limit without resetting usage.
  `budget()` returns the remaining passes. An optional CLI `--budget` is a ceiling.
- `protect(name)` or `protect({'part1', 'part2'})` prevents changes to those parts
  when saving. Use canonical part names or exact file tensor names.

Pass dataset names to observations, such as `run(net, 'practice')`. Inline inputs
are also accepted. `confirm` requires a registered, previously unobserved dataset.

Successful observations record the data's content hash and first exposure.
Renaming identical data does not make it fresh. Budget refusals do not consume
passes or expose data; failures after charging retain the charge and exposure.

Content hashes cover array shape, numerical bytes, strings, and cell structure.
They do not detect every overlap or re-encoding. Keep evaluation inputs separate
from repair inputs and run trusted programs. Registered datasets reside in memory.

## Captured activations

Native models loaded by `hf_model` provide states directly in the program:

```wisp
state = capture(model, {'Your prompt'});
H = state.hidden; // Act provenance; one observation pass for the batch
```

See the complete [Bonsai strawberry program](strawberry.md).

An external inference host can also supply captured model states with `Act` provenance:

```wisp
H = activations('cached_hidden');  // one observation pass
```

The host registers these using `harness.register_activations(name, value)`.
Ordinary `dataset(...)` declarations cannot impersonate activation caches.
Cache reads preserve provenance, obey the budget, and record exposure. The same
names can be passed to `run(head, name)` for charged component replay.

The host is responsible for extraction, source hashes, upstream-cache validity,
and external inference accounting. A cache-read charge is additional to collection
cost. See [Bonsai behavioral repair](bonsai-behavior-repair.md) for the recorded
host workflow.

## Text/token inputs

Transformer observations accept a cell of sequences. Each sequence is one of:

- A string, when every vocabulary entry is a single character: `{'aab', 'baa'}`.
- A cell of exact vocabulary token strings: `{{'hello', 'world'}}`.
- A row/column of 0-based token IDs: `{[0 1 1]}`.

There is no implicit BPE/SentencePiece tokenizer, BOS insertion, padding, or EOS
handling. Inputs must be nonempty and within the network's configured context.
Token IDs are validated before indexing weights.

## `tf(net, texts)` — teacher forcing

Cost: `ceil(number_of_sequences / 1024)` passes, charged before forward execution.

```wisp
[logp, hidden] = tf(net, 'texts');
// Or one output:
r = tf(net, {'aab'});
logp = r.logprobs;
hidden = r.hidden;
```

Both outputs are cells to accommodate variable lengths. For a sequence of T tokens:

- `logp{i}` is a 1×(T−1) row. Entry t is the log probability of token t+1 given
  tokens 0..t. The first token has no score because no BOS context is inserted.
- `hidden{i}` is T×d: final-layer residual hidden states for every token.

Log probabilities use a max-shifted log-sum-exp calculation. Returned numeric
arrays carry `Act` provenance. A one-token sequence has an empty log-probability row.

## `attn(net, texts)` — attention maps

Cost: one pass per sequence.

Returns a cell of tensors with shape `[layers, heads, query_position, key_position]`.
Future/masked key positions are zero; each valid attention row sums to one.
The tensors carry `Act` provenance. No logits are returned.

## `gen(net, prompts, max_new_tokens, …)` — reproducible generation

Cost: `ceil(number_of_prompts / 512)` passes. One completion is generated per prompt.
The third argument is the maximum number of **new tokens** per prompt.

```wisp
g = gen(net, 'prompts', 4, 'seed', 7, 'temperature', 1, ...
        'grade', @(text) count(text, 'b'));
```

Options:

- `'seed', integer`: local deterministic SplitMix64 sampling, default 0.
- `'temperature', scalar`: nonnegative, finite; default 1. Zero chooses the
  highest-logit token deterministically, breaking ties by lowest token index.
- `'grade', @function`: optional callback taking the completion string and returning
  one finite scalar. Calls made by the grader are independently budgeted.

Result struct:

- `texts`: cell of completion strings, obtained by concatenating vocabulary entries.
- `tokens`: cell of generated token-ID rows (`Act`), excluding the prompt.
- `scores`: column of grades (`Act`), present when a grader is supplied.

Prompt plus completion must fit the configured context. Generation produces the
requested number of tokens, recomputing the prefix at each step. Sampling uses the
full vocabulary with the specified temperature. Returned strings carry no numeric
provenance; use observation-derived numeric scores for paired metrics.

## `pair(a, b, items, metric)` — paired evaluation

Cost: one pass for the outer call; observation calls inside each metric evaluation
are logged as covered by that pass.

```wisp
metric = @(model, items) run(model, items);
[gain, se] = pair(a, b, 'train', metric);
```

The metric receives the net and the original items argument (a dataset name or
inline inputs). It must observe those same data and return one finite,
observation-derived scalar per item, as a vector. At least two items are required.
Calls observing different data, metrics returning weight-only/constant scores,
and nested `pair`/`confirm` calls are refused.

For paired differences `d = scores_b - scores_a`:

- `gain = mean(d)`
- `se = sqrt(sample_variance(d) / number_of_items)`

Both outputs have `Act` provenance. They describe improvement only if the metric
is oriented so larger is better.

## `confirm(a, b, set_name, metric)` — fresh-set confirmation

Cost: one pass; metric observations are covered as with `pair`.

The set name must be registered and accessible to the program. Its exact data identity
must not have been exposed through an earlier tracked observation, even under a
different name or as identical inline data. Rejection uses `E-REUSED-SET` and names
the earlier call/source location.

Returns a logical scalar with `Act` provenance: `gain > 2 * se`.

A failed callback stops immediately, does not invoke the second metric, and restores
ordinary charging. The already-observed confirmation set remains consumed.
Metrics must observe the supplied dataset. `examples/edits.wisp` declares its
practice and confirmation inputs directly in the program.
