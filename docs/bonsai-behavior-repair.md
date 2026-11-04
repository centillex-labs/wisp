# Bonsai behavioral repair

This experiment edits **Bonsai-1.7B-unpacked** weights and measures improvements
on a defined character-counting task.

The endpoint is a **constrained boolean classifier**: after the chat prompt, choose
whichever of the single tokens `False` or `True` has the larger next-token logit.
The Python host runs the Qwen3-based Bonsai decoder; Wisp replays and edits the
selected output head. Evaluation covers these two output tokens.

## What was reproduced and repaired

The prompt is:

```text
Evaluate whether this numerical comparison is correct. Answer with a boolean value only.
Statement: The word 'strawberry' contains exactly 3 occurrences of the letter 'r'.
```

Six practice words (`strawberry`, `raspberry`, `blackberry`, `blueberry`, `cranberry`,
`cherry`) each have a false and a true count statement. The baseline chose `False`
for all twelve. Twelve integer-comparison cases provide regression constraints.

Before fitting, the confirmation cases were fixed to the corresponding six plural
words, each again with a false and true statement. No confirmation forward pass
ran until the candidate file was frozen by SHA-256. The runner refuses refitting
or another confirmation attempt after that point, including after a failed check.

| Suite | Baseline | Saved edit, applied to the actual model |
| --- | ---: | ---: |
| Practice repair | 6/12 | 12/12 |
| Integer-comparison regression | 12/12 | 12/12 |
| Frozen plural-word confirmation | 6/12 | 12/12 |

- Wisp's paired confirmation: gain **0.5**, SE **0.1507556723**; `gain > 2*SE` passes.
- Maximum regression margin change in the host: **0.0015659332**, below the **0.05** bound.
- **4,092 FP16 entries** changed, in only two embedding/output rows: token IDs
  **4049 (`False`)** and **2514 (`True`)**.
- The relative Frobenius-norm change of those two rows is **0.3934648**, below 0.5.
  This is a substantial local change, not a tiny perturbation of the selected head.
- Edited patch provenance is **`Act`**, not `Lit` or a falsely observation-free edit.
- The original checkpoint's hash remains unchanged. The host applies the patch in
  memory; no complete edited checkpoint is written.

Full case-level evidence and accounting are in
[`examples/results/bonsai_behavior_repair.json`](../examples/results/bonsai_behavior_repair.json).
The saved head SHA-256 is
`07385bb3ca96aed8517e5b2e65399fede4d1888d8eeac0b7527a8887202aa6b9`.

### Attempts that did not become successful repairs

The initial decimal-comparison suite was already 12/12 correct, so no repair was
claimed. A practice-only discovery set then identified the character-counting
failure. A 128-coordinate correction was rejected because its relative head
change was 2.104, above the 0.5 bound. Its source and observation log were retained.

The accepted correction uses all coordinates of the two rows to minimize its norm.
The optimization target was adjusted using practice data only; the acceptance
requirements remained correct practice labels, margin at least 0.25, head norm
change at most 0.5, and regression drift at most 0.05. **No tuning followed the
confirmation result.** This is not evidence for an eight-entry/sign-only repair.

## The Wisp part

[`examples/bonsai_behavior_repair.wisp`](../examples/bonsai_behavior_repair.wisp):

1. Load the two actual output rows as a supported linear MLP head.
2. Fetch trusted cached final hidden states with `activations('repair')` and
   `activations('regression')`. These cost passes and carry `Act` provenance.
3. Assert that the baseline really fails the repair suite and passes regression.
4. Compute a supervised, ridge-regularized minimum-norm correction using `\`.
   This is **closed-form fitting**, not gradient descent and not a claim of
   learning without supervision.
5. Assert the edit scope and relative norm bound.
6. Save to FP16, reload, and assert correct repair labels, minimum repair margin,
   preserved regression labels, and bounded regression margin drift.

The core calculation is:

```wisp
H = [R; G];
residual = [(2*y-1)*0.3 - margin; zeros(rows(G), 1)];
change = H' * ((H*H' + 0.0001*eye(rows(H))) \ residual);
candidate.W1[1, :] += (change/2)';
candidate.W1[0, :] -= (change/2)';
```

The actual script also checks the baseline and the stored-dtype artifact. A failed
assertion stops the runner before candidate freezing.

[`examples/bonsai_behavior_confirm.wisp`](../examples/bonsai_behavior_confirm.wisp)
rechecks repair/regression and calls:

```wisp
assert(confirm(base, stored, 'confirmation', @accuracy));
```

`accuracy` performs a charged `run` and returns one observation-derived correctness
score per item. It does not return constant or weight-only scores.

## Why local replay represents this endpoint

The host extracts the final normalized hidden state from the actual decoder. A
linear projection onto the two selected vocabulary rows is exactly the endpoint's
restricted output head; computing the other vocabulary logits is unnecessary.

Bonsai ties its input embeddings and output head. The host:

- checks that they actually share storage;
- refuses any tested prompt containing either edited token ID;
- applies the saved FP16 rows to the actual model with `index_copy_`;
- reruns the decoder after the edit;
- asserts bit-identical before/after hidden states on confirmation and regression.

Thus this particular edit cannot change upstream computation for these tested
prompts. It **can** affect upstream computation for other prompts containing the
edited token IDs. The example does not claim arbitrary internal-block replay is
equivalent to end-to-end inference.

## Reproduce from a fresh experiment directory

From the workspace root, with approximately 10 GiB of available RAM:

```bash
cargo build --release
python3 scripts/fetch_bonsai_17.py

uv venv "$HOME/.cache/wisp/inference-venv"
PY="$HOME/.cache/wisp/inference-venv/bin/python"
uv pip install --python "$PY" 'torch==2.15.0+cpu' \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$PY" -r scripts/requirements-bonsai-inference.txt

"$PY" scripts/bonsai_behavior_repair.py prepare
"$PY" scripts/bonsai_behavior_repair.py fit
"$PY" scripts/bonsai_behavior_repair.py verify
"$PY" scripts/bonsai_behavior_repair.py summarize
```

The host uses local files only for inference and disables remote model code. It
uses the official chat template with thinking disabled and
`fix_mistral_regex=True`. Computation is CPU **float32**, using the exact values
from the FP16 checkpoint. Four PyTorch threads were used. The verification stage
in the recorded run took about **101 seconds**; preparation took about **37 seconds**.

Model revision and hash are pinned by `scripts/fetch_bonsai_17.py`. Default artifacts:

```text
~/.cache/wisp/models/prism-ml--Bonsai-1.7B-unpacked/
  a7f720bd688d7563714f3118edd97b83d06f0615/letter_count_repair/
```

Important files:

- `preparation.json`, `practice_states.npz`, `task_repair.wisp`: trusted extraction.
- `head_candidate.safetensors`, `.patch.json`: saved two-token head and Wisp patch.
- `frozen.json`: candidate/preparation/case hashes, created only after assertions pass.
- `row_patch.json`: maps head rows back to `model.embed_tokens.weight`; **not a full checkpoint**.
- `evaluation.json`, `summary.json`: actual model results, including failures if any.
- `external_observations.jsonl`, Wisp JSONL logs: observations, including rejected attempts.

Existing frozen/confirmed experiments are not silently overwritten. `summarize`
can inspect the completed run without another model observation. `WISP_EXPERIMENT_DIR`
can select another external output directory, but replaying these published cases
there does **not** make them a new unseen holdout for a subsequently tuned edit.

## Accounting and trust boundary

The recorded exploration used **24 logical observation charges**: 10 external
batches and 14 Wisp charges, including the rejected sparse attempt. External
execution was sequential within each logical batch: **144 actual decoder forward
executions** are recorded, not disguised as 24 physical forwards.

For the accepted fitting script alone, Wisp charges six passes: two activation-cache
reads and four head replays. Its final assertion script costs four passes. External
inference has a separate precharged host ledger and a 16-batch budget per experiment.
Costs are added, not treated as free because states were later cached.

Captured activations are supplied by the **trusted inference host**. Their
provenance depends on that host's extraction and validation. The host verifies model
and artifact hashes and enforces freeze-before-confirmation. Its external ledger
is separate from the Wisp interpreter's data-exposure history. Wisp's `confirm`
replays this frozen comparison in its execution scope; it is **not a second independent
holdout experiment**. This does not solve general sandbox/provenance loopholes.

## Evaluation and deployment

- The demonstrated improvement is **this constrained boolean endpoint on this
  narrow statement family**, not open-ended generation or overall Bonsai quality.
- Confirmation uses morphological variants of the practice words. Twelve curated,
  related items are not an independent representative benchmark. The `2*SE` rule
  passing is not a broad statistical guarantee.
- The smallest confirmation margin is only **0.0054**. Different kernels, precision,
  prompts, or quantization may change the result. Revalidate on the deployment backend.
- The dense correction breaks the original 1-bit block structure in the two edited
  rows. Deploying a packed GGUF/MLX artifact requires a separate conversion and validation workflow.
- Only the stated regression cases were checked. Other boolean judgments and other
  tasks may regress. No whole-model preservation claim is made.
- Model execution uses the external Transformers host. Wisp executes the extracted
  head and checks the saved edits.

Offline Cargo tests cover activation provenance, authorization, freshness, charging,
dataset registration, and both Wisp programs using a synthetic FP16 fixture.
These tests run offline. Checkpoint-level results are recorded separately by the
host experiment above.
