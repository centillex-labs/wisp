# Editing Qwen3.5-0.8B weights

The experiment uses the public **Qwen/Qwen3.5-0.8B** checkpoint (Apache-2.0), pinned
to Hugging Face revision `2fc06364715b967f1860aea9cf38778875588b17`.

- Stored parameters reported by the repository: 873,438,784 (including vision/MTP).
- Weight file: 1,746,942,600 bytes.
- SHA-256: `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.
- Source: https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17

This workflow inspects and edits three language-model MLP tensors. Behavioral
evaluation requires an external inference backend supporting Qwen3.5's hybrid
linear/full attention, gated attention, and vision components.

## Download, run, rebuild, independently verify

From the Wisp workspace:

```bash
ROOT="$PWD"
cargo build --release
python3 scripts/fetch_qwen_08.py
MODEL_DIR="${WISP_MODEL_CACHE:-$HOME/.cache/wisp/models}/Qwen--Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17"

(
  cd "$MODEL_DIR"
  "$ROOT/target/release/wisp" "$ROOT/examples/qwen_08.wisp" --budget 0
  "$ROOT/target/release/wisp" apply \
    model.safetensors-00001-of-00001.safetensors \
    qwen08_layer0_edit.patch.json qwen08_layer0_rebuilt.safetensors
)
python3 scripts/verify_qwen_08.py
```

No Python packages beyond the standard library are needed. Downloads and generated
weights stay outside the workspace/Dropbox, under `~/.cache/wisp/models` by default.
The download script verifies the pinned full-checkpoint SHA-256. It retains the
repository config, index, README, and license beside the weights.

## Why selective loading

Wisp currently expands weights to float64 and keeps a base snapshot for patches.
Loading all 873M parameters would require about 14 GB just for those two copies,
plus temporary buffers. Rather than exhaust RAM, the experiment loads only the
first language-model layer's three MLP tensors:

```wisp
net = wload('model.safetensors-00001-of-00001.safetensors', 'parts', {
    'model.language_model.layers.0.mlp.down_proj.weight', ...
    'model.language_model.layers.0.mlp.gate_proj.weight', ...
    'model.language_model.layers.0.mlp.up_proj.weight'
});
```

`'parts'` accepts exact **file tensor names**, as a string or cell. It currently
requires safetensors, not npz. The loader validates the whole header/offset layout,
then seeks to and decodes only the selected tensors. It does not read the entire
checkpoint into RAM. Empty selections and missing names are errors.

Known HF dense layer projections map to `L[l].mlp.down`, `.gate`, `.up`, and
`L[l].attn.q/k/v/o`. These PyTorch projection matrices have `[out, in]` orientation.
Other file names remain untouched. Selected bundles expose tensors for inspection
and editing; execution requires an architecture-compatible backend.

### Important output scope

A selectively loaded net is a **tensor bundle**, not the complete network.
`wsave` writes only those selected tensors. Its sidecar and patch preserve the
selection; the patch records `scope: "selected_parts"` and `selected_files`.
`wisp apply` rebuilds that same bundle without decoding the unselected checkpoint.

- Do not use the bundle as a replacement for the full Qwen checkpoint.
- The edit-budget denominator is the number of loaded weights, not all 873M weights.
- The original checkpoint is never overwritten by this example.
- Integrate the selected tensors into a complete checkpoint and validate it before deployment.

## The actual edit

`examples/qwen_08.wisp` loads 11,010,048 weights and computes:

```wisp
D = net.L[0].mlp.down;        // 1024 outputs × 3584 MLP neurons
energy = sum(D.^2, 0);       // squared norm of each residual-writing column
[largest, neurons] = maxk(energy, 4);

for j = neurons
    [~, out] = max(abs(D[:, j]));
    D[out, j] *= 0.5;        // one coefficient per selected neuron, not the whole column
end
```

All operations inspect weights only. The program succeeds under `--budget 0`.

Observed results on the pinned checkpoint:

| Neuron | Squared column norm (rounded) | Output dimension | Before | After |
|---:|---:|---:|---:|---:|
| 204 | 0.091068 | 0 | -0.2001953125 | -0.10009765625 |
| 2557 | 0.090760 | 0 | -0.08642578125 | -0.043212890625 |
| 282 | 0.090195 | 0 | 0.060302734375 | 0.0301513671875 |
| 1833 | 0.090106 | 512 | 0.031005859375 | 0.0155029296875 |

Exactly **four stored BF16 entries** change, all in the down projection. The gate
and up projections remain byte-identical. This is illustrative sparse attenuation,
not a validated behavioral intervention.

On this machine, the Wisp editing run alone took about **0.41 s** and peaked at
**263 MiB RAM**. Download/build time and full-model inference are not included.

The independent Python verifier checks:

1. The original checkpoint still has its pinned SHA-256.
2. Top-column ranking and strongest-coefficient selection match Wisp's result.
3. Exactly four stored entries changed, each to half its original value.
4. Patch base/edited tensor hashes match the actual bytes.
5. The two other selected tensors remain unchanged.
6. Every rebuilt selected tensor is byte-identical to the saved edited tensor.
