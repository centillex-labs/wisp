# Programming a small Bonsai checkpoint

This example programs **PrismML Bonsai-1.7B**, using its official **unpacked FP16
safetensors export**. The workflow edits and verifies selected tensors.

## Pinned source

- Repository: https://huggingface.co/prism-ml/Bonsai-1.7B-unpacked
- Revision: `a7f720bd688d7563714f3118edd97b83d06f0615`
- License: Apache-2.0; LICENSE and NOTICE.txt are retained in the download cache.
- Parameters reported by the repository: 1,720,028,160.
- Weight file: `model.safetensors`, 3,440,091,640 bytes.
- SHA-256: `cf9a24cbd02e6e257bcfd3177475aaca7f8bd1a63a745441f30d3e40f4313a6b`.

This export stores the 1-bit model's coefficients as FP16 values. Plan memory use
for FP16 storage and Wisp's decoded tensors. Its Qwen3 configuration uses GQA,
RMSNorm, SwiGLU, and YaRN rotary-position scaling. For native behavioral evaluation, see the [strawberry workflow](strawberry.md).

## Run with an existing checkpoint

The structural example expects the pinned `model.safetensors` in its working
directory. With the release binary on your PATH:

```bash
cd /path/to/checkpoint
wisp /path/to/wisp/examples/bonsai_17.wisp --budget 0
```

Use [native HF downloads](huggingface.md) to acquire pinned files. The original
download/verification helpers have been retired. Recorded results below refer to
that earlier experiment. For a self-contained download-to-generation workflow,
run [strawberry.wisp](strawberry.md).

## The Wisp program

`examples/bonsai_17.wisp` selectively loads one actual checkpoint tensor:

```wisp
net = wload('model.safetensors', 'parts', 'model.layers.0.mlp.down_proj.weight');
D = net.L[0].mlp.down;  // 2048 × 6144: 12,582,912 weights
changed = rebalance_blocks(D, 128, 4);
```

The reusable `rebalance_blocks` function in that file:

1. Views the down projection as row-major 128-coefficient blocks.
2. Counts positive coefficients in each block using weight-only operations.
3. Selects the four blocks with the most positives, breaking ties by lower index.
4. Verifies each selected block has one nonzero absolute magnitude, i.e. values
   are exactly `+scale` or `-scale` in the unpacked tensor.
5. Flips the first two positive coefficients in each selected block to negative.
6. Asserts that all magnitudes and squared norms in those blocks are unchanged,
   positive counts fell by two, and signed sums fell by exactly `4*scale`.

The key write is simply:

```wisp
D[at] = -D[at];
```

This reduces the selected blocks' positive sign imbalance without changing their
scales. It is an **illustrative structural edit**, not evidence of better model
behavior. No inference, training, or gradients are used.

## Verified results

| Block (0-based) | Output row | Positive coefficients before → after | Absolute scale |
|---:|---:|---:|---:|
| 27674 | 576 | 88 → 86 | 0.044189453125 |
| 85551 | 1782 | 87 → 85 | 0.04248046875 |
| 16855 | 351 | 86 → 84 | 0.0308837890625 |
| 49314 | 1027 | 86 → 84 | 0.040771484375 |

Independent verification in the recorded run confirmed:

- The original complete checkpoint still has its pinned SHA-256.
- The selected blocks match an independently computed ranking.
- Exactly eight stored FP16 entries changed, each solely by XOR with `0x8000`
  (the sign bit).
- Every coefficient magnitude across the entire selected tensor is unchanged.
- Patch base and edited hashes match the raw tensor bytes.
- `wisp apply` reconstructs the edited tensor byte-for-byte.

The Wisp run used **0 observation passes**, took approximately **2.76 seconds**,
and peaked at **511 MiB RAM** on this machine. These measurements exclude download,
build, and independent verification time; they are not inference benchmarks.

## Saved artifacts

The output is a **selected-tensor bundle** containing the layer-0 down projection,
plus an eight-entry patch. The edit-budget denominator is the loaded 12,582,912
weights. Retain the original full checkpoint; applying the patch rebuilds this
selected bundle. Deployment requires integrating the tensor into the target model
and validating its storage format.

The workflow covers selective loading, canonical projection naming, structural
weight editing, FP16-preserving saves, and hash-checked reconstruction. For
behavioral checks, see [Bonsai behavioral repair](bonsai-behavior-repair.md).

Offline regression tests in `crates/wisp-runtime/tests/bonsai.rs` execute the same
Wisp helper on small synthetic Bonsai-style FP16 fixtures, check magnitude/sign-bit
preservation and patch round trips, and verify rejection of non-binary blocks.
These tests run offline.
