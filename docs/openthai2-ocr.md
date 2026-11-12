# OpenThai2 OCR: native CPU forward execution

Wisp's interpreter and persistent kernel can execute the OpenThai2.0 convolutional
line recognizer directly. `run` reads the model's **current parts on every call**;
it does not use pre-recorded activations or an external inference process.

The implementation is Rust FP32 CPU math: explicit grouped/depthwise convolution
kernels plus Candle tensor operations. No Python, ONNX Runtime, GGML library,
network service, or subprocess is needed for the forward pass. This is not a
new LLVM graph compiler or a general ONNX importer.

## Export and load

The companion `ocr-thai` project contains the development converter
`tools/export_wisp_ocr.py`. From that project:

```sh
analysis/openthai2/.venv/bin/python tools/export_wisp_ocr.py \
  --model models/openthai2-rec-f32.gguf \
  --output models/openthai2-rec-wisp-native-f32.safetensors
```

It exports all 108 FP32 tensors without changing their bytes, and writes a
`.safetensors.wisp.json` sidecar containing the bounded graph. The sidecar is
required for execution and is preserved by `wsave`. A separate provenance file
records hashes. Existing exports are not overwritten.

The graph retains ONNX storage layout, including `[in,out]` head matrices and
NCHW convolution filters. **Do not assume these named graph operands have the
usual Wisp MLP `[out,in]` layout.** The converter explicitly disables automatic
transposition. Scalar constants retain scalar shape.

## Observe

```wisp
budget(2);
model = wload('/absolute/path/to/openthai2-rec-wisp-native-f32.safetensors');
inputs = wload('/absolute/path/to/prepared-input.safetensors');
x = 1 * inputs.x;

probabilities = run(model, x);
[probabilities, states] = run(model, x);
assert(budget() == 0);
```

- Input: finite FP32-compatible **normalized `[N,3,48,W]`** tensors;
  `N=1..16`, `W=16..3200`. Pass exactly the same preprocessing and zero-padding
  used by the OCR recognizer (`pixel/127.5-1` after its flatten/resize pipeline).
- Output: FP32 probabilities exposed as Wisp arrays, `[N,T,530]`.
- Optional second result: a struct of all node outputs, keyed by graph IDs
  (`v0000` through `v0117` for the published model). The converter's existing
  GGUF trace maps these IDs back to ONNX names.
- Both probabilities and captured states carry **`Act` provenance**.
- A call costs **one observation pass**, including optional intermediate capture.
  Budget zero and held-out inputs are rejected by the existing harness before
  executing the graph. Native computation errors after admission consume the pass.
- Individual tensors are capped at 64M elements. Capture additionally caps the
  sum of captured tensor elements at 64M; use small inputs for full capture.
  These are allocation safeguards, not a hard process-memory/time sandbox.
- `RAYON_NUM_THREADS` controls the convolution worker pool. The pool is process-wide.

The graph covers Conv1d/Conv2d with groups, asymmetric spatial strides, bias,
GELU expressed through Erf, ReLU, hard-sigmoid/swish, average pooling, squeeze,
transpose, broadcast arithmetic, classifier matmuls, FP32 Cast and softmax.
Unsupported operators/attributes and malformed graph references fail explicitly.

No image decoding, crop geometry, CTC string decoding, detector, page layout,
reading-order logic, or gradients are added here. Decode greedy CTC from the
probabilities using the original 530-class dictionary (blank class 0).

## Kernel use and edits

`examples/openthai2_kernel.wisp` loads the model named by `WISP_OCR_MODEL` and
reserves two observations. Start it with:

```sh
WISP_OCR_MODEL=/absolute/path/model.safetensors \
  ./target/release/wisp kernel examples/openthai2_kernel.wisp --log observations.jsonl
```

Submit ordinary Wisp snippets, separated by blank lines. Load or register inputs,
then call `run`. Weight inspection alone uses no passes. For strictly weight-only
analysis, use a setup containing `budget(0)` instead; a budget cannot be raised
later in the session.

Use `copy(model)` for candidates. Edits to any graph weight affect the next call.
`wsave` still enforces its protection, changed-entry budget, and global-knob
rules; the native executor does not bypass them. There is no stale compiled-weight
cache. Saving FP32 rounds edits to storage precision; forward also casts live
parts to FP32, making saved/reloaded inference reproducible.

## Validation

Offline workspace tests include grouped/1D convolution fixtures, live edits,
copy isolation, save/reload, state provenance, observation accounting, zero-budget
and held-out refusal, malformed graphs, and invalid input shapes/values:

```sh
cargo test --release --workspace
```

The companion `tools/validate_wisp_ocr.py` compares the published model against
CPU ONNX Runtime (development validation only). It checks 22 inputs including
odd/extreme widths, batches up to 16, constants, rendered Thai/English, and three
real-document crops; all 118 intermediate nodes on one rendered input; and a
candidate with individual convolution/head coefficient edits against the same
edits in ONNX. It also verifies saved/reloaded inference and baseline isolation.

This establishes forward implementation parity on the tested inputs, **not**
improved OCR accuracy or recognizer-wide correction.
