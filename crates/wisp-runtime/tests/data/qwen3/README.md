# Native Qwen3 numerical reference

Synthetic MIT-licensed test data, not a pretrained checkpoint. No download or
Python runtime is needed to run the Rust tests.

- Two layers; width 8; intermediate width 16; two query heads, one KV head;
  head dimension 4; vocabulary 16; YaRN factor 4 and original context 32.
- Generated with Transformers 4.57.6 / torch 2.15.0+cpu, eager attention.
- In `Qwen3ForCausalLM.named_parameters()` order, fill parameters from a torch
  generator seeded with 321: `randn(shape) * 0.1`; add 1 for names containing
  `norm.weight`. Round to FP16 before storing and before reference inference.
- Reference inference uses float32 weights converted from these FP16 values.
- `reference.json` records tokenizer IDs, final hidden states, and every logit
  for two prompts. `tokenizer.json` is a local 16-token WordLevel tokenizer.

Tests cover the native tokenizer/chat format, GQA, Q/K normalization, YaRN,
causal attention, request isolation, output-row edits, and stored-dtype replay.
