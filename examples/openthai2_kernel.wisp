// Native OCR observations are explicit; inspection alone spends no passes.
// WISP_OCR_MODEL must name the exported safetensors with its graph sidecar.
budget(2);
model = wload(getenv('WISP_OCR_MODEL'));
// Submit: inputs=wload('/path/to/input.safetensors');
// Then: [probabilities,states]=run(model,1*inputs.x);
// For weight-only sessions, use budget(0) in a separate setup instead.
