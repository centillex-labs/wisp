// Real Qwen3.5-0.8B weight programming, with NO inference or training.
// Run from the cache directory printed by scripts/fetch_qwen_08.py.
// The 'parts' option reads only these three tensors from the actual checkpoint.
// Saved output is a SELECTED TENSOR BUNDLE, not a complete runnable Qwen model.
net = wload('model.safetensors-00001-of-00001.safetensors', 'parts', {
    'model.language_model.layers.0.mlp.down_proj.weight', ...
    'model.language_model.layers.0.mlp.gate_proj.weight', ...
    'model.language_model.layers.0.mlp.up_proj.weight'
});

D = net.L[0].mlp.down;  // reference to the real [out, in] down projection
assert(rows(D) == 1024);
assert(cols(D) == 3584);
fprintf('Qwen layer 0: down %d x %d, selected weights %d\n', rows(D), cols(D), numel(net));

// Rank the MLP neurons by the squared norm of their residual-writing columns.
// This uses weights only: elementwise arithmetic/reductions cost no observations.
energy = sum(D.^2, 0);
[largest, neurons] = maxk(energy, 4);
disp('Four strongest residual-writing MLP neurons (0-based):');
disp(neurons);
disp('Their squared column norms:');
disp(largest);

// Illustrative sparse edit: halve ONE weight in each selected column.
// This is not a whole-part rescale. No behavioral improvement is claimed.
for j = neurons
    [magnitude, out] = max(abs(D[:, j]));
    before = D[out, j];
    D[out, j] *= 0.5;
    fprintf('down[out=%d, in=%d]: %.9g -> %.9g\n', out, j, before, D[out, j]);
end

wsave(net, 'qwen08_layer0_edit.safetensors', ...
      'budget', 0.000001, ...
      'obstacle', 'illustrative sparse attenuation; no behavioral evaluation', ...
      'preserve', 'all other loaded entries unchanged; unselected checkpoint tensors not included');
assert(budget() == 0);
