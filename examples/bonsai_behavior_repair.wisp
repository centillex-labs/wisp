// Run only via scripts/bonsai_behavior_repair.py fit, after trusted preparation.
// This is a closed-form, supervised two-token head correction, not a gradient update.
// Actual Bonsai final hidden states are charged observations with Act provenance.
base = wload('head_base.safetensors');
R = activations('repair');
G = activations('regression');
y = mod((0..rows(R)-1)', 2);  // false/true pairs, fixed by the case generator
z = mod((0..rows(G)-1)', 2);

before = run(base, 'repair');
[~, predicted] = max(before, [], 1);
assert(any(predicted != y)); // reproduce a real failure, not a made-up repair
before_reg = run(base, 'regression');
[~, predicted_reg] = max(before_reg, [], 1);
assert(all(predicted_reg == z));

// Edit only the two declared output-token rows. A sparse 128-coordinate attempt
// exceeded the norm bound and was rejected; use a minimum-norm two-row correction.
coordinates = 0..cols(R)-1;
H = [R; G];
F = H[:, coordinates];
margin = before[:, 1] - before[:, 0];
residual = [(2*y-1)*0.3 - margin; zeros(rows(G), 1)];

// Ridge minimum-norm correction. Regression examples request zero margin change.
change = F' * ((F*F' + 0.0001*eye(rows(F))) \ residual);
candidate = copy(base);
candidate.W1[1, coordinates] += (change/2)';
candidate.W1[0, coordinates] -= (change/2)';
assert(isequal(candidate.b1, base.b1));
assert(rows(candidate.W1) == 2);
assert(numel(find(candidate.W1 != base.W1)) <= 2*cols(R));
relative = norm(candidate.W1 - base.W1) / norm(base.W1);
fprintf('relative head change: %.6g\n', relative);
assert(relative <= 0.5);

// Validate the actual stored FP16 values, not just float64 arithmetic.
wsave(candidate, 'head_candidate.safetensors', 'budget', 1, ...
      'obstacle', 'repair constrained boolean judgments of character counts', ...
      'preserve', 'integer comparison decisions; fixed regression logit margins');
stored = wload('head_candidate.safetensors');
after = run(stored, 'repair');
[~, repaired] = max(after, [], 1);
assert(all(repaired == y));
assert(all((2*y-1).*(after[:,1]-after[:,0]) >= 0.25));
after_reg = run(stored, 'regression');
[~, preserved] = max(after_reg, [], 1);
assert(all(preserved == z));
regression_drift = max(abs((after_reg[:,1]-after_reg[:,0]) - (before_reg[:,1]-before_reg[:,0])));
fprintf('maximum regression margin drift: %.6g\n', regression_drift);
assert(regression_drift <= 0.05);
fprintf('repair: %d/%d -> %d/%d; regression: %d/%d preserved\n', ...
        sum(predicted==y), rows(R), sum(repaired==y), rows(R), sum(preserved==z), rows(G));
