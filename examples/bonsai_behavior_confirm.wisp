// The trusted host creates these caches only AFTER freezing and applying the edit.
function scores = accuracy(model, items)
    Y = run(model, items);
    [~, predicted] = max(Y, [], 1);
    expected = mod((0..rows(Y)-1)', 2);
    scores = predicted == expected;
end

base = wload('head_base.safetensors');
stored = wload('head_candidate.safetensors');
repair = run(stored, 'repair_recheck');
[~, labels] = max(repair, [], 1);
assert(all(labels == mod((0..rows(repair)-1)', 2)));

before = run(base, 'regression_recheck');
after = run(stored, 'regression_recheck');
[~, before_labels] = max(before, [], 1);
[~, after_labels] = max(after, [], 1);
assert(all(before_labels == after_labels));
assert(all(after_labels == mod((0..rows(after)-1)', 2)));
assert(max(abs((after[:,1]-after[:,0]) - (before[:,1]-before[:,0]))) <= 0.05);

// First exposure of this dataset in Wisp; host evaluated it only after freezing.
assert(confirm(base, stored, 'confirmation', @accuracy));
assert(budget() == 0);
disp('Stored head repair, regression, and fresh confirmation assertions passed');
