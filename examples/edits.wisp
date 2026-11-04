/* Cases 2, 4 and 5 on a small random transformer, with checks. */
budget(100);
dataset('held_out_b', {'aaaa', 'bbbb', 'cccc', 'dddd', 'acbd', 'bdac'});
net = transformer('vocab', {'a','b','c','d'}, 'ctx', 8, 'layers', 2, 'heads', 2, 'd', 16, 'ff', 32);
nm = names(net);
for i = 0..last(nm)
    p = nm{i};
    net.(p) = 0.3 * randn(size(net.(p)), 'seed', i + 1);   // fill every part
end
disp(net)

// ---- Case 4: remove a direction from every residual writer
r = randn(16, 1, 'seed', 99);  r = r / norm(r);
clean = remove(net, writers(net), r, 1, 'on', 'resid');
W = writers(clean);
worst = 0;
for i = 0..last(W)
    worst = max(worst, max(abs(r' * W[i])));      // r'W must be 0 for every writer
end
fprintf('case 4: max |r''W| over %d writers = %.2e\n', numel(W), worst);
assert(worst < 1e-12);
assert(max(abs(clean.unembed[:] - net.unembed[:])) == 0, 'readers must be untouched');

// ---- Case 2: per-layer directions, only the 4 strongest writers per layer
u = {};
u{0} = randn(16, 1, 'seed', 5);
u{1} = randn(16, 1, 'seed', 6);
q = remove(net, parts(net, 'L[0,1].mlp.down'), u, 6, 'on', 'out', 'top', 4);
for l = 0..1
    changed = sum(any(abs(q.L[l].mlp.down - net.L[l].mlp.down) > 0, 0));
    fprintf('case 2: layer %d: %d of 32 columns changed\n', l, changed);
    assert(changed == 4);
end

// ---- Case 5: task arithmetic
a = copy(net);  a.L[0].mlp.up = a.L[0].mlp.up + 0.1;
b = copy(net);  b.L[1].mlp.up = b.L[1].mlp.up - 0.2;  b.unembed = b.unembed + 9;
tau = delta(a, net, 'except', 'unembed') + delta(b, net, 'except', 'unembed');
new = net + 0.5 * tau;
assert(max(abs(new.L[0].mlp.up[:] - net.L[0].mlp.up[:] - 0.05)) < 1e-12);
assert(max(abs(new.L[1].mlp.up[:] - net.L[1].mlp.up[:] + 0.1)) < 1e-12);
assert(isequal(new.unembed[:], net.unembed[:]), 'excluded parts pass through');
disp('case 5: pre + 0.5 * (tau_a + tau_b) is correct, unembed passed through')

// ---- cause and confirm (the metric observes behaviour through run)
X = {'abcd', 'dcba', 'aabb', 'bbaa', 'abab', 'cdcd'};
score = @(m) -mean(abs(run(m, X)[:, 0]));          // smaller logit for 'a' is better here
cut = @(g) remove(net, writers(net), net.unembed[0, :]', g, 'on', 'resid');
c = cause('the unembedding direction of a drives its logit', cut, [0 0.5 1], score);
fprintf('best dose %g\n', c.best);
// Confirm on the separate dataset declared above.
per_item = @(m, s) -abs(run(m, s)[:, 0])';
ok = confirm(net, cut(c.best), 'held_out_b', per_item);
fprintf('budget left: %g\n', budget());
