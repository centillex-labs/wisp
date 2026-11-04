// Run with: wisp observations.wisp
budget(6);
dataset('prompts', {'aab', 'baa'});
dataset('calibration', [0; 1; 2]);
dataset('fresh', [3; 4; 5]);
net = transformer('vocab', {'a','b'}, 'ctx', 8, 'layers', 1, 'heads', 1, 'd', 2, 'ff', 2);
net.embed[0, :] = 1;
net.unembed[1, 0] = 3;

// Teacher-forced next-token log probabilities and final hidden states.
[logp, hidden] = tf(net, 'prompts');
assert(numel(logp) == 2);
assert(rows(hidden{0}) == 3);

// Each cell holds a [layer, head, query, key] tensor.
maps = attn(net, 'prompts');
assert(maps{0}[0, 0, 0, 1] == 0);

// One completion per prompt, two new tokens, with an explicit grader.
g = gen(net, 'prompts', 2, 'temperature', 0, 'seed', 7, 'grade', @(s) count(s, 'b'));
assert(g.texts{0} == 'bb');
assert(g.scores == [2; 2]);

// Paired evaluation on calibration data; confirmation on different, unseen data.
a = mlp([1 1], 'relu');
a.W1[0, 0] = 1;
b = copy(a);
b.b1[0] = 0.25;
metric = @(m, items) run(m, items);
[gain, se] = pair(a, b, 'calibration', metric);
assert(gain == 0.25);
assert(se == 0);
assert(confirm(a, b, 'fresh', metric));
assert(budget() == 0);
disp('tf, attn, gen, pair and fresh-set confirm passed');
