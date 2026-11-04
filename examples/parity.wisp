/* Case 6: a hand-written parity transformer (after Singh's write-up).
   Attention averages over all 11 positions; a value weight of 11 turns the
   average into N, the number of 1s; the MLP computes N - 2*floor(N/2). */

net = transformer('vocab', {'0','1','=','_'}, 'ctx', 11, 'layers', 1, 'heads', 1, 'd', 4, 'ff', 10);
R   = channels({'one', 'is1', 'N', 'p'});
A   = net.L[0].attn;  M = net.L[0].mlp;          // references: edits go into net

net.embed[R.one, :] = 1;                          // every token carries a constant 1
net.embed[R.is1, tok(net, '1')] = 1;

// q = k = 0, so attention is a uniform average over the 11 positions
A.v[R.N, R.is1] = 11;   A.o[R.N, R.N] = 1;        // N = number of 1s

/* p = -2 * #{k : N >= 2k}, so N + p = N mod 2.
   Each step [N >= 2k] = relu(N - 2k + 1) - relu(N - 2k). */
for k = 1..5
    a = 2*k - 2;   b = a + 1;
    M.up[a, R.N] = 1;   M.up_b[a] = -(2*k - 1);   M.down[R.p, a] = -2;
    M.up[b, R.N] = 1;   M.up_b[b] = -2*k;         M.down[R.p, b] =  2;
end

net.unembed[tok(net, '1'), [R.N, R.p]] = 1;                     // logit = N mod 2
net.unembed[tok(net, '0'), [R.one, R.N, R.p]] = [1, -1, -1];    // logit = 1 - that

// ---- test on every bit string of length 1 to 10 (one charged pass)
X = {};  want = [];
for L = 1..10
    for n = 0..2^L - 1
        bits = dec2bin(n, L);
        X += {strcat(repmat('_', 1, 10 - L), bits, '=')};
        want[numel(want)] = mod(count(bits, '1'), 2);
    end
end
Y = run(net, X);                    // logits at the last position, one row per string
[_, k] = max(Y, [], 1);             // 0-based: token 0 is '0', token 1 is '1'
acc = mean(k' == want);
fprintf('parity: %d strings, accuracy %.4f, %d nonzero weights of %d\n', ...
        numel(want), acc, sum(abs([net.embed[:]; net.L[0].attn.v[:]; net.L[0].attn.o[:]; ...
        net.L[0].mlp.up[:]; net.L[0].mlp.up_b[:]; net.L[0].mlp.down[:]; net.unembed[:]]) > 0), numel(net));
assert(acc == 1, 'parity network is wrong');

wsave(net, 'out/parity.safetensors', 'obstacle', 'none: written from scratch');
