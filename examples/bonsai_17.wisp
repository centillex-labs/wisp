// Real Bonsai-1.7B UNPACKED FP16 weights, not packed GGUF/MLX support.
// Run from a directory containing the pinned model.safetensors checkpoint.
// This is a structural sign-edit experiment. It makes no behavioral claim.

function changed = rebalance_blocks(D, block_size, k)
    assert(mod(numel(D), block_size) == 0);
    assert(mod(cols(D), block_size) == 0);
    positive_counts = sum(reshape(D > 0, numel(D)/block_size, block_size), 1);
    [counts, blocks] = maxk(positive_counts, k);
    changed = 0;
    for block_id = blocks
        lo = block_id * block_size;
        hi = lo + block_size - 1;
        before = D[lo..hi];
        scale = max(abs(before));
        assert(scale > 0);
        assert(min(abs(before)) == scale); // verify equal-magnitude +/- scale structure
        assert(sum(before > 0) > block_size/2 + 2);
        positive = find(before > 0);

        // Flip two positive coefficients; preserve their magnitudes and block scale.
        for p = positive[0..1]
            at = lo + p;
            D[at] = -D[at];
            changed += 1;
        end
        after = D[lo..hi];
        assert(abs(after) == abs(before));
        assert(sum(after.^2) == sum(before.^2));
        assert(sum(after > 0) == sum(before > 0) - 2);
        assert(sum(after) == sum(before) - 4*scale);
        fprintf('block %d, output row %d: positive count %d -> %d; scale %.9g unchanged\n', ...
                block_id, floor(lo/cols(D)), sum(before > 0), sum(after > 0), scale);
    end
end

net = wload('model.safetensors', 'parts', 'model.layers.0.mlp.down_proj.weight');
D = net.L[0].mlp.down;
assert(rows(D) == 2048);
assert(cols(D) == 6144);
fprintf('Bonsai layer 0 down projection: %d x %d, %d selected weights\n', rows(D), cols(D), numel(net));

changed = rebalance_blocks(D, 128, 4);
assert(changed == 8);
wsave(net, 'bonsai17_layer0_sign_edit.safetensors', ...
      'budget', 0.000001, ...
      'obstacle', 'illustrative sign rebalance; no inference or behavioral evaluation', ...
      'preserve', 'all coefficient magnitudes and block scales; eight sign flips only; selected tensor bundle');
assert(budget() == 0);
