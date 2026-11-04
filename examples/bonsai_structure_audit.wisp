// All numerical measurements and the structural decision are made in Wisp.
// The host only verifies/copies original tensor bytes and supplies their metadata.
function [s, column_energy] = inspect_weights(D, matrix)
    v = reshape(D, 1, numel(D));
    row_energy = sum(D.^2, 1);
    column_energy = sum(D.^2, 0);
    [peak, row_id] = max(row_energy);
    s = zeros(1, 21);
    s[0] = numel(D);
    s[1] = sum((v-v) != 0); // NaN and infinity are not finite; no arbitrary cutoff
    s[2] = sum(v == 0);
    s[3] = sum(v > 0);
    s[4] = sum(row_energy);
    s[5] = max(abs(v));
    s[6] = sum(row_energy == 0);
    s[7] = min(row_energy);
    s[8] = peak;
    s[18] = row_id;
    s[19] = min(v);
    s[20] = max(v);
    if matrix
        candidates = [32 64 128 256];
        A = abs(D);
        for j = 0..3
            b = candidates[j];
            assert(mod(cols(D), b) == 0);
            blocks = reshape(A, numel(A)/b, b);
            hi = max(blocks, [], 1);
            lo = min(blocks, [], 1);
            s[9+j] = sum(hi != lo);
            if b == 128
                s[13] = numel(hi);
                s[14] = min(hi);
                s[15] = max(hi);
                positives = sum(reshape(D > 0, numel(D)/b, b), 1);
                s[16] = min(positives);
                s[17] = max(positives);
            end
        end
    end
end

// HOST_MANIFEST
// Required host variables: names, shards, matrix_flags, original_rows,
// expected_weights, layers, checkpoint. No computed weight statistics supplied.

bad_groups = zeros(1,4);
nonfinite = 0;
zero_rows = 0;
visited = 0;
matrix_count = 0;
for t = 0..numel(names)-1
    files = shards{t};
    row_offset = 0;
    for j = 0..numel(files)-1
        n = wload(files{j});
        [chunk, ce] = inspect_weights(n.W1, matrix_flags[t]);
        chunk[18] += row_offset;
        if j == 0
            s = chunk;
            column_energy = ce;
        else
            if chunk[8] > s[8], s[18] = chunk[18]; end
            for k = [0 1 2 3 4 6 9 10 11 12 13], s[k] += chunk[k]; end
            for k = [7 14 16 19], s[k] = min(s[k], chunk[k]); end
            for k = [5 8 15 17 20], s[k] = max(s[k], chunk[k]); end
            column_energy += ce;
        end
        row_offset += rows(n.W1);
    end
    assert(row_offset == original_rows[t]);
    [peak_column, column_id] = max(column_energy);
    concentration = 0;
    if s[4] > 0, concentration = peak_column/s[4]; end
    fprintf('TENSOR,%d', t);
    for x = s, fprintf(',%.17g', x); end
    fprintf(',%d,%.17g\n', column_id, concentration);
    visited += s[0];
    nonfinite += s[1];
    zero_rows += s[6];
    if matrix_flags[t]
        bad_groups += s[9..12];
        matrix_count += 1;
    end
end
assert(visited == expected_weights);
assert(budget() == 0);
fprintf('COVERAGE,%d,%d,%d,%d,%d\n', numel(names), matrix_count, visited, nonfinite, zero_rows);

// Infer the largest universally valid aligned power-of-two group tested, rather
// than assuming a 128-element group from the previous sign-edit example.
learned_group = 0;
candidates = [32 64 128 256];
for j = 0..3
    fprintf('GROUP,%d,%d\n', candidates[j], bad_groups[j]);
    if bad_groups[j] == 0, learned_group = candidates[j]; end
end
fprintf('LEARNED_GROUP,%d\n', learned_group);

// Inspect all coupled SwiGLU paths, not isolated down-projection coefficients.
// |silu(z)| <= |z| gives an input-independent per-channel bound coefficient:
// ||down[:,j]|| * ||gate[j,:]|| * ||up[j,:]|| times ||input||^2.
for layer = 0..layers-1
    prefix = sprintf('model.layers.%d.', layer);
    m = wload(checkpoint, 'parts', {strcat(prefix,'mlp.down_proj.weight'), ...
              strcat(prefix,'mlp.gate_proj.weight'), strcat(prefix,'mlp.up_proj.weight')});
    D = m.L[layer].mlp.down;
    G = m.L[layer].mlp.gate;
    U = m.L[layer].mlp.up;
    assert(rows(G) == rows(U));
    assert(cols(D) == rows(G));
    assert(cols(G) == rows(D));
    gn = sqrt(sum(G.^2,1));
    un = sqrt(sum(U.^2,1));
    dn = sqrt(sum(D.^2,0))';
    bound = gn.*un.*dn;
    dead = sum(bound == 0);
    [top, channel] = max(bound);
    ordered = sort(bound);
    middle = (ordered[floor((numel(ordered)-1)/2)] + ordered[floor(numel(ordered)/2)])/2;
    cosine = sum(G.*U,1) ./ max(gn.*un, 1e-300); // numerical denominator guard, not a repair threshold
    fprintf('MLP,%d,%d,%.17g,%.17g,%d,%.17g,%.17g\n', ...
            layer, dead, top, middle, channel, mean(abs(cosine)), max(abs(cosine)));

    a = wload(checkpoint, 'parts', {strcat(prefix,'self_attn.q_proj.weight'), ...
              strcat(prefix,'self_attn.k_proj.weight'), strcat(prefix,'self_attn.v_proj.weight'), ...
              strcat(prefix,'self_attn.o_proj.weight')});
    // Head dimension is supplied by config, not inferred from quantization groups.
    qh = rows(a.L[layer].attn.q)/head_dim;
    kvh = rows(a.L[layer].attn.v)/head_dim;
    assert(mod(qh,kvh) == 0);
    assert(rows(a.L[layer].attn.k) == rows(a.L[layer].attn.v));
    assert(cols(a.L[layer].attn.o) == qh*head_dim);
    qn = sqrt(sum(reshape(sum(a.L[layer].attn.q.^2,1), qh, head_dim),1));
    kn = sqrt(sum(reshape(sum(a.L[layer].attn.k.^2,1), kvh, head_dim),1));
    vn = sqrt(sum(reshape(sum(a.L[layer].attn.v.^2,1), kvh, head_dim),1));
    on = sqrt(sum(reshape(sum(a.L[layer].attn.o.^2,0), qh, head_dim),1));
    vo = zeros(qh,1);
    for h = 0..qh-1, vo[h] = vn[floor(h/(qh/kvh))]*on[h]; end
    [largest, h] = max(vo);
    fprintf('ATTENTION,%d,%d,%d,%.17g,%.17g,%.17g,%d,%.17g\n', ...
            layer, qh, kvh, min(qn), max(qn), max(kn), h, largest);
end

if nonfinite != 0
    disp('DECISION,QUARANTINE: nonfinite weights require source verification, not guessed replacements');
elseif zero_rows != 0
    disp('DECISION,INVESTIGATE: zero rows are structural findings, not proof of a behavioral defect');
else
    disp('DECISION,KEEP_BASE: no nonfinite weights or zero rows; norm and sign outliers alone do not justify edits');
end
if learned_group > 0
    disp('CONSTRAINT,preserve the measured block-magnitude structure when proposing native-format edits');
end
disp('CAUTION,channel bounds are worst-case coefficients, not observed importance or causal evidence');
disp('CAUTION,Q/K have per-head RMS normalization; raw projection norms do not establish attention strength');
assert(budget() == 0);
