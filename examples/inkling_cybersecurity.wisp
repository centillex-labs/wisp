/* Rebuild Inkling-2-Mini-Cybersecurity-UD-Q4_K_XL.gguf.

   Merges a rank-1 LoRA into the tensors it targets and writes them into a copy of Unsloth's
   UD-Q4_K_XL file. Every other tensor is copied byte for byte, including the NVFP4 ones Wisp
   cannot decode. Edited tensors keep their type when it is Q8_0, BF16 or F32; otherwise they
   become Q8_0 and their companion '.scale' tensor is dropped.

   Run:  INKLING_SRC=... INKLING_BF16=... INKLING_LORA=... INKLING_OUT=... \
         wisp examples/inkling_cybersecurity.wisp --budget 0
   See docs/gguf.md. */

function yes = has(list, item)
    yes = false;
    for i = 0..last(list)
        if isequal(list{i}, item)
            yes = true;
        end
    end
end

// base + alpha * (B outer A), computed the way numpy does it in float32, then rounded through BF16.
function x = edited(name, base, lora, alpha)
    w = gguf_read(base, name);
    A = gguf_read(lora, strcat(name, '.lora_a'));
    B = gguf_read(lora, strcat(name, '.lora_b'));
    if numel(size(w)) == 3
        d = B .* A;                                  // one outer product per expert
    elseif isequal(name, 'token_embd.weight') && numel(A) == rows(w)
        d = reshape(A, [numel(A), 1]) .* reshape(B, [1, numel(B)]);
    else
        d = reshape(B, [numel(B), 1]) .* reshape(A, [1, numel(A)]);
    end
    d = single(d);
    assert(isequal(size(d), size(w)), 'the LoRA product must have the shape of the weight');
    x = bf16(single(w + single(single(alpha) * d)));
    fprintf('edit %s: relative change %.5f\n', name, norm(d) / norm(w));
end

src  = getenv('INKLING_SRC',  'Inkling-2-Mini-UD-Q4_K_XL.gguf');
base = getenv('INKLING_BF16', 'Inkling-2-Mini-BF16.gguf');
lora = getenv('INKLING_LORA', 'Inkling-2-Mini-Uncensored-LoRA.gguf');
out  = getenv('INKLING_OUT',  'Inkling-2-Mini-Cybersecurity-UD-Q4_K_XL.gguf');
want = getenv('INKLING_SHA256', '2cb1fa7c2a0f6f41b6b7de8a215ab26f1fcbe5c2ada09596a69688ccf6211ca2');

alpha = gguf_meta(lora, 'adapter.lora_alpha');
in_src = gguf_names(src);
in_lora = gguf_names(lora);

targets = {};
types = {};
drop = {};
for i = 0..last(in_lora)
    n = in_lora{i};
    if endsWith(n, '.lora_a')
        t = strrep(n, '.lora_a', '');
        ty = gguf_type(src, t);
        k = numel(targets);
        targets{k} = t;
        if isequal(ty, 'Q8_0') || isequal(ty, 'BF16') || isequal(ty, 'F32')
            types{k} = ty;
        else
            types{k} = 'Q8_0';
            scale = strrep(t, '.weight', '.scale');
            if has(in_src, scale)
                drop{numel(drop)} = scale;
            end
        end
    end
end
fprintf('%d tensors to edit, %d scale tensors to drop, alpha %g\n', numel(targets), numel(drop), alpha);
assert(numel(targets) == 49);
assert(numel(drop) == 20);

r = gguf_rewrite(src, out, targets, types, @(name) edited(name, base, lora, alpha), ...
                 'drop', drop, ...
                 'meta', {'general.name', 'Inkling-2-Mini-Cybersecurity', ...
                          'splice.lora', 'Inkling-2-Mini-Uncensored-LoRA.gguf', ...
                          'splice.src', 'Inkling-2-Mini-UD-Q4_K_XL.gguf'});
assert(r.edited == 49);
assert(r.dropped == 20);
assert(r.copied == 675);
fprintf('sha256 %s\n', r.sha256);
assert(isequal(r.sha256, want), 'output differs from the published file');
disp('MATCH')
assert(budget() == 0);
