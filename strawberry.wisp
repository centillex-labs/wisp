// Native CPU Bonsai repair: download/cache, observe, edit, save, reload, generate.
budget(3);
model = hf_model('prism-ml/Bonsai-1.7B-unpacked', ...
    'revision', 'a7f720bd688d7563714f3118edd97b83d06f0615', ...
    'sha256', 'cf9a24cbd02e6e257bcfd3177475aaca7f8bd1a63a745441f30d3e40f4313a6b');

// Four repair cases and two controls. These are fitting data, not a holdout.
P = {'Return only the number of occurrences of the character r in the string strawberry.', ...
     'Count the letter ''r'' in ''cherry''. Answer with just the number.', ...
     'Count the letter ''r'' in ''banana''. Answer with just the number.', ...
     'Count the letter ''r'' in ''orange''. Answer with just the number.', ...
     sprintf('Evaluate whether this numerical comparison is correct. Answer with a boolean value only.\nStatement: 9 is larger than 4.'), ...
     sprintf('Evaluate whether this numerical comparison is correct. Answer with a boolean value only.\nStatement: 4 is larger than 9.')};
y = [3; 2; 0; 1];
before = gen(model, P, 1, 'temperature', 0);
state = capture(model, P);
assert(before.texts{0} != '3', 'this checkpoint already answers the target prompt correctly');

// Edit only the four digit output rows; leave the decoder/input embeddings intact.
head = token_head(model, {'0', '1', '2', '3'});
ids = head.token_ids;
protect('b1');
base_path = sprintf('%s/strawberry_base.safetensors', model.cache_dir);
path = sprintf('%s/strawberry_head.safetensors', model.cache_dir);
wsave(head, base_path);
head = wload(base_path);

// Minimum-norm output correction with zero desired drift on both controls.
H = state.hidden;
rhs = zeros(rows(H), 4);
for i = 0..3
    rhs[i, y[i]] = max(state.logits[i, :]) + 1 - state.logits[i, ids[y[i]]];
end
change = H' * ((H*H' + 0.0001*eye(rows(H))) \ rhs);
head.W1 += change';
wsave(head, path);

// Install the SAVED weights and run the full model again, with unrestricted logits.
stored = wload(path);
fixed = with_head(model, stored);
after = gen(fixed, P, 1, 'temperature', 0);
for i = 0..3
    fprintf('%s: %s -> %s\n', P{i}, before.texts{i}, after.texts{i});
    assert(after.texts{i} == sprintf('%d', y[i]));
end
for i = 4..5
    fprintf('control %d: %s -> %s\n', i-3, before.texts{i}, after.texts{i});
    assert(after.texts{i} == before.texts{i}, 'control generation changed');
end
assert(after.texts{0} == '3');
fprintf('Saved output-row edit: %s\n', path);
fprintf('Bonsai generated: %s\n', after.texts{0});
