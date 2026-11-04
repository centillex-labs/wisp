/* Rebuild the post's three hand-written MinAtar networks from their logged
   `edits` answers (converted to Wisp by `wisp edits2w`) and compare them with the
   networks the authors released. Files store W as [in, out]; wload converts. */
games = {'breakout', 'seaquest', 'asterix'};
for i = 0..last(games)
    g = games{i};
    ours = wload(sprintf('out/%s_wisp.safetensors', g));
    post = wload(sprintf('data/%s_reference.npz', g), 'orientation', 'in_out');
    fprintf('%s:\n', g);
    d = wdiff(ours, post);
    assert(d < 1e-6, '%s differs from the released network', g);
end
disp('all three networks match the released weights')
