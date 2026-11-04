a = wload('out/breakout_patched.safetensors');
b = wload('out/rebuilt.safetensors');
assert(wdiff(a, b) == 0, 'rebuilt model differs');
disp('rebuilt model is identical to the patched model')
