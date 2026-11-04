/* Compile the post's Breakout network to native code with LLVM and check that
   it computes exactly what the interpreter computes. */
net = wload('data/breakout_reference.npz', 'orientation', 'in_out');
X   = double(rand(2000, 600, 'seed', 7) > 0.97);   // 2000 random sparse grids

tic();  Y0 = run(net, X);  t0 = toc();            // interpreted forward pass
compile(net);
tic();  Y1 = run(net, X);  t1 = toc();            // LLVM-compiled forward pass

err = max(abs(Y1[:] - Y0[:]));
fprintf('2000 forward passes: interpreted %.4f s, native %.4f s (%.1fx); max |difference| %g\n', t0, t1, t0 / t1, err);
assert(err < 1e-9, 'native and interpreted outputs differ');
