/* Patch the post's Breakout network and save a hash-keyed patch.
   Rebuild it with:  wisp apply data/breakout_reference.npz out/breakout_patched.patch.json out/rebuilt.safetensors */
net = wload('data/breakout_reference.npz', 'orientation', 'in_out');
before = copy(net);

// A small targeted edit: make the "no-op when positioned" default slightly stronger
// and sharpen gate 0 (illustrative, not from the post).
net.b3[0] = 5;
net.W2[out=0, in=0] = -1200;

wdiff(net, before);
wsave(net, 'out/breakout_patched.safetensors', 'obstacle', 'illustrative edit', 'budget', 0.001);
