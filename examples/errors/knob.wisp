net = wload('../data/breakout_reference.npz', 'orientation', 'in_out');
net.W2 = net.W2 * 2;          // rescales a whole part
wsave(net, '../out/knob.safetensors');
