budget(5);
protect('b3');
net = wload('../data/breakout_reference.npz', 'orientation', 'in_out');
net.b3[0] = 1;
wsave(net, '../out/prot.safetensors');
