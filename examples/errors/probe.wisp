net = mlp([3 2], 'relu');
y = net.W1 * [1; 2; 3];   // a hand-built forward pass
