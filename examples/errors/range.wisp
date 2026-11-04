net = mlp([600 32 6], 'tanh');
net.W1[0, 600] = 1;
