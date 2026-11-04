net = mlp([3 2], 'relu');
c = cause('one dose', @(g) net, [1], @(m) 0);
