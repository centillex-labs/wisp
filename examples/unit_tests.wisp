// Run with: wisp test examples/unit_tests.wisp --budget 1
// Top-level statements are skipped by the test runner.

function test_array_sum(t)
    assert(t.name == 'test_array_sum');
    assert(sum([1, 2, 3]) == 6, 'array sum should be 6');
end

function test_output_weights(t)
    budget(1);
    net = linear_model();
    actual = run(net, [2, 3]);
    assert(actual == 8, 'expected 2*1 + 3*2');
end

function net = linear_model()
    net = mlp([2, 1], 'linear');
    net.W1 = [1, 2];
    net.b1 = [0];
end
