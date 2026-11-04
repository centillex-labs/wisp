// Whole-program scalar JIT: no interpreter executes this script or its functions.
// Run with: wisp jit native.wisp
function steps = collatz_steps(n)
    steps = 0;
    while n > 1
        if mod(n, 2) == 0
            n /= 2;
        else
            n = 3*n + 1;
        end
        steps += 1;
    end
end

assert(collatz_steps(27) == 111);
total = 0;
for n = 1..10000
    total += collatz_steps(n);
end
assert(total == 849666);
disp('Collatz steps for 1..10000:');
disp(total);
