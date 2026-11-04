/* Scalar functions compile to native code with `wisp run --jit`. */

// Landing column of a Breakout ball at (row r, column col) moving sideways by s.
function x = land1(col, r, s)
    x = col;  dx = s;
    for step = r..8
        x = x + dx;
        if x < 0 || x > 9, dx = -dx; x = x + 2*dx; end   // wall bounce
    end
end

function y = collatz_steps(n)
    y = 0;
    while n != 1
        if mod(n, 2) == 0
            n = n / 2;
        else
            n = 3*n + 1;
        end
        y = y + 1;
    end
end

tic();
total = 0;
for n = 1..20000
    total = total + collatz_steps(n);
end
fprintf('collatz: %d steps for n = 1..20000 in %.3f s\n', total, toc());

L = zeros(9, 10);
for r = 0..8
    for c = 0..9
        L[r, c] = land1(c, r, 1);
    end
end
fprintf('landing columns from row 0, moving right: %s\n', num2str(L[0, :]));
