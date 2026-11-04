//! Regression coverage for single-evaluation assignment targets.
use wisp_runtime::Interp;

fn run(src: &str, passes: u64) -> Interp {
    let program = wisp_syntax::parse(src).unwrap();
    let mut interp = Interp::new("assignment.wisp", src);
    interp.echo = false;
    interp.harness.passes = Some(passes);
    interp.run(&program).unwrap();
    interp
}

const COUNTER: &str = r#"
function i = next(counter)
    i = counter.W1[0];
    counter.W1[0] += 1;
end
function f = field(counter)
    counter.W1[0] += 1;
    f = 'values';
end
counter = mlp([1 1], 'relu');
"#;

#[test]
fn array_targets_evaluate_once_for_every_compound_operator() {
    for (op, expected) in [("+=", 12), ("-=", 8), ("*=", 20), ("/=", 5)] {
        run(&format!("{COUNTER}\nx = [10 30]; x[next(counter)] {op} 2;\nassert(x[0] == {expected}); assert(x[1] == 30); assert(counter.W1[0] == 1);"), 0);
    }
}

#[test]
fn nested_cells_and_dynamic_fields_evaluate_once() {
    run(
        &format!(
            r#"{COUNTER}
s.values = {{[10 30]}};
s.(field(counter)){{0}}[next(counter)-1] += 2;
assert(counter.W1[0] == 2);
assert(s.values{{0}}[0] == 12);
assert(s.values{{0}}[1] == 30);
counter.W1[0] = 0;
c = {{10, 30}};
c{{next(counter)}} += 2;
assert(counter.W1[0] == 1);
assert(c{{0}} == 12); assert(c{{1}} == 30);
c += 40; assert(c{{2}} == 40);
"#
        ),
        0,
    );
}

#[test]
fn part_targets_and_named_indices_evaluate_once() {
    run(
        &format!(
            r#"{COUNTER}
n = mlp([2 2], 'relu');
n.W1 = [10 20; 30 40];
a = n;
a.W1[out=next(counter), in=last] += 2;
assert(counter.W1[0] == 1);
assert(n.W1[0, 1] == 22); assert(n.W1[1, 1] == 40);
n.(field(counter)) = 'metadata';
assert(counter.W1[0] == 2);
"#
        ),
        0,
    );
}

#[test]
fn network_paths_and_whole_part_fields_evaluate_once() {
    run(
        &format!(
            r#"{COUNTER}
n = transformer('vocab', {{'0', '1'}}, 'ctx', 2, 'layers', 2, 'heads', 1, 'd', 2, 'ff', 2);
a = n.L;
a[next(counter)].mlp.down += ones(2, 2);
assert(counter.W1[0] == 1);
assert(n.L[0].mlp.down[:] == ones(4, 1));
assert(n.L[1].mlp.down[:] == zeros(4, 1));
function f = weight(counter)
    counter.W1[0] += 1;
    f = 'W1';
end
m = mlp([2 2], 'relu');
m.(weight(counter)) += ones(2, 2);
assert(counter.W1[0] == 2);
assert(m.W1[:] == ones(4, 1));
"#
        ),
        0,
    );
}

#[test]
fn charged_index_is_observed_only_once() {
    let interp = run(
        r#"
function i = observed(n)
    y = run(n, [0 0]);
    i = 0;
end
n = mlp([2 1], 'relu');
x = [10 20];
x[observed(n)] += 2;
assert(x[0] == 12);
assert(budget() == 0);
"#,
        1,
    );
    assert_eq!(interp.harness.used, 1);
    assert_eq!(interp.harness.log.len(), 1);
}

#[test]
fn assignment_growth_last_and_value_copies_still_work() {
    run(
        r#"
x = [1 2]; y = x;
x[numel(x)] = 3;
x[last] += 4;
assert(x == [1 2 7]); assert(y == [1 2]);
s.a = [4 5]; t = s;
s.a[last] *= 2;
assert(s.a == [4 10]); assert(t.a == [4 5]);
n = mlp([2 2], 'relu'); n.W1 = [1 2; 3 4];
n.W1 += ones(2, 2);
assert(n.W1[:] == [2; 3; 4; 5]);
"#,
        0,
    );
}
