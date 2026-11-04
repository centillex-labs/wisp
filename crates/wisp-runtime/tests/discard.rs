use wisp_runtime::Interp;

#[test]
fn underscore_discards_builtin_outputs_without_binding_a_variable() {
    let src = "[_, index] = max([3, 9, 4]); assert(index == 1);";
    let program = wisp_syntax::parse(src).unwrap();
    let mut it = Interp::new("discard.wisp", src);
    let env = it.run(&program).unwrap();
    assert!(!env.contains_key("_"));
}

#[test]
fn discards_preserve_output_positions_and_evaluate_once() {
    let src = r#"
function [a, b, c] = outputs(counter)
    counter.W1[0] += 1;
    a = 10;
    b = 20;
    c = 30;
end
counter = mlp([1, 1], 'linear');
[_, middle, _] = outputs(counter);
assert(middle == 20);
assert(counter.W1[0] == 1);
[_, _, _] = outputs(counter);
assert(counter.W1[0] == 2);
[_named, ~, last_value] = outputs(counter);
assert(_named == 10);
assert(last_value == 30);
assert(counter.W1[0] == 3);
"#;
    let program = wisp_syntax::parse(src).unwrap();
    let mut it = Interp::new("discard.wisp", src);
    it.run(&program).unwrap();
}

#[test]
fn discarding_observations_still_charges_budget() {
    let src = r#"
budget(1);
net = mlp([1, 1], 'linear');
[_] = run(net, [1]);
assert(budget() == 0);
"#;
    let program = wisp_syntax::parse(src).unwrap();
    let mut it = Interp::new("discard.wisp", src);
    it.run(&program).unwrap();
    assert_eq!(it.harness.used, 1);
}

#[test]
fn discarded_outputs_do_not_hide_runtime_errors_or_missing_results() {
    for (src, expected) in [
        ("[_, _] = zeros(1);", "E-ARGS"),
        ("[_] = error('still fails');", "E-USER"),
    ] {
        let program = wisp_syntax::parse(src).unwrap();
        let mut it = Interp::new("discard.wisp", src);
        let error = it.run(&program).err().expect("must fail");
        assert_eq!(error.code, expected);
    }
}

#[test]
fn tilde_remains_compatible_and_logical_negation_is_unchanged() {
    let src = "[~, index] = max([3, 9, 4]); assert(index == 1); assert(~false);";
    let program = wisp_syntax::parse(src).unwrap();
    let mut it = Interp::new("discard.wisp", src);
    it.run(&program).unwrap();
}
