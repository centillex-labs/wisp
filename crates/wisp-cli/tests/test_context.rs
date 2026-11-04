use std::process::{Command, Output};

fn run(source: &str) -> Output {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("context.wisp");
    std::fs::write(&path, source).unwrap();
    Command::new(env!("CARGO_BIN_EXE_wisp"))
        .arg("test")
        .arg(path)
        .output()
        .unwrap()
}

fn text(out: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

#[test]
fn formatted_output_and_bound_methods_in_helpers() {
    let out = run(r#"
function test_print(t)
    t.fprintf('%s: %d\n', t.name, 42);
    print_again(t.fprintf);
end
function print_again(printer)
    printer('from helper\n');
end
"#);
    let s = text(&out);
    assert!(out.status.success(), "{s}");
    assert!(
        s.contains("test_print: 42\nfrom helper\nPASS test_print"),
        "{s}"
    );
}

#[test]
fn errors_and_fail_are_sticky_but_continue_and_do_not_leak() {
    let out = run(r#"
function test_errors(t)
    record_error(t);
    t.error('second error');
    fail = t.fail;
    fail();
    t.fprintf('still running\n');
end
function record_error(t)
    t.error('bad value %d', 42);
end
function test_next(t)
    assert(true);
end
"#);
    let s = text(&out);
    assert!(!out.status.success());
    for expected in [
        "bad value 42",
        "second error",
        "test marked failed",
        "still running",
        "FAIL test_errors",
        "PASS test_next",
        "1 passed, 1 failed, 2 total",
    ] {
        assert!(s.contains(expected), "missing {expected}: {s}");
    }
}

#[test]
fn skip_unwinds_helpers_and_continues_suite() {
    let out = run(r#"
function test_skip(t)
    skip_helper(t);
    error('must not execute');
end
function skip_helper(t)
    t.skip('requires %s', 'GPU');
    error('must not execute either');
end
function test_default(t)
    t.skip();
end
function test_pass(t)
    assert(true);
end
"#);
    let s = text(&out);
    assert!(out.status.success(), "{s}");
    assert!(s.contains("SKIP test_skip: requires GPU"), "{s}");
    assert!(s.contains("SKIP test_default: test skipped"), "{s}");
    assert!(s.contains("1 passed, 0 failed, 2 skipped, 3 total"), "{s}");
    assert!(!s.contains("must not execute"), "{s}");
    assert!(!s.contains("E-TEST-SKIP"), "{s}");
}

#[test]
fn skip_does_not_erase_failures_and_assertions_remain_fatal() {
    let out = run(r#"
function test_fail_skip(t)
    t.fail();
    t.skip('not an escape');
end
function test_error_skip(t)
    t.error('keep this');
    t.skip();
end
function test_assert(t)
    t.error('earlier error');
    assert(false, 'fatal assertion');
    t.fprintf('UNREACHABLE');
end
"#);
    let s = text(&out);
    assert!(!out.status.success());
    for expected in [
        "FAIL test_fail_skip",
        "keep this",
        "earlier error",
        "fatal assertion",
        "0 passed, 3 failed, 3 total",
    ] {
        assert!(s.contains(expected), "missing {expected}: {s}");
    }
    assert!(!s.contains("UNREACHABLE"));
    assert!(!s.contains("SKIP test_"));
}

#[test]
fn validates_method_arguments() {
    for call in [
        "t.fprintf()",
        "t.fprintf(1)",
        "t.error()",
        "t.error(false)",
        "t.fail('message')",
        "t.skip(1)",
    ] {
        let out = run(&format!("function test_invalid(t)\n {call};\nend"));
        let s = text(&out);
        assert!(!out.status.success(), "{call}: {s}");
        assert!(s.contains("E-ARGS"), "{call}: {s}");
        assert!(s.contains("context.wisp:2:"), "{call}: {s}");
    }
}

#[test]
fn all_skipped_is_success_without_claiming_passes() {
    let out = run("function test_skip(t)\n t.skip();\nend");
    assert!(out.status.success(), "{}", text(&out));
    assert!(text(&out).contains("0 passed, 0 failed, 1 skipped, 1 total"));
}
