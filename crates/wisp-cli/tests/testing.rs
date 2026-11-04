use std::process::{Command, Output};

fn suite(source: &str, options: &[&str]) -> Output {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("suite.wisp");
    std::fs::write(&path, source).unwrap();
    Command::new(env!("CARGO_BIN_EXE_wisp"))
        .arg("test")
        .arg(path)
        .args(options)
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
fn discovers_in_source_order_with_context_and_helpers_without_running_script() {
    let out = suite(
        r#"
error('top-level must not execute');
function test_z(t)
    assert(t.name == 'test_z');
    assert(t.file != '');
    check_sum(2, 3);
end
function check_sum(a, b)
    assert(a + b == 5, 'helper assertion');
end
function test_a(context)
    assert(context.name == 'test_a');
    assert(all([1, 2] == [1, 2]));
end
function Test_ignored(t)
    error('case-sensitive discovery');
end
"#,
        &[],
    );
    let s = text(&out);
    assert!(out.status.success(), "{s}");
    assert!(s.find("PASS test_z").unwrap() < s.find("PASS test_a").unwrap());
    assert!(s.contains("2 passed, 0 failed, 2 total"), "{s}");
}

#[test]
fn failures_include_source_and_do_not_stop_later_tests() {
    let out = suite("function test_bad(t)\n assert(false, 'expected failure');\nend\nfunction test_error(t)\n error('runtime failure');\nend\nfunction test_good(t)\n assert(true);\nend\n", &[]);
    let s = text(&out);
    assert!(!out.status.success());
    for expected in [
        "FAIL test_bad",
        "E-ASSERT",
        "expected failure",
        "suite.wisp:2:",
        "FAIL test_error",
        "E-USER",
        "PASS test_good",
        "1 passed, 2 failed, 3 total",
    ] {
        assert!(s.contains(expected), "missing {expected}: {s}");
    }
}

#[test]
fn validates_all_signatures_and_duplicates_before_execution() {
    for bad in [
        "function test_bad()\nend",
        "function test_bad(t, extra)\nend",
        "function result = test_bad(t)\n result = 1;\nend",
        "function test_ok(t)\nend",
        "function helper()\nend\nfunction helper()\nend",
    ] {
        let out = suite(
            &format!("function test_ok(t)\n disp('MUST_NOT_RUN');\nend\n{bad}"),
            &[],
        );
        let s = text(&out);
        assert!(!out.status.success(), "{s}");
        assert!(s.contains("E-TEST"), "{s}");
        assert!(!String::from_utf8_lossy(&out.stdout).contains("MUST_NOT_RUN"));
    }
}

#[test]
fn no_tests_and_parse_errors_fail() {
    for src in ["x = 1;", "function helper(t)\nend", "function test_broken("] {
        let out = suite(src, &[]);
        assert!(!out.status.success(), "{}", text(&out));
    }
}

#[test]
fn harness_state_is_fresh_for_each_test() {
    let out = suite(
        r#"
function test_one(t)
    budget(0);
    dataset('inputs', [1, 2]);
    protect('W1');
end
function test_two(t)
    dataset('inputs', [3, 4]);
    net = mlp([2, 1], 'linear');
    net.W1 = [1, 1];
    y = run(net, 'inputs');
    assert(y == 7);
end
"#,
        &["--budget", "1"],
    );
    assert!(out.status.success(), "{}", text(&out));
}

#[test]
fn cli_budget_is_a_ceiling_even_if_test_requests_more() {
    let out = suite(
        r#"
function test_budget(t)
    budget(100);
    net = mlp([1, 1], 'linear');
    y = run(net, [1]);
end
function test_next(t)
    assert(true);
end
"#,
        &["--budget", "0"],
    );
    let s = text(&out);
    assert!(!out.status.success());
    assert!(s.contains("E-BUDGET"), "{s}");
    assert!(s.contains("PASS test_next"), "{s}");
}

#[test]
fn each_test_gets_the_full_observation_budget() {
    let mut source = String::new();
    for name in ["first", "second"] {
        source.push_str(&format!(
            "function test_{name}(t)\n net = mlp([1, 1], 'linear');\n y = run(net, [1]);\nend\n"
        ));
    }
    let out = suite(&source, &["--budget", "1"]);
    assert!(out.status.success(), "{}", text(&out));
    assert!(text(&out).contains("2 passed, 0 failed, 2 total"));
}

#[test]
fn captures_output_and_reports_assertions_in_helpers() {
    let out = suite(
        "function test_helper(t)\n fprintf('from %s', t.name);\n fail_helper();\nend\nfunction fail_helper()\n assert(false, 'bad value %d', 42);\nend",
        &[],
    );
    let s = text(&out);
    assert!(!out.status.success());
    assert!(s.contains("from test_helper\nFAIL test_helper"), "{s}");
    assert!(s.contains("bad value 42"), "{s}");
    assert!(s.contains("suite.wisp:6:"), "{s}");
}

#[test]
fn rejects_invalid_options() {
    for args in [
        vec!["--budget"],
        vec!["--budget", "-1"],
        vec!["--budget", "1.5"],
        vec!["--budget", "18446744073709551616"],
        vec!["--check"],
        vec!["--task", "task.wisp"],
        vec!["--log", "out.jsonl"],
        vec!["other.wisp"],
    ] {
        let out = suite("function test_ok(t)\nend", &args);
        assert!(!out.status.success(), "{args:?}: {}", text(&out));
    }
}

#[test]
fn test_help_and_missing_path() {
    let help = Command::new(env!("CARGO_BIN_EXE_wisp"))
        .args(["test", "--help"])
        .output()
        .unwrap();
    assert!(help.status.success());
    assert!(text(&help).contains("per-test limit"));
    let missing = Command::new(env!("CARGO_BIN_EXE_wisp"))
        .arg("test")
        .output()
        .unwrap();
    assert!(!missing.status.success());
}

#[test]
fn normal_execution_does_not_run_test_functions() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("normal.wisp");
    std::fs::write(
        &path,
        "assert(true);\nfunction test_fail(t)\n assert(false);\nend",
    )
    .unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_wisp"))
        .arg(path)
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", text(&out));
}
