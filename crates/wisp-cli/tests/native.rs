//! Strict JIT tests: compile the whole script or refuse it before execution.
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};
struct Script(PathBuf);
impl Script {
    fn new(src: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "wisp-native-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&dir).unwrap();
        std::fs::write(dir.join("test.wisp"), src).unwrap();
        Self(dir)
    }
    fn run(&self, command: &str) -> Output {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_wisp"));
        cmd.args([
            if command == "interpret" {
                "run"
            } else {
                command
            },
            "test.wisp",
        ]);
        if command == "interpret" {
            cmd.arg("--interpret");
        }
        cmd.current_dir(&self.0).output().unwrap()
    }
}
impl Drop for Script {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn text(out: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}
fn success(out: Output) -> String {
    assert!(out.status.success(), "{}", text(&out));
    String::from_utf8(out.stdout).unwrap()
}

#[test]
fn whole_collatz_program_matches_interpreter() {
    let s = Script::new(include_str!("../../../examples/native.wisp"));
    let native = s.run("jit");
    assert!(native.stderr.is_empty());
    assert_eq!(success(native), success(s.run("interpret")));
}

#[test]
fn control_flow_function_calls_and_short_circuit_are_native() {
    let s = Script::new(
        r#"
function y = failed()
    assert(false);
    y = 0;
end
function y = twice(x)
    y = x * 2;
end
function y = nested(x)
    y = twice(x) + 1;
    return;
    y = 100;
end
assert(true || failed());
assert(!(false && failed()));
assert(nested(3) == 7);
x = 0;
for i = 5..1 by -1
    if i == 4, continue; elseif i == 2, break; else, x += i; end
end
assert(x == 8);
i = 0;
while i < 4
    i += 1;
    if i == 2, continue; end
    x += i;
end
assert(x == 16);
for a = 0..2
    for b = 0..2, x += 1; end
end
assert(x == 25);
assert(mod(-5, 3) == 1);
assert(-2^2 == -4);
assert(2^3^2 == 512);
assert(abs(sqrt(9) - 3) < 0.00001);
assert(min(1, 2) == 1 && max(1, 2) == 2);
assert(sigmoid(0) == 0.5);
disp('native 100% works');
disp(x);
return;
assert(false);
"#,
    );
    assert_eq!(success(s.run("jit")), success(s.run("interpret")));
}

#[test]
fn unsupported_programs_are_rejected_before_any_execution() {
    for src in [
        "disp('SHOULD_NOT_RUN'); n = mlp([2 1], 'relu');",
        "disp('SHOULD_NOT_RUN'); x = [1 2];",
        "disp('SHOULD_NOT_RUN'); x = budget();",
        "disp('SHOULD_NOT_RUN'); grad_solve(1, 2);",
        "function y = f(x), y = f(x); end; disp('SHOULD_NOT_RUN'); x = f(1);",
        "function [a,b] = f(x), a=x; b=x; end; disp('SHOULD_NOT_RUN');",
        "disp('SHOULD_NOT_RUN'); c = {'x'};",
        "disp('SHOULD_NOT_RUN'); break;",
        "disp('SHOULD_NOT_RUN'); disp = 1; disp(1);",
    ] {
        let s = Script::new(src);
        let out = s.run("jit");
        assert!(!out.status.success(), "{src}");
        assert!(out.stdout.is_empty(), "{}", text(&out));
        assert!(text(&out).contains("E-JIT-UNSUPPORTED"), "{}", text(&out));
    }
}

#[test]
fn wrong_function_arity_is_rejected_before_codegen() {
    for src in [
        "function y = f(x), y=x; end; f();",
        "function y = f(x), y=x; end; f(1, 2);",
        "sigmoid();",
        "min(1);",
        "max(1, 2, 3);",
    ] {
        let out = Script::new(src).run("jit");
        assert!(!out.status.success());
        assert!(text(&out).contains("E-ARGS"), "{}", text(&out));
        assert!(!text(&out).contains("clang failed"));
    }
}

#[test]
fn native_runtime_errors_have_codes_and_source_spans() {
    for (src, code) in [
        ("assert(false);\ndisp('AFTER');", "E-ASSERT"),
        (
            "if false, x = 1; end;\ndisp(x); disp('AFTER');",
            "E-UNDEFINED",
        ),
        ("x += 1; disp('AFTER');", "E-UNDEFINED"),
        (
            "for i = 2..1, x=1; end; disp(i); disp('AFTER');",
            "E-UNDEFINED",
        ),
        (
            "function y=f(), if false, y=1; end; end; f(); disp('AFTER');",
            "E-UNDEFINED",
        ),
        ("for i = 0..2 by 0, disp(i); end; disp('AFTER');", "E-SHAPE"),
        ("for i = 0..(1/0), disp(i); end; disp('AFTER');", "E-SHAPE"),
        ("for i = 0..(0/0), disp(i); end; disp('AFTER');", "E-SHAPE"),
    ] {
        let out = Script::new(src).run("jit");
        assert!(!out.status.success(), "{src}");
        assert!(
            text(&out).contains(&format!("error[{code}]")),
            "{}",
            text(&out)
        );
        assert!(text(&out).contains("test.wisp:"));
        assert!(!String::from_utf8_lossy(&out.stdout).contains("AFTER"));
    }
}

#[test]
fn failures_propagate_through_native_calls_without_losing_location() {
    let out = Script::new("function y = bad()\n    assert(false);\n    y=1;\nend\nfunction y=outer(), y=bad()+1; end\nx=outer(); disp('AFTER');").run("jit");
    assert!(!out.status.success());
    assert!(text(&out).contains("test.wisp:2:"), "{}", text(&out));
    assert!(out.stdout.is_empty());
}

#[test]
fn missing_clang_is_an_error_not_an_interpreter_fallback() {
    let s = Script::new("disp('SHOULD_NOT_RUN');");
    let out = Command::new(env!("CARGO_BIN_EXE_wisp"))
        .args(["jit", "test.wisp"])
        .env("WISP_CLANG", s.0.join("missing-clang"))
        .env("TMPDIR", &s.0)
        .current_dir(&s.0)
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(out.stdout.is_empty());
    assert!(text(&out).contains("could not run clang"));
    assert_eq!(
        std::fs::read_dir(&s.0).unwrap().count(),
        1,
        "failed compilation must clean up temporary files"
    );
}

#[test]
fn whole_script_ir_can_be_inspected_without_running_or_clang() {
    let s = Script::new("x = 3; assert(x == 3); disp(x);");
    let out = Command::new(env!("CARGO_BIN_EXE_wisp"))
        .args(["emit-llvm", "test.wisp", "--native"])
        .env("WISP_CLANG", s.0.join("missing-clang"))
        .current_dir(&s.0)
        .output()
        .unwrap();
    let ir = success(out);
    assert!(ir.contains("define void @wisp_native_entry"));
    assert!(ir.contains("define double @wisp___wisp_script_impl"));
    assert!(ir.contains("call i32 (ptr, ...) @printf"));
}

#[test]
#[cfg(target_os = "linux")]
fn packaged_executable_is_native_by_default() {
    let s = Script::new(include_str!("../../../examples/native.wisp"));
    let out = Command::new(env!("CARGO_BIN_EXE_wispc"))
        .args(["test.wisp", "-o", "native"])
        .current_dir(&s.0)
        .output()
        .unwrap();
    success(out);
    std::fs::remove_file(s.0.join("test.wisp")).unwrap();
    let out = Command::new(s.0.join("native")).output().unwrap();
    assert!(out.stderr.is_empty());
    assert!(success(out).contains("849666"));
    let out = Command::new(s.0.join("native"))
        .args(["--task", "missing.wisp"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(text(&out).contains("use --interpret explicitly"));
}
