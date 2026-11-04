//! Standalone packaging tests. Explicit interpreter execution needs no PATH tools.
#![cfg(target_os = "linux")]
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

// Concurrent forks can inherit a writable executable descriptor until exec,
// producing ETXTBSY even after this process closes its own descriptor.
static PROCESS_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct Dir(PathBuf);
impl Dir {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let p = std::env::temp_dir().join(format!(
            "wisp-exec-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&p).unwrap();
        Self(p)
    }
    fn write(&self, name: &str, src: &str) {
        std::fs::write(self.0.join(name), src).unwrap();
    }
    fn build(&self, src: &str, out: &str) -> Output {
        let _guard = PROCESS_LOCK.lock().unwrap();
        Command::new(env!("CARGO_BIN_EXE_wispc"))
            .args([src, "-o", out])
            .current_dir(&self.0)
            .env("PATH", "")
            .output()
            .unwrap()
    }
    fn run(&self, name: &str, args: &[&str]) -> Output {
        let _guard = PROCESS_LOCK.lock().unwrap();
        Command::new(self.0.join(name))
            .args(args)
            .current_dir(&self.0)
            .env("PATH", "")
            .output()
            .unwrap()
    }
}
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn text(o: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    )
}
fn success(o: Output) -> String {
    assert!(o.status.success(), "{}", text(&o));
    text(&o)
}

#[test]
fn executable_is_relocatable_and_source_independent() {
    let d = Dir::new();
    d.write("hello world.wisp", "disp('hello standalone');");
    success(d.build("hello world.wisp", "hello program"));
    std::fs::remove_file(d.0.join("hello world.wisp")).unwrap();
    let moved = Dir::new();
    std::fs::rename(d.0.join("hello program"), moved.0.join("hello")).unwrap();
    assert!(success(moved.run("hello", &["--interpret"])).contains("hello standalone"));
    assert!(success(moved.run("hello", &["--help"])).contains("embedded-runtime"));
}

#[test]
fn parity_executable_runs_without_compiler_and_writes_patch_and_log() {
    let d = Dir::new();
    d.write("parity.wisp", include_str!("../../../examples/parity.wisp"));
    success(d.build("parity.wisp", "parity"));
    // Building parses but never runs the program.
    assert!(!d.0.join("out").exists());
    std::fs::remove_file(d.0.join("parity.wisp")).unwrap();
    std::fs::create_dir(d.0.join("out")).unwrap();
    let output = success(d.run(
        "parity",
        &["--interpret", "--budget", "1", "--log", "out/run.jsonl"],
    ));
    assert!(output.contains("2046 strings, accuracy 1.0000"), "{output}");
    assert!(output.contains("budget: 1 of 1 passes used"), "{output}");
    assert!(d.0.join("out/parity.safetensors").exists());
    assert!(d.0.join("out/parity.patch.json").exists());
    let log = std::fs::read_to_string(d.0.join("out/run.jsonl")).unwrap();
    let events: Vec<serde_json::Value> = log
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0]["cost"], 1);
}

#[test]
fn budget_refusal_and_diagnostics_survive_packaging() {
    let d = Dir::new();
    d.write(
        "observe.wisp",
        "n = mlp([2 1], 'relu');\ny = run(n, [0 0]);\n",
    );
    success(d.build("observe.wisp", "observe"));
    let out = d.run(
        "observe",
        &["--interpret", "--budget", "0", "--log", "refused.jsonl"],
    );
    assert!(!out.status.success());
    assert!(text(&out).contains("E-BUDGET"), "{}", text(&out));
    assert!(text(&out).contains("observe.wisp:2"), "{}", text(&out));
    let log: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(d.0.join("refused.jsonl")).unwrap()).unwrap();
    assert_eq!(log["refused"], true);
    assert_eq!(log["left"], 0);
}

#[test]
fn task_protection_and_gradient_permission_remain_enforced() {
    let d = Dir::new();
    d.write(
        "edit.wisp",
        "n = mlp([2 2], 'relu'); n.W1[0, 0] = 1; wsave(n, 'edited.safetensors');",
    );
    d.write("task.wisp", "task.protect = {'W1'};");
    success(d.build("edit.wisp", "edit"));
    let out = d.run("edit", &["--interpret", "--task", "task.wisp"]);
    assert!(!out.status.success());
    assert!(text(&out).contains("E-PROTECTED"), "{}", text(&out));
    assert!(!d.0.join("edited.safetensors").exists());
    d.write("grad.wisp", "grad_solve(mlp([2 2], 'relu'), 0);");
    success(d.build("grad.wisp", "grad"));
    let out = d.run("grad", &["--interpret"]);
    assert!(!out.status.success());
    assert!(text(&out).contains("E-FORBIDDEN"), "{}", text(&out));
}

#[test]
fn invalid_builds_do_not_create_or_overwrite_output() {
    let d = Dir::new();
    d.write("invalid.wisp", "% wrong comment");
    let out = d.build("invalid.wisp", "invalid");
    assert!(!out.status.success());
    assert!(text(&out).contains("E-LEX-PERCENT"));
    assert!(!d.0.join("invalid").exists());
    d.write("ok.wisp", "disp('ok');");
    d.write("existing", "keep me");
    assert!(!d.build("ok.wisp", "existing").status.success());
    assert_eq!(
        std::fs::read_to_string(d.0.join("existing")).unwrap(),
        "keep me"
    );
    assert!(!d.build("ok.wisp", "ok.wisp").status.success());
    assert_eq!(
        std::fs::read_to_string(d.0.join("ok.wisp")).unwrap(),
        "disp('ok');"
    );
    assert!(!std::fs::read_dir(&d.0).unwrap().any(|e| e
        .unwrap()
        .file_name()
        .to_string_lossy()
        .ends_with(".tmp")));
}

#[test]
fn executable_rejects_unknown_or_incomplete_runtime_flags() {
    let d = Dir::new();
    d.write("ok.wisp", "disp('must not execute');");
    success(d.build("ok.wisp", "ok"));
    for args in [
        vec!["--budget"],
        vec!["--unknown"],
        vec!["--task", "--jit"],
        vec!["--budget", "abc"],
    ] {
        let out = d.run("ok", &args);
        assert!(!out.status.success(), "{args:?}");
        assert!(!text(&out).contains("must not execute"));
    }
}

#[test]
fn corrupt_payload_is_rejected_without_panicking() {
    use std::io::{Seek, SeekFrom, Write};
    let d = Dir::new();
    d.write("ok.wisp", "disp('ok');");
    success(d.build("ok.wisp", "ok"));
    {
        let _guard = PROCESS_LOCK.lock().unwrap();
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .open(d.0.join("ok"))
            .unwrap();
        f.seek(SeekFrom::End(-24)).unwrap();
        f.write_all(&u64::MAX.to_le_bytes()).unwrap();
    }
    let out = d.run("ok", &[]);
    assert!(!out.status.success());
    assert!(text(&out).contains("invalid Wisp executable payload length"));
    assert!(!text(&out).contains("panicked"));
}
