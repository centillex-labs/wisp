use std::{
    path::PathBuf,
    process::{Command, Output},
    sync::atomic::{AtomicUsize, Ordering},
};

struct Dir(PathBuf);
impl Dir {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "wisp-program-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn write(&self, name: &str, source: &str) {
        std::fs::write(self.0.join(name), source).unwrap();
    }
    fn run(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_wisp"))
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
fn output(result: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    )
}

#[test]
fn self_contained_program_supports_networks_budget_and_log_without_clang() {
    let dir = Dir::new();
    dir.write("model program.wisp", "budget(10); dataset('inputs',[1;2]); n=mlp([1 1],'relu'); y=run(n,'inputs'); assert(rows(y)==2); assert(budget()==0); disp('passed');");
    let result = dir.run(&[
        "model program.wisp",
        "--budget",
        "1",
        "--log",
        "events.jsonl",
    ]);
    assert!(result.status.success(), "{}", output(&result));
    assert!(output(&result).contains("passed"));
    let log = std::fs::read_to_string(dir.0.join("events.jsonl")).unwrap();
    let event: serde_json::Value = serde_json::from_str(log.trim()).unwrap();
    assert_eq!(event["call"], "run");
    assert_eq!(event["cost"], 1);
}

#[test]
fn check_flag_in_either_position_validates_without_execution() {
    let dir = Dir::new();
    dir.write("check.wisp", "error('must not execute');");
    for args in [["check.wisp", "--check"], ["--check", "check.wisp"]] {
        let result = dir.run(&args);
        assert!(result.status.success(), "{}", output(&result));
        assert!(output(&result).contains("check.wisp: OK"));
        assert!(!output(&result).contains("must not execute"));
    }
    dir.write("bad.wisp", "x = [;");
    let result = dir.run(&["bad.wisp", "--check"]);
    assert!(!result.status.success());
    assert!(output(&result).contains("bad.wisp"));
}

#[test]
fn invalid_check_and_execution_arguments_are_rejected_before_execution() {
    let dir = Dir::new();
    dir.write("program.wisp", "disp('must not execute');");
    for args in [
        vec!["--check"],
        vec!["--unknown"],
        vec!["program.wisp", "--check", "--budget", "0"],
        vec!["program.wisp", "--task", "missing.wisp", "--check"],
        vec!["program.wisp", "--task", "missing.wisp"],
        vec!["program.wisp", "--check", "--check"],
        vec!["program.wisp", "--log", "--check"],
        vec!["program.wisp", "--budget"],
        vec!["program.wisp", "--unknown"],
        vec!["program.wisp", "extra.wisp"],
    ] {
        let result = dir.run(&args);
        assert!(!result.status.success(), "{args:?}");
        assert!(!output(&result).contains("must not execute"));
    }
}

#[test]
fn hidden_run_and_check_aliases_remain_compatible() {
    let dir = Dir::new();
    dir.write("program.wisp", "disp('ok');");
    for command in ["run", "check"] {
        let result = dir.run(&[command, "program.wisp"]);
        assert!(result.status.success(), "{}", output(&result));
    }
}
