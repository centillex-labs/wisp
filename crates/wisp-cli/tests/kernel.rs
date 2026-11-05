use std::{
    io::{BufRead, Write},
    process::{Child, Command, Stdio},
};

struct Kernel {
    child: Child,
    stdout: std::io::BufReader<std::process::ChildStdout>,
}

impl Kernel {
    fn start(setup: &str, extra: &[&str]) -> Self {
        Self::start_with("", extra, setup)
    }

    fn start_with(transcript: &str, extra: &[&str], setup: &str) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("setup.wisp");
        std::fs::write(&path, setup).unwrap();
        let mut child = Command::new(env!("CARGO_BIN_EXE_wisp"))
            .arg("kernel")
            .arg(path)
            .args(extra)
            .args(if transcript.is_empty() {
                vec![]
            } else {
                vec!["--transcript".to_string(), transcript.to_string()]
            })
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let stdout = std::io::BufReader::new(child.stdout.take().unwrap());
        // Keep the tempdir alive for the child's lifetime.
        std::mem::forget(dir);
        Self { child, stdout }
    }

    fn send(&mut self, chunk: &str) -> serde_json::Value {
        let stdin = self.child.stdin.as_mut().unwrap();
        stdin.write_all(chunk.as_bytes()).unwrap();
        stdin.write_all(b"\n\n").unwrap();
        stdin.flush().unwrap();
        self.receive()
    }

    fn receive(&mut self) -> serde_json::Value {
        let mut line = String::new();
        self.stdout.read_line(&mut line).unwrap();
        serde_json::from_str(line.trim()).unwrap()
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn kernel_persists_state_across_statements_and_reports_budget() {
    let mut k = Kernel::start("x = 40;", &[]);
    let ready = k.receive();
    assert_eq!(ready["event"], "ready");
    let a = k.send("x += 2;\ndisp(x);");
    assert_eq!(a["ok"], true, "{a}");
    assert_eq!(a["output"], "42\n");
    let b = k.send("y = x * 2;\ndisp(y);");
    assert_eq!(b["output"], "84\n", "{b}");
    let c = k.send("budget(5); disp(budget());");
    assert_eq!(c["output"], "5\n", "{c}");
    assert_eq!(c["budget_remaining"], 5);
    let done = k.send("exit");
    assert_eq!(done["event"], "exit");
}

#[test]
fn kernel_charges_one_budget_across_statements_and_survives_errors() {
    let mut k = Kernel::start(
        "budget(1);\nnet = mlp([1, 1], 'linear');\ndataset('points', [1]);",
        &[],
    );
    assert_eq!(k.receive()["event"], "ready");
    let a = k.send("y = run(net, 'points');");
    assert_eq!(a["ok"], true, "{a}");
    assert_eq!(a["budget_remaining"], 0);
    let b = k.send("z = run(net, 'points');");
    assert_eq!(b["ok"], false, "{b}");
    assert!(b["error"].as_str().unwrap().contains("E-BUDGET"), "{b}");
    // The session continues after the budget error.
    let c = k.send("disp('still alive');");
    assert_eq!(c["ok"], true, "{c}");
    assert_eq!(c["output"], "still alive\n");
    let bad = k.send("this is (not parseable ;");
    assert_eq!(bad["ok"], false);
    let good = k.send("disp(2 + 3);");
    assert_eq!(good["output"], "5\n", "{good}");
}

#[test]
fn kernel_functions_defined_in_one_chunk_are_callable_later() {
    let mut k = Kernel::start("x = 1;", &[]);
    assert_eq!(k.receive()["event"], "ready");
    let defined = k.send("function r = double_it(v)\n    r = 2 * v;\nend");
    assert_eq!(defined["ok"], true, "{defined}");
    let called = k.send("disp(double_it(x));");
    assert_eq!(called["output"], "2\n", "{called}");
    // A runtime error inside a call does not kill the session.
    let failing =
        k.send("function broken(v)\n    assert(false, 'expected failure');\nend\ndisp(broken(x));");
    assert_eq!(failing["ok"], false, "{failing}");
    let after = k.send("disp('recovered');");
    assert_eq!(after["output"], "recovered\n", "{after}");
}

#[test]
fn kernel_log_flag_and_end_of_input() {
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("events.jsonl").display().to_string();
    let mut k = Kernel::start(
        "budget(1);\nnet = mlp([1, 1], 'linear');\ndataset('points', [1]);",
        &["--log", &log],
    );
    assert_eq!(k.receive()["event"], "ready");
    let a = k.send("y = run(net, 'points');");
    assert_eq!(a["ok"], true, "{a}");
    drop(k); // end of input closes stdin and the session
    let events = std::fs::read_to_string(&log).unwrap();
    assert_eq!(events.lines().count(), 1, "{events}");
}

#[test]
fn kernel_records_a_transcript_and_resume_replays_it() {
    let dir = tempfile::tempdir().unwrap();
    let transcript = dir.path().join("session.jsonl").display().to_string();
    {
        let mut k = Kernel::start_with(&transcript, &[], "x = 1;");
        assert_eq!(k.receive()["event"], "ready");
        let a = k.send("y = x + 41;\ndisp(y);");
        assert_eq!(a["output"], "42\n");
        k.send("exit");
    }
    let raw = std::fs::read_to_string(&transcript).unwrap();
    eprintln!("TRANSCRIPT={:?}", raw);
    let lines: Vec<serde_json::Value> = raw
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines[0]["event"], "setup");
    assert_eq!(lines[1]["event"], "chunk");
    assert_eq!(lines[1]["response"]["output"], "42\n");

    // Resume replays the recorded chunks; state and budget carry forward.
    let mut k = Kernel::start_with(&transcript, &["--resume", &transcript], "x = 1;");
    assert_eq!(k.receive()["event"], "ready");
    let next = k.send("disp(y + 1);");
    assert_eq!(next["output"], "43\n", "{}", next);
    k.send("exit");
}

#[test]
fn kernel_resume_refuses_a_different_setup_program() {
    let dir = tempfile::tempdir().unwrap();
    let transcript = dir.path().join("session.jsonl").display().to_string();
    {
        let mut k = Kernel::start_with(&transcript, &[], "x = 1;");
        assert_eq!(k.receive()["event"], "ready");
        k.send("disp(x);");
        k.send("exit");
    }
    let other = dir.path().join("other-setup.wisp");
    std::fs::write(&other, "x = 999;").unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_wisp"))
        .arg("kernel")
        .arg(&other)
        .args(["--resume", &transcript])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("different setup program"));
}
