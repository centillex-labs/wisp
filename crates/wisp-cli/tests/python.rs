#![cfg(feature = "python")]
use std::io::Write;
use std::process::{Command, Stdio};

fn wisp() -> Command {
    Command::new(env!("CARGO_BIN_EXE_wisp"))
}

#[test]
fn rejects_bad_arguments_and_oversize_source() {
    let result = wisp().args(["python"]).output().unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("usage:"));
    let result = wisp()
        .args(["python", "example.py", "--worker", "/obsolete"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    let mut source = tempfile::NamedTempFile::new().unwrap();
    source
        .write_all(&vec![b' '; wisp_python::MAX_SOURCE_BYTES + 1])
        .unwrap();
    let result = wisp()
        .args(["python", source.path().to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("source limit exceeded"));
}

#[test]
fn self_contained_example_and_persistent_kernel() {
    let example = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../examples/restricted_python.py"
    );
    for args in [vec!["python", example], vec![example]] {
        // No PATH, Python home, external runtime, or library installation.
        let result = wisp()
            .args(args)
            .env_clear()
            .current_dir("/")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&result.stdout),
            "[14.0, 20.0, 30.0, 44.0]\n"
        );
    }
    let mut child = wisp()
        .arg("python-kernel")
        .env_clear()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    for code in ["a = tensor([2], 1, 1)", "print((a @ a).values())"] {
        writeln!(stdin, "{}", serde_json::to_string(code).unwrap()).unwrap();
    }
    drop(stdin);
    let result = child.wait_with_output().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let responses = String::from_utf8(result.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(responses.len(), 2);
    assert_eq!(responses[0]["ok"], true);
    assert_eq!(responses[1]["output"], "[4.0]\n");
}

#[test]
fn scripts_cannot_read_or_modify_host_files() {
    let directory = tempfile::tempdir().unwrap();
    let secret = directory.path().join("secret.txt");
    std::fs::write(&secret, "host-only-secret").unwrap();
    let path = serde_json::to_string(secret.to_str().unwrap()).unwrap();
    for code in [format!("print(open({path}))"), format!("open({path}, 'w')")] {
        let source = directory.path().join("script.py");
        std::fs::write(&source, code).unwrap();
        let result = wisp()
            .args(["python", source.to_str().unwrap()])
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert!(!String::from_utf8_lossy(&result.stdout).contains("host-only-secret"));
        assert_eq!(
            std::fs::read_to_string(&secret).unwrap(),
            "host-only-secret"
        );
    }
}
