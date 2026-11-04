use std::process::Command;

#[test]
fn public_help_lists_supported_workflows_without_compiler_development_options() {
    for flag in ["--help", "-h"] {
        let output = Command::new(env!("CARGO_BIN_EXE_wisp"))
            .arg(flag)
            .output()
            .unwrap();
        assert!(output.status.success());
        assert!(output.stderr.is_empty());
        let help = String::from_utf8(output.stdout).unwrap();
        for option in ["wisp <program.wisp>", "--check", "--budget", "--log"] {
            assert!(help.contains(option));
        }
        for hidden in [
            "wisp run",
            "wisp check",
            "wisp apply",
            "wisp edits2w",
            "wisp jit",
            "--jit",
            "emit-llvm",
            "--native",
            "--allow",
            "--task",
        ] {
            assert!(!help.contains(hidden));
        }
    }
    let output = Command::new(env!("CARGO_BIN_EXE_wispc"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(!String::from_utf8(output.stdout).unwrap().contains("--jit"));
}

#[cfg(target_os = "linux")]
#[test]
fn packaged_help_hides_compiler_development_options() {
    let dir = std::env::temp_dir().join(format!("wisp-public-help-{}", std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    std::fs::write(dir.join("hello.wisp"), "disp('hello');").unwrap();
    let build = Command::new(env!("CARGO_BIN_EXE_wispc"))
        .args(["hello.wisp", "-o", "hello"])
        .current_dir(&dir)
        .env("PATH", "")
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let output = Command::new(dir.join("hello"))
        .arg("--help")
        .env("PATH", "")
        .output()
        .unwrap();
    assert!(output.status.success());
    let help = String::from_utf8(output.stdout).unwrap();
    assert!(help.contains("--budget"));
    assert!(!help.contains("--jit"));
    std::fs::remove_dir_all(dir).unwrap();
}
