use std::process::Command;

#[test]
fn environment_defaults_and_hf_offline_mode_work_without_python_or_credentials() {
    let dir = std::env::temp_dir().join(format!("wisp-env-cli-{}", std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    let script = dir.join("environment.wisp");
    std::fs::write(&script, "assert(getenv('WISP_TEST_VALUE') == 'private-org/private-model');\nassert(getenv('WISP_TEST_MISSING', 'fallback') == 'fallback');\nassert(getenv('WISP_TEST_EMPTY', 'fallback') == '');\nassert(getenv('WISP_TEST_MISSING') == '');\nassert(budget() == 0);\n").unwrap();
    let invoke = || {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_wisp"));
        cmd.arg("run")
            .arg(&script)
            .args(["--budget", "0"])
            .env("PATH", "")
            .env("WISP_TEST_VALUE", "private-org/private-model")
            .env_remove("WISP_TEST_MISSING")
            .env("WISP_TEST_EMPTY", "")
            .env("HF_TOKEN", "hf_fake_do_not_print_this")
            .env("HF_HUB_OFFLINE", "1")
            .env("HF_HUB_DISABLE_IMPLICIT_TOKEN", "0")
            .env("WISP_MODEL_CACHE", dir.join("cache"));
        cmd.output().unwrap()
    };
    let result = invoke();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    for (source, code, message) in [
        ("getenv('A=B');", "E-ENV", "invalid environment"),
        ("getenv('X', 3);", "E-TYPE", "default"),
        ("getenv('X','a','b');", "E-ARGS", "getenv needs"),
        ("hf_download('org/model','model.safetensors','revision','main');", "E-HF", "pinned"),
        ("hf_download('org/model','model.safetensors','token','not-supported');", "E-ARGS", "unsupported"),
        ("hf_download('org/model','model.safetensors','offline',0,'revision','0123456789abcdef0123456789abcdef01234567');", "E-HF", "offline mode"),
        ("hf_download('org/model','model.safetensors','max_bytes',NaN);", "E-ARGS", "positive exact integer"),
        ("hf_download('org/model','x','revision','a','revision','b');", "E-ARGS", "duplicate"),
    ] {
        std::fs::write(&script, source).unwrap();
        let result = invoke();
        assert!(!result.status.success());
        let output = format!("{}{}", String::from_utf8_lossy(&result.stdout), String::from_utf8_lossy(&result.stderr));
        assert!(output.contains(code) && output.contains(message), "{output}");
        assert!(!output.contains("hf_fake_do_not_print_this"));
    }
    std::fs::remove_dir_all(dir).unwrap();
}
