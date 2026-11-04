use std::path::PathBuf;
use std::process::Command;

#[test]
fn trusted_activation_task_registration_is_validated_and_charged() {
    let dir = std::env::temp_dir().join(format!("wisp-activation-cli-{}", std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    std::fs::write(
        dir.join("read.wisp"),
        "H=activations('cache'); assert(H==[1 2;3 4]); assert(budget()==0);",
    )
    .unwrap();
    for (task, expected) in [
        (
            "task.budget=struct('passes',1); task.activations.cache=[1 2;3 4];",
            None,
        ),
        (
            "task.activations.cache='not an array';",
            Some("activation caches must be numeric arrays"),
        ),
        (
            "task.activations=1;",
            Some("task.activations must be a struct"),
        ),
        (
            "task.activations.cache=[1 2;3 4]; task.datasets.cache=[1 2;3 4];",
            Some("already registered"),
        ),
    ] {
        std::fs::write(dir.join("task.wisp"), task).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_wisp"))
            .args([
                "run",
                "read.wisp",
                "--task",
                "task.wisp",
                "--log",
                "log.jsonl",
            ])
            .current_dir(&dir)
            .env("PATH", "")
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        if let Some(message) = expected {
            assert!(!output.status.success());
            assert!(stderr.contains(message), "{stderr}");
        } else {
            assert!(output.status.success(), "{stderr}");
            assert!(stderr.contains("budget: 1 of 1 passes used"));
            let log = std::fs::read_to_string(dir.join("log.jsonl")).unwrap();
            let event: serde_json::Value = serde_json::from_str(log.trim()).unwrap();
            assert_eq!(event["call"], "activations");
            assert_eq!(event["cost"], 1);
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn self_contained_observations_run_without_configuration_files_or_clang() {
    let examples = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples");
    let dir = std::env::temp_dir().join(format!("wisp-observation-cli-{}", std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    let log = dir.join("run.jsonl");
    let output = Command::new(env!("CARGO_BIN_EXE_wisp"))
        .args(["observations.wisp", "--log"])
        .arg(&log)
        .current_dir(examples)
        .env("PATH", "")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout)
        .contains("tf, attn, gen, pair and fresh-set confirm passed"));
    assert!(String::from_utf8_lossy(&output.stderr).contains("budget: 6 of 6 passes used"));
    let events: Vec<serde_json::Value> = std::fs::read_to_string(&log)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(events.len(), 9);
    assert_eq!(
        events
            .iter()
            .map(|v| v["cost"].as_u64().unwrap())
            .sum::<u64>(),
        6
    );
    assert_eq!(events[0]["call"], "tf");
    assert_eq!(events[1]["call"], "attn");
    assert_eq!(events[2]["call"], "gen");
    assert!(!events.iter().any(|v| v["args"] == "hidden_test"));
    std::fs::remove_dir_all(dir).unwrap();
}
