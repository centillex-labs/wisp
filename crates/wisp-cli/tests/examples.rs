//! End-to-end tests: run the example programs through the `wisp` binary.

use std::path::PathBuf;
use std::process::Command;

fn examples() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples")
}

fn wisp(args: &[&str]) -> (bool, String) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_wisp"));
    cmd.args(args);
    if args.first() == Some(&"run") {
        cmd.arg("--interpret");
    }
    let out = cmd.current_dir(examples()).output().unwrap();
    let text =
        String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr);
    (out.status.success(), text)
}

fn wisp_in(dir: &str, args: &[&str]) -> (bool, String) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_wisp"));
    cmd.args(args);
    if args.first() == Some(&"run") {
        cmd.arg("--interpret");
    }
    let out = cmd.current_dir(examples().join(dir)).output().unwrap();
    let text =
        String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr);
    (out.status.success(), text)
}

#[test]
fn parity_is_exact_with_one_pass() {
    std::fs::create_dir_all(examples().join("out")).unwrap();
    let (ok, out) = wisp(&["run", "parity.wisp", "--budget", "1"]);
    assert!(ok, "{out}");
    assert!(out.contains("2046 strings, accuracy 1.0000"), "{out}");
    assert!(out.contains("budget: 1 of 1 passes used"), "{out}");
}

#[test]
fn post_networks_rebuild_bit_exactly() {
    std::fs::create_dir_all(examples().join("out")).unwrap();
    for g in ["breakout", "seaquest", "asterix"] {
        let (ok, out) = wisp(&["run", &format!("{g}.wisp")]);
        assert!(ok, "{out}");
    }
    let (ok, out) = wisp(&["run", "compare_with_post.wisp"]);
    assert!(ok, "{out}");
    assert!(
        out.contains("all three networks match the released weights"),
        "{out}"
    );
    assert!(!out.contains("float32-identical: no"), "{out}");
}

#[test]
fn llvm_forward_matches_interpreter() {
    let (ok, out) = wisp(&["run", "llvm_forward.wisp", "--budget", "2"]);
    assert!(ok, "{out}");
    assert!(out.contains("max |difference| 0"), "{out}");
}

#[test]
fn jit_gives_same_results() {
    let (ok1, a) = wisp(&["run", "jit.wisp"]);
    let (ok2, b) = wisp(&["run", "jit.wisp", "--jit"]);
    assert!(ok1 && ok2, "{a}\n{b}");
    assert!(
        b.contains("compiled with LLVM: collatz_steps, land1"),
        "{b}"
    );
    let pick = |s: &str| {
        s.lines()
            .filter(|l| l.starts_with("landing") || l.starts_with("collatz:"))
            .map(|l| l.split(" in ").next().unwrap().to_string())
            .collect::<Vec<_>>()
    };
    assert_eq!(pick(&a), pick(&b));
}

#[test]
fn edits_cases_2_4_5() {
    let (ok, out) = wisp(&["edits.wisp"]);
    assert!(ok, "{out}");
    assert!(
        out.contains("case 2: layer 1: 4 of 32 columns changed"),
        "{out}"
    );
    assert!(out.contains("confirmed"), "{out}");
    assert!(out.contains("budget: 11 of 100 passes used"), "{out}");
}

#[test]
fn patch_round_trip() {
    std::fs::create_dir_all(examples().join("out")).unwrap();
    let (ok, out) = wisp(&["run", "patch_breakout.wisp"]);
    assert!(ok, "{out}");
    let (ok, out) = wisp(&[
        "apply",
        "data/breakout_reference.npz",
        "out/breakout_patched.patch.json",
        "out/rebuilt.safetensors",
    ]);
    assert!(ok && out.contains("hash OK"), "{out}");
    let (ok, out) = wisp(&["run", "verify_rebuilt.wisp"]);
    assert!(ok, "{out}");
}

#[test]
fn diagnostics_have_spec_codes() {
    let cases: &[(&str, &[&str], &str)] = &[
        ("slice.wisp", &[], "E-SLICE-COLON"),
        ("percent.wisp", &[], "E-LEX-PERCENT"),
        ("paren.wisp", &[], "E-PAREN-INDEX"),
        ("end.wisp", &[], "E-END-INDEX"),
        ("range.wisp", &[], "E-INDEX-RANGE"),
        ("undefined.wisp", &[], "E-UNDEFINED"),
        ("probe.wisp", &["--budget", "0"], "E-BUDGET"),
        ("budget.wisp", &["--budget", "0"], "E-BUDGET"),
        ("dose.wisp", &[], "E-SINGLE-DOSE"),
        ("knob.wisp", &[], "E-GLOBAL-KNOB"),
        ("protect.wisp", &[], "E-PROTECTED"),
        ("grad.wisp", &[], "E-FORBIDDEN"),
    ];
    for (file, extra, code) in cases {
        let mut args = vec!["run", file];
        args.extend_from_slice(extra);
        let (ok, out) = wisp_in("errors", &args);
        assert!(!ok, "{file} should fail");
        assert!(
            out.contains(&format!("error[{code}]")),
            "{file}: expected {code}, got:\n{out}"
        );
    }
}
