use sha2::{Digest, Sha256};
use std::{
    path::PathBuf,
    process::{Command, Output},
};
const REPO: &str = "test/qwen3";
const REV: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const LOAD: &str = "m=hf_model('test/qwen3','revision','aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa','offline',true);";
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
struct Fixture(tempfile::TempDir);
impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let fixtures =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../wisp-runtime/tests/data/qwen3");
        for name in ["config.json", "tokenizer.json", "model.safetensors"] {
            let bytes = std::fs::read(fixtures.join(name)).unwrap();
            let cache = dir
                .path()
                .join("cache/huggingface")
                .join(hash(REPO.as_bytes()))
                .join(REV)
                .join(hash(name.as_bytes()));
            std::fs::create_dir_all(&cache).unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                for directory in cache.ancestors().take_while(|p| p.starts_with(dir.path())) {
                    std::fs::set_permissions(directory, std::fs::Permissions::from_mode(0o700))
                        .unwrap();
                }
            }
            let ext = name.rsplit('.').next().unwrap();
            std::fs::write(cache.join(format!("artifact.{ext}")), &bytes).unwrap();
            let receipt = serde_json::json!({"repo":REPO,"file":name,"revision":REV,"sha256":hash(&bytes),"bytes":bytes.len()});
            std::fs::write(
                cache.join("receipt.json"),
                serde_json::to_vec(&receipt).unwrap(),
            )
            .unwrap();
        }
        Self(dir)
    }
    fn run(&self, source: &str, extra: &[&str]) -> Output {
        std::fs::write(self.0.path().join("program.wisp"), source).unwrap();
        Command::new(env!("CARGO_BIN_EXE_wisp"))
            .arg("program.wisp")
            .args(extra)
            .current_dir(self.0.path())
            .env("WISP_MODEL_CACHE", self.0.path().join("cache"))
            .env("HF_HUB_OFFLINE", "1")
            .env_remove("HF_TOKEN")
            .env("PATH", "")
            .output()
            .unwrap()
    }
}
fn output(o: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    )
}

#[test]
fn native_program_loads_observes_edits_roundtrips_and_generates_without_python_or_task_files() {
    let fixture = Fixture::new();
    let source=format!("budget(3); {LOAD}
        P={{'a b'}};
        before=gen(m,P,1,'temperature',0);
        state=capture(m,P);
        head=token_head(m,{{'3'}}); id=head.token_ids[0]; protect('b1');
        wsave(head,'base.safetensors'); head=wload('base.safetensors'); assert(head.token_ids[0]==id);
        lift=max(state.logits)+1-state.logits[0,id];
        head.W1 += lift*state.hidden/sum(state.hidden.^2);
        wsave(head,'edit.safetensors'); stored=wload('edit.safetensors');
        edited=with_head(m,stored); after=gen(edited,P,1,'temperature',0);
        assert(after.texts{{0}}=='3'); assert(budget()==0); disp('native repair passed');");
    let o = fixture.run(&source, &["--log", "events.jsonl"]);
    assert!(o.status.success(), "{}", output(&o));
    let log = std::fs::read_to_string(fixture.0.path().join("events.jsonl")).unwrap();
    let events: Vec<serde_json::Value> = log
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(events.len(), 3);
    assert_eq!(
        events
            .iter()
            .map(|e| e["cost"].as_u64().unwrap())
            .sum::<u64>(),
        3
    );
    assert_eq!(events[1]["call"], "capture");
    let patch: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture.0.path().join("edit.patch.json")).unwrap())
            .unwrap();
    assert_eq!(patch["scope"], "output_rows");
    assert_eq!(patch["parts"]["W1"]["provenance"], "Act");
    assert!(patch["changed_entries"].as_u64().unwrap() > 0);
    // New process: only cached checkpoint + saved edit, no fitting or captured states.
    let o=fixture.run(&format!("budget(1); {LOAD} n=with_head(m,wload('edit.safetensors')); result=gen(n,{{'a b'}},1); assert(result.texts{{0}}=='3');"),&[]);
    assert!(o.status.success(), "{}", output(&o));
}

#[test]
fn native_generation_advances_tokens_and_stops_at_eos() {
    let fixture = Fixture::new();
    let source=format!("budget(3); {LOAD} P={{'a b'}}; state=capture(m,P); original=gen(m,P,2); assert(numel(original.tokens{{0}})==2); head=token_head(m,{{'<|im_end|>'}}); id=head.token_ids[0]; lift=max(state.logits)+1-state.logits[0,id]; head.W1 += lift*state.hidden/sum(state.hidden.^2); edited=with_head(m,head); stopped=gen(edited,P,4); assert(numel(stopped.tokens{{0}})==1); assert(stopped.tokens{{0}}[0]==id); assert(stopped.texts{{0}}==''); assert(budget()==0);");
    let o = fixture.run(&source, &[]);
    assert!(o.status.success(), "{}", output(&o));
}

#[test]
fn native_observations_obey_budget_and_reject_unsupported_requests() {
    let fixture = Fixture::new();
    for (body, code) in [
        ("budget(0); capture(m,{'a b'});", "E-BUDGET"),
        ("m*2;", "E-UNSUPPORTED"),
        ("names(m);", "E-UNSUPPORTED"),
        ("delta(m,m);", "E-UNSUPPORTED"),
        ("budget(0); run(m,{'a b'});", "E-BUDGET"),
        ("budget(0); gen(m,{'a b'},1);", "E-BUDGET"),
        ("gen(m,{'a b'},1,'temperature',1);", "E-ARGS"),
        ("gen(m,{'a b'},0);", "E-ARGS"),
        ("gen(m,{'a b'},65);", "E-ARGS"),
        ("gen(m,{},1);", "E-ARGS"),
        (
            "wsave(m,'not-a-full-checkpoint.safetensors');",
            "E-UNSUPPORTED",
        ),
    ] {
        let o = fixture.run(&format!("{LOAD} {body}"), &[]);
        assert!(!o.status.success(), "{body}");
        assert!(output(&o).contains(code), "{}", output(&o));
    }
    assert!(!fixture
        .0
        .path()
        .join("not-a-full-checkpoint.safetensors")
        .exists());
}

#[test]
fn capture_layers_attn_score_and_writer_edits_work_from_the_language() {
    let fixture = Fixture::new();
    let source = format!(
        "budget(6); {LOAD}
captured = capture(m, {{'a b'}}, 'layers', [0, 1]);
assert(rows(captured.layers) == 2); assert(cols(captured.layers) == cols(captured.hidden));
assert(captured.layer_ids[0] == 0); assert(captured.layer_ids[1] == 1);
attended = capture(m, {{'a b'}}, 'attn', [0]);
assert(rows(attended.attn) == 1); assert(numel(attended.attn[0, :]) > 0);
assert(sum(attended.attn[0, :]) > 0);
parts = read_weights(m, {{'model.layers.0.mlp.down_proj.weight'}});
assert(parts.model_identity == m.identity);
D = parts.('L[0].mlp.down');
D[:, 0] = D[:, 0] .* 2;
edited = with_writers(m, parts);
assert(numel(edited.edited_parts) == 1);
before = run(m, {{'a b'}});
after = run(edited, {{'a b'}});
assert(~isequal(before, after));
fitset = dataset('fit', {{'a b'}});
heldout = dataset('hold', {{'a b'}}, 'heldout', true);
grader = @(s) count(s, '3');
result = score(m, 'hold', 4, grader);
assert(numel(result.scores) == 1);
assert(result.mean == result.scores[0]);
disp('capture-score-writers OK');",
    );
    let o = fixture.run(&source, &[]);
    assert!(o.status.success(), "{}", output(&o));
    assert!(output(&o).contains("capture-score-writers OK"));
    // Held-out data must be rejected by fit-purpose calls.
    let rejected = fixture.run(
        &format!(
            "budget(1); {LOAD} dataset('hold', {{'a b'}}, 'heldout', true); capture(m, 'hold');"
        ),
        &[],
    );
    assert!(!rejected.status.success());
    assert!(
        output(&rejected).contains("E-HELDOUT"),
        "{}",
        output(&rejected)
    );
    let rejected = fixture.run(
        &format!(
            "budget(1); {LOAD} dataset('hold', {{'a b'}}, 'heldout', true); gen(m, 'hold', 2);"
        ),
        &[],
    );
    assert!(
        output(&rejected).contains("E-HELDOUT"),
        "{}",
        output(&rejected)
    );
}
