use std::path::{Path, PathBuf};
use wisp_runtime::{
    decoder::Model,
    io::{self, LoadOpts, SaveOpts},
    tensor::Tag,
};

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/qwen3")
}
fn model() -> Model {
    let p = fixture();
    Model::load(
        &p.join("model.safetensors"),
        &p.join("config.json"),
        &p.join("tokenizer.json"),
    )
    .unwrap()
}
fn save_opts() -> SaveOpts {
    SaveOpts {
        obstacle: None,
        preserve: None,
        budget: None,
        protect: vec![],
    }
}
fn load(path: &Path) -> wisp_runtime::net::NetData {
    io::load(
        path.to_str().unwrap(),
        &LoadOpts {
            in_out: false,
            act: "linear".into(),
        },
    )
    .unwrap()
}

#[test]
fn native_qwen3_matches_independent_transformers_gqa_yarn_reference() {
    let model = model();
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture().join("reference.json")).unwrap()).unwrap();
    for case in reference["cases"].as_array().unwrap() {
        let ids = model.encode(case["prompt"].as_str().unwrap()).unwrap();
        assert_eq!(serde_json::to_value(&ids).unwrap(), case["ids"]);
        let (h, l) = model.forward(&ids).unwrap();
        for (actual, field) in [(&h, "hidden"), (&l, "logits")] {
            let expected = case[field].as_array().unwrap();
            assert_eq!(actual.len(), expected.len());
            for (&a, b) in actual.iter().zip(expected) {
                assert!(
                    (a - b.as_f64().unwrap()).abs() < 2e-5,
                    "{field}: {a} vs {b}"
                );
            }
        }
        assert_eq!(
            model.forward(&ids).unwrap(),
            (h, l),
            "no KV state may leak between requests"
        );
    }
}

#[test]
fn saved_output_row_edit_changes_full_vocab_generation_without_changing_decoder_or_base() {
    let model = model();
    let ids = model.encode("a b").unwrap();
    let (h, before) = model.forward(&ids).unwrap();
    let token = ["3", "2"]
        .iter()
        .map(|s| model.token_id(s).unwrap())
        .find(|&id| before[id as usize] < before.iter().copied().fold(f64::NEG_INFINITY, f64::max))
        .unwrap();
    let head = model.token_head(&[token]).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base.safetensors");
    let path = dir.path().join("edit.safetensors");
    io::save(&head, base.to_str().unwrap(), &save_opts()).unwrap();
    let mut head = load(&base);
    let lift =
        before.iter().copied().fold(f64::NEG_INFINITY, f64::max) + 1.0 - before[token as usize];
    let norm = h.iter().map(|x| x * x).sum::<f64>();
    let w = head.parts.get_mut("W1").unwrap();
    for (weight, state) in w.data.iter_mut().zip(&h) {
        *weight += lift * state / norm;
    }
    w.max_tag = Tag::Act;
    let report = io::save(&head, path.to_str().unwrap(), &save_opts()).unwrap();
    assert!(report.changed_entries > 0);
    let saved = load(&path);
    let fixed = model.with_head(&saved).unwrap();
    let (after_h, after) = fixed.forward(&ids).unwrap();
    assert_eq!(h, after_h);
    assert!(after
        .iter()
        .enumerate()
        .all(|(i, &v)| i == token as usize || after[token as usize] > v));
    for i in 0..before.len() {
        if i != token as usize {
            assert_eq!(before[i], after[i]);
        }
    }
    assert_eq!(model.forward(&ids).unwrap(), (h, before));
    let rebuilt = dir.path().join("rebuilt.safetensors");
    io::apply_patch(
        base.to_str().unwrap(),
        &report.patch_path,
        rebuilt.to_str().unwrap(),
    )
    .unwrap();
    assert_eq!(
        std::fs::read(path).unwrap(),
        std::fs::read(&rebuilt).unwrap()
    );
    assert_eq!(load(&rebuilt).head_binding, saved.head_binding);
    let mut patch: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&report.patch_path).unwrap()).unwrap();
    assert_eq!(patch["scope"], "output_rows");
    patch["head_binding"]["model"] = "different model".into();
    std::fs::write(&report.patch_path, serde_json::to_vec(&patch).unwrap()).unwrap();
    assert!(io::apply_patch(
        base.to_str().unwrap(),
        &report.patch_path,
        rebuilt.to_str().unwrap()
    )
    .unwrap_err()
    .contains("binding"));
}

#[test]
fn native_readout_rejects_invalid_bindings_shapes_biases_and_nonfinite_weights() {
    let model = model();
    let token = model.token_id("3").unwrap();
    assert!(model.token_head(&[]).is_err());
    assert!(model.token_head(&[token, token]).is_err());
    assert!(model.token_head(&[model.vocab() as u32]).is_err());
    assert!(model.token_id("a b").is_err());
    for change in 0..7 {
        let mut head = model.token_head(&[token]).unwrap();
        match change {
            0 => head.head_binding = None,
            1 => head.head_binding.as_mut().unwrap().model = "wrong checkpoint".into(),
            2 => head.head_binding.as_mut().unwrap().tokens[0] = 0,
            3 => head.head_binding.as_mut().unwrap().base_sha256 = "wrong rows".into(),
            4 => head.parts.get_mut("b1").unwrap().data[0] = 1.0,
            5 => {
                head.parts.get_mut("W1").unwrap().data.pop();
            }
            6 => head.parts.get_mut("W1").unwrap().data[0] = f64::INFINITY,
            _ => unreachable!(),
        }
        assert!(model.with_head(&head).is_err(), "case {change}");
    }
}

#[test]
fn malformed_native_configuration_and_checkpoint_are_rejected() {
    let p = fixture();
    let dir = tempfile::tempdir().unwrap();
    let original: serde_json::Value =
        serde_json::from_slice(&std::fs::read(p.join("config.json")).unwrap()).unwrap();
    for (key, value) in [
        ("num_key_value_heads", serde_json::json!(0)),
        ("head_dim", serde_json::json!(3)),
        ("attention_bias", serde_json::json!(true)),
        ("model_type", serde_json::json!("other")),
    ] {
        let mut cfg = original.clone();
        cfg[key] = value;
        let path = dir.path().join("config.json");
        std::fs::write(&path, serde_json::to_vec(&cfg).unwrap()).unwrap();
        assert!(Model::load(
            &p.join("model.safetensors"),
            &path,
            &p.join("tokenizer.json")
        )
        .is_err());
    }
    let mut bytes = std::fs::read(p.join("model.safetensors")).unwrap();
    bytes.pop();
    let path = dir.path().join("bad.safetensors");
    std::fs::write(&path, bytes).unwrap();
    assert!(Model::load(&path, &p.join("config.json"), &p.join("tokenizer.json")).is_err());
    let model = model();
    assert!(model.forward(&[]).is_err());
    assert!(model.forward(&[model.vocab() as u32]).is_err());
    assert!(model.encode(&"a ".repeat(model.context())).is_err());
}
