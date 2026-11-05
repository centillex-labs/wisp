use safetensors::{tensor::TensorView, Dtype};
use wisp_runtime::{
    tensor::{Tag, Tensor},
    value::Value,
    Interp,
};

#[test]
fn real_repair_and_confirmation_programs_pass_on_an_offline_fp16_fixture() {
    struct Dir(std::path::PathBuf);
    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let dir = Dir(std::env::temp_dir().join(format!("wisp-head-repair-{}", std::process::id())));
    std::fs::create_dir(&dir.0).unwrap();
    let base = dir.0.join("head_base.safetensors");
    let candidate = dir.0.join("head_candidate.safetensors");
    let weights: Vec<u8> = [-1f32, 0., 0., 0., 1., 0., 0., 0.]
        .into_iter()
        .flat_map(|x| wisp_runtime::net::f32_to_f16(x).to_le_bytes())
        .collect();
    let bias = [0u8; 4];
    safetensors::serialize_to_file(
        vec![
            (
                "W1",
                TensorView::new(Dtype::F16, vec![2, 4], &weights).unwrap(),
            ),
            ("b1", TensorView::new(Dtype::F16, vec![2], &bias).unwrap()),
        ],
        &None,
        &base,
    )
    .unwrap();
    let r = Value::Arr(Tensor::new(
        vec![12, 4],
        (0..12)
            .flat_map(|i| [-0.1, if i % 2 == 0 { -1. } else { 1. }, 1., 0.])
            .collect(),
        Tag::Lit,
    ));
    let g = Value::Arr(Tensor::new(
        vec![12, 4],
        (0..12)
            .flat_map(|i| {
                [
                    (if i % 2 == 0 { -1. } else { 1. }) * (1. + (i / 2) as f64 / 10.),
                    0.,
                    0.,
                    0.,
                ]
            })
            .collect(),
        Tag::Lit,
    ));
    let fresh = Value::Arr(Tensor::new(
        vec![12, 4],
        (0..12)
            .flat_map(|i| {
                [
                    -0.1,
                    if i % 2 == 0 { -1. } else { 1. },
                    1.,
                    (i + 1) as f64 / 10.,
                ]
            })
            .collect(),
        Tag::Lit,
    ));
    let paths = |src: &str| {
        src.replace("head_base.safetensors", base.to_str().unwrap())
            .replace("head_candidate.safetensors", candidate.to_str().unwrap())
    };
    let mut it = interp(8);
    it.harness
        .register_activations("repair", r.clone())
        .unwrap();
    it.harness
        .register_activations("regression", g.clone())
        .unwrap();
    run(
        &mut it,
        &paths(include_str!(
            "../../../examples/bonsai_behavior_repair.wisp"
        )),
    )
    .unwrap();
    assert_eq!(it.harness.used, 6);
    assert!(it.out.contains("repair: 6/12 -> 12/12"));
    let patch = dir.0.join("head_candidate.patch.json");
    let json: serde_json::Value = serde_json::from_slice(&std::fs::read(&patch).unwrap()).unwrap();
    assert_eq!(json["parts"]["W1"]["provenance"], "Act");
    let rebuilt = dir.0.join("rebuilt.safetensors");
    wisp_runtime::io::apply_patch(
        base.to_str().unwrap(),
        patch.to_str().unwrap(),
        rebuilt.to_str().unwrap(),
    )
    .unwrap();
    assert_eq!(
        std::fs::read(&candidate).unwrap(),
        std::fs::read(rebuilt).unwrap()
    );

    it.harness.passes = Some(10);
    it.harness
        .register_activations("repair_recheck", r)
        .unwrap();
    it.harness
        .register_activations("regression_recheck", g)
        .unwrap();
    it.harness
        .register_activations("confirmation", fresh)
        .unwrap();
    run(
        &mut it,
        &paths(include_str!(
            "../../../examples/bonsai_behavior_confirm.wisp"
        )),
    )
    .unwrap();
    assert_eq!(it.harness.used, 10);
    assert!(it.out.contains("fresh confirmation assertions passed"));
}

fn array() -> Value {
    Value::Arr(Tensor::new(vec![2, 2], vec![1., 2., 3., 4.], Tag::Lit))
}
fn interp(passes: u64) -> Interp {
    let mut it = Interp::new("cache.wisp", "");
    it.echo = false;
    it.harness.passes = Some(passes);
    it
}
fn run(it: &mut Interp, src: &str) -> Result<wisp_runtime::interp::Env, wisp_syntax::Diag> {
    it.src = std::rc::Rc::from(src);
    it.run(&wisp_syntax::parse(src).unwrap())
}

#[test]
fn trusted_cache_is_charged_and_keeps_observation_provenance() {
    let mut it = interp(1);
    it.harness.register_activations("cache", array()).unwrap();
    let env = run(
        &mut it,
        "H = activations('cache'); Y = H + 1; assert(budget() == 0);",
    )
    .unwrap();
    assert_eq!(it.harness.used, 1);
    assert_eq!(it.harness.observed, vec!["cache"]);
    for name in ["H", "Y"] {
        let Value::Arr(a) = &env[name] else {
            panic!("expected tensor")
        };
        assert_eq!(a.tag, Tag::Act);
    }
    it.harness
        .register_dataset("alias", array(), false)
        .unwrap();
    let data = it.harness.resolve_data(&Value::str("alias")).unwrap();
    assert_eq!(
        it.harness.require_fresh(&data).unwrap_err().0,
        "E-REUSED-SET"
    );
}

#[test]
fn cache_authorization_hidden_data_and_budget_are_enforced() {
    let mut it = interp(2);
    it.harness.register_dataset("raw", array(), false).unwrap();
    assert_eq!(
        run(&mut it, "H = activations('raw');").err().unwrap().code,
        "E-FORBIDDEN"
    );
    it.harness.register_activations("secret", array()).unwrap();
    it.harness.test_sets.push("secret".into());
    assert_eq!(
        run(&mut it, "H = activations('secret');")
            .err()
            .unwrap()
            .code,
        "E-FORBIDDEN"
    );
    assert_eq!(it.harness.used, 0);
    assert!(it.harness.observed.is_empty());

    let mut it = interp(0);
    it.harness.register_activations("cache", array()).unwrap();
    assert_eq!(
        run(&mut it, "H = activations('cache');")
            .err()
            .unwrap()
            .code,
        "E-BUDGET"
    );
    let data = it.harness.resolve_data(&Value::str("cache")).unwrap();
    it.harness.require_fresh(&data).unwrap();
    assert_eq!(it.harness.used, 0);
    assert!(it
        .harness
        .register_activations("bad", Value::str("not activations"))
        .is_err());
    assert!(it.harness.register_activations("cache", array()).is_err());
}
