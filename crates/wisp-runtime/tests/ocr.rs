//! Small offline graph fixture: integration tests do not download OCR weights.
use safetensors::{tensor::TensorView, Dtype};
use serde_json::json;
use std::rc::Rc;
use wisp_runtime::{
    tensor::{Tag, Tensor},
    value::Value,
    Interp,
};

fn fixture(dir: &std::path::Path) -> String {
    let path = dir.join("ocr.safetensors");
    let weights = vec![0u8; 3 * 530 * 4];
    safetensors::serialize_to_file(
        vec![(
            "head",
            TensorView::new(Dtype::F32, vec![3, 530], &weights).unwrap(),
        )],
        &None,
        &path,
    )
    .unwrap();
    let graph = json!({"version":1,"input":"input","output":"prob","weights":{"head":{"shape":[3,530]}},"nodes":[
        {"op":"AveragePool","in":["input"],"out":"pooled","attrs":{"kernel_shape":[48,1],"strides":[48,1]}},
        {"op":"Squeeze","in":["pooled"],"out":"squeezed","attrs":{"axes":[2]}},
        {"op":"Transpose","in":["squeezed"],"out":"features","attrs":{"perm":[0,2,1]}},
        {"op":"MatMul","in":["features","head"],"out":"logits","attrs":{}},
        {"op":"Softmax","in":["logits"],"out":"prob","attrs":{"axis":2}}
    ]});
    std::fs::write(format!("{}.wisp.json",path.display()),json!({"arch":{"kind":"openthai2_rec","graph":graph},"parts":{"head":{"file":"head","in_out":false}}}).to_string()).unwrap();
    path.to_str().unwrap().into()
}
fn interp(passes: u64, heldout: bool, shape: Vec<usize>, fill: f64) -> Interp {
    let mut it = Interp::new("ocr-test.wisp", "");
    it.echo = false;
    it.harness.passes = Some(passes);
    it.harness
        .register_dataset(
            "images",
            Value::Arr(Tensor::new(
                shape.clone(),
                vec![fill; shape.iter().product()],
                Tag::Lit,
            )),
            heldout,
        )
        .unwrap();
    it
}
fn run(it: &mut Interp, src: &str) -> Result<wisp_runtime::interp::Env, wisp_syntax::Diag> {
    it.src = Rc::from(src);
    it.run(&wisp_syntax::parse(src).unwrap())
}
#[test]
fn live_weights_capture_provenance_and_save_reload() {
    let dir = tempfile::tempdir().unwrap();
    let model = fixture(dir.path());
    let candidate = dir.path().join("candidate.safetensors");
    let mut it = interp(4, false, vec![2, 3, 48, 16], 1.);
    let env = run(
        &mut it,
        &format!(
            "n=wload('{model}'); [y,states]=run(n,'images');
        candidate=copy(n); candidate.head[0,1]+=1;
        changed=run(candidate,'images'); assert(norm(reshape(changed-y,1,numel(y)))>0);
        wsave(candidate,'{}','budget',1); stored=wload('{}');
        again=run(stored,'images'); assert(isequal(changed,again));
        original=run(n,'images'); assert(isequal(original,y)); assert(budget()==0);",
            candidate.display(),
            candidate.display()
        ),
    )
    .unwrap();
    let Value::Arr(y) = &env["y"] else { panic!() };
    assert_eq!(y.shape, [2, 16, 530]);
    assert_eq!(y.tag, Tag::Act);
    assert!(y.data.iter().all(|v| (v - 1. / 530.).abs() < 1e-8));
    let Value::Struct(states) = &env["states"] else {
        panic!()
    };
    assert_eq!(states.len(), 5);
    for v in states.values() {
        let Value::Arr(v) = v else { panic!() };
        assert_eq!(v.tag, Tag::Act);
    }
    assert_eq!(it.harness.used, 4);
    assert_eq!(
        it.harness.log.iter().filter(|l| l["call"] == "run").count(),
        4
    );
}
#[test]
fn zero_budget_and_heldout_are_rejected_before_forward() {
    let dir = tempfile::tempdir().unwrap();
    let model = fixture(dir.path());
    let src = format!("n=wload('{model}'); y=run(n,'images');");
    let mut it = interp(0, false, vec![1, 3, 48, 16], 1.);
    assert_eq!(run(&mut it, &src).err().unwrap().code, "E-BUDGET");
    assert_eq!(it.harness.used, 0);
    let mut it = interp(1, true, vec![1, 3, 48, 16], 1.);
    assert_eq!(run(&mut it, &src).err().unwrap().code, "E-HELDOUT");
    assert_eq!(it.harness.used, 0);
}
#[test]
fn malformed_shapes_and_nonfinite_inputs_fail_cleanly() {
    let dir = tempfile::tempdir().unwrap();
    let model = fixture(dir.path());
    let src = format!("n=wload('{model}'); y=run(n,'images');");
    for (shape, fill) in [
        (vec![1, 3, 47, 16], 0.),
        (vec![1, 3, 48, 15], 0.),
        (vec![17, 3, 48, 16], 0.),
        (vec![1, 3, 48, 16], f64::NAN),
    ] {
        let mut it = interp(1, false, shape, fill);
        assert_eq!(run(&mut it, &src).err().unwrap().code, "E-OCR");
    }
}
#[test]
fn malformed_graph_is_rejected_on_load() {
    let dir = tempfile::tempdir().unwrap();
    let model = fixture(dir.path());
    let side = format!("{model}.wisp.json");
    let original: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&side).unwrap()).unwrap();
    for (key, value) in [
        ("op", json!("Resize")),
        ("in", json!(["missing"])),
        ("out", json!("input")),
        ("attrs", json!({"kernel_shape":[48,1],"unknown":true})),
    ] {
        let mut bad = original.clone();
        bad["arch"]["graph"]["nodes"][0][key] = value;
        std::fs::write(&side, bad.to_string()).unwrap();
        let mut it = interp(0, false, vec![1, 3, 48, 16], 0.);
        assert_eq!(
            run(&mut it, &format!("n=wload('{model}');"))
                .err()
                .unwrap()
                .code,
            "E-IO"
        );
    }
}
