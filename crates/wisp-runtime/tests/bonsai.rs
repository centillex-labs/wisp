//! Offline Bonsai-style FP16 fixtures. Recorded checkpoint-level verification
//! is documented in docs/bonsai-17.md; this suite needs no downloaded model.
use safetensors::{tensor::TensorView, Dtype};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};
use wisp_runtime::{io, Interp};

const NAME: &str = "model.layers.0.mlp.down_proj.weight";
struct Dir(PathBuf);
impl Dir {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "wisp-bonsai-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).unwrap();
        Self(dir)
    }
    fn path(&self, name: &str) -> String {
        self.0.join(name).to_str().unwrap().into()
    }
}
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn fixture(path: &str, invalid: bool) -> Vec<u8> {
    let mut bytes = Vec::new();
    for (block, count) in [96, 90, 80, 70].into_iter().enumerate() {
        let scale = (block + 1) as f32 / 2.0;
        for i in 0..128 {
            let x = if invalid && block == 0 && i == 0 {
                scale * 1.5
            } else if i < count {
                scale
            } else {
                -scale
            };
            bytes.extend_from_slice(&wisp_runtime::net::f32_to_f16(x).to_le_bytes());
        }
    }
    let view = TensorView::new(Dtype::F16, vec![2, 256], &bytes).unwrap();
    safetensors::serialize_to_file(vec![(NAME, view)], &None, std::path::Path::new(path)).unwrap();
    bytes
}
fn interpreter(src: &str) -> Interp {
    let mut it = Interp::new("bonsai-test.wisp", src);
    it.echo = false;
    it.harness.passes = Some(0);
    it.load_functions(
        &wisp_syntax::parse(include_str!("../../../examples/bonsai_17.wisp")).unwrap(),
    );
    it
}

#[test]
fn actual_bonsai_program_helper_preserves_fp16_magnitudes_and_patch_round_trips() {
    let dir = Dir::new();
    let base = dir.path("base.safetensors");
    let out = dir.path("edit.safetensors");
    let before = fixture(&base, false);
    let src=format!("n=wload('{base}','parts','{NAME}'); changed=rebalance_blocks(n.L[0].mlp.down,128,4); assert(changed==8); wsave(n,'{out}'); assert(budget()==0);");
    let mut it = interpreter(&src);
    it.run(&wisp_syntax::parse(&src).unwrap()).unwrap();
    assert!(it.harness.log.is_empty());
    let bytes = fs::read(&out).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let tensor = st.tensor(NAME).unwrap();
    assert_eq!(tensor.dtype(), Dtype::F16);
    let changed: Vec<_> = before
        .chunks_exact(2)
        .zip(tensor.data().chunks_exact(2))
        .enumerate()
        .filter_map(|(i, (a, b))| {
            let (a, b) = (
                u16::from_le_bytes(a.try_into().unwrap()),
                u16::from_le_bytes(b.try_into().unwrap()),
            );
            assert_eq!(a & 0x7fff, b & 0x7fff);
            if a != b {
                assert_eq!(a ^ b, 0x8000);
                Some(i)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(changed, vec![0, 1, 128, 129, 256, 257, 384, 385]);
    let rebuilt = dir.path("rebuilt.safetensors");
    io::apply_patch(&base, &dir.path("edit.patch.json"), &rebuilt).unwrap();
    assert_eq!(fs::read(out).unwrap(), fs::read(rebuilt).unwrap());
    let base_bytes = fs::read(base).unwrap();
    let original = safetensors::SafeTensors::deserialize(&base_bytes).unwrap();
    assert_eq!(before, original.tensor(NAME).unwrap().data());
}

#[test]
fn sign_edit_refuses_a_block_without_equal_magnitude_weights() {
    let dir = Dir::new();
    let base = dir.path("base.safetensors");
    fixture(&base, true);
    let out = dir.path("must-not-exist.safetensors");
    let src=format!("n=wload('{base}','parts','{NAME}'); rebalance_blocks(n.L[0].mlp.down,128,4); wsave(n,'{out}');");
    let mut it = interpreter(&src);
    let error = it.run(&wisp_syntax::parse(&src).unwrap()).err().unwrap();
    assert_eq!(error.code, "E-ASSERT");
    assert!(!std::path::Path::new(&out).exists());
    assert!(it.harness.log.is_empty());
}
