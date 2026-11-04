use safetensors::{tensor::TensorView, Dtype};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};
use wisp_runtime::{
    io::{self, LoadOpts, SaveOpts},
    net::Arch,
};

struct Dir(PathBuf);
impl Dir {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let p = std::env::temp_dir().join(format!(
            "wisp-selected-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
    fn path(&self, n: &str) -> String {
        self.0.join(n).to_str().unwrap().to_string()
    }
}
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn options() -> LoadOpts {
    LoadOpts {
        in_out: false,
        act: "relu".into(),
    }
}
const DOWN: &str = "model.language_model.layers.0.mlp.down_proj.weight";
const UP: &str = "model.language_model.layers.0.mlp.up_proj.weight";
fn fixture(path: &str) {
    let down: Vec<u8> = [1f32, 2., 3., 4.]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let up: Vec<u8> = [5f32, 6., 7., 8.]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let unused = [0u8; 16];
    // An unsupported integer tensor is legal in the file but must not be decoded
    // when the user selects only the two floating-point projection tensors.
    let tensors = vec![
        (
            DOWN,
            TensorView::new(Dtype::F32, vec![2, 2], &down).unwrap(),
        ),
        (UP, TensorView::new(Dtype::F32, vec![2, 2], &up).unwrap()),
        (
            "unused",
            TensorView::new(Dtype::I64, vec![2], &unused).unwrap(),
        ),
    ];
    safetensors::serialize_to_file(tensors, &None, std::path::Path::new(path)).unwrap();
}

#[test]
fn selection_skips_unselected_data_and_maps_known_hf_projections() {
    let d = Dir::new();
    let p = d.path("model.safetensors");
    fixture(&p);
    let n = io::load_selected(&p, &options(), &[DOWN.into(), UP.into()]).unwrap();
    assert_eq!(n.parts.len(), 2);
    assert!(matches!(n.arch, Arch::Unknown));
    assert_eq!(n.parts["L[0].mlp.down"].data, vec![1., 2., 3., 4.]);
    assert_eq!(n.parts["L[0].mlp.up"].file_name, UP);
    assert_eq!(n.selection.as_ref().unwrap().len(), 2);
    assert!(io::load(&p, &options()).is_err());
    assert!(io::load_selected(&p, &options(), &["missing".into()]).is_err());
    assert!(io::load_selected(&p, &options(), &[]).is_err());
}

#[test]
fn selected_patch_round_trip_is_explicitly_a_bundle_and_keeps_base_unchanged() {
    let d = Dir::new();
    let p = d.path("model.safetensors");
    fixture(&p);
    let original = fs::read(&p).unwrap();
    let mut n = io::load_selected(&p, &options(), &[DOWN.into(), UP.into()]).unwrap();
    n.parts.get_mut("L[0].mlp.down").unwrap().data[0] = 0.5;
    let out = d.path("edit.safetensors");
    let report = io::save(
        &n,
        &out,
        &SaveOpts {
            obstacle: None,
            preserve: None,
            budget: Some(0.2),
            protect: vec![],
        },
    )
    .unwrap();
    assert_eq!(report.changed_entries, 1);
    let patch: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&report.patch_path).unwrap()).unwrap();
    assert_eq!(patch["scope"], "selected_parts");
    assert_eq!(patch["selected_files"].as_array().unwrap().len(), 2);
    let rebuilt = d.path("rebuilt.safetensors");
    assert!(io::apply_patch(&p, &report.patch_path, &rebuilt)
        .unwrap()
        .contains("hash OK"));
    assert_eq!(fs::read(&p).unwrap(), original);
    assert_eq!(fs::read(&out).unwrap(), fs::read(&rebuilt).unwrap());
    let loaded = io::load(&rebuilt, &options()).unwrap();
    assert!(loaded.selection.is_some());
    assert_eq!(loaded.parts.len(), 2);
}

#[test]
fn selected_loading_validates_unselected_offsets_before_reading_values() {
    let d = Dir::new();
    let p = d.path("bad.safetensors");
    let header = serde_json::to_vec(&serde_json::json!({
        "wanted":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},
        "bad":{"dtype":"F32","shape":[1],"data_offsets":[8,12]}
    }))
    .unwrap();
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend(header);
    file.extend([0u8; 12]);
    fs::write(&p, file).unwrap();
    let error = io::load_selected(&p, &options(), &["wanted".into()])
        .err()
        .unwrap();
    assert!(error.contains("invalid offsets"), "{error}");
}
