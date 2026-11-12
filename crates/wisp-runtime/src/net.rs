//! Networks: named weight parts with canonical names and an `[out, in]` orientation.

use crate::tensor::{Tag, Tensor};
use crate::value::Value;
use indexmap::IndexMap;
use std::cell::RefCell;
use std::rc::Rc;

pub type NetRef = Rc<RefCell<NetData>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F64,
    BF16,
    F16,
}

impl Dtype {
    pub fn name(self) -> &'static str {
        match self {
            Dtype::F32 => "float32",
            Dtype::F64 => "float64",
            Dtype::BF16 => "bfloat16",
            Dtype::F16 => "float16",
        }
    }
    /// Round a value to what this dtype can store.
    pub fn round(self, v: f64) -> f64 {
        match self {
            Dtype::F64 => v,
            Dtype::F32 => v as f32 as f64,
            Dtype::BF16 => bf16_to_f32(f32_to_bf16(v as f32)) as f64,
            Dtype::F16 => f16_to_f32(f32_to_f16(v as f32)) as f64,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PartData {
    /// Canonical shape: `[out, in]` for matrices, `[n]` for vectors.
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
    pub file_name: String,
    /// True when the file stores this matrix as `[in, out]`.
    pub in_out: bool,
    pub dtype: Dtype,
    /// The values at load time, for patches.
    pub base: Option<Vec<f64>>,
    pub base_sha: Option<String>,
    /// Highest provenance tag among values written into this part.
    pub max_tag: Tag,
}

impl PartData {
    pub fn new(shape: Vec<usize>, file_name: &str) -> Self {
        let n = shape.iter().product();
        PartData {
            shape,
            data: vec![0.0; n],
            file_name: file_name.to_string(),
            in_out: false,
            dtype: Dtype::F32,
            base: None,
            base_sha: None,
            max_tag: Tag::Lit,
        }
    }
    /// The part as a Wisp array (vectors become n×1 columns).
    pub fn tensor(&self) -> Tensor {
        let shape = if self.shape.len() == 1 {
            vec![self.shape[0], 1]
        } else {
            self.shape.clone()
        };
        Tensor::new(shape, self.data.clone(), Tag::W)
    }
}

#[derive(Clone, Debug)]
pub struct TCfg {
    pub vocab: Vec<String>,
    pub ctx: usize,
    pub layers: usize,
    pub heads: usize,
    pub d: usize,
    pub ff: usize,
    pub head_dim: usize,
    pub act: String,
}

#[derive(Clone, Debug)]
pub enum Arch {
    Mlp {
        sizes: Vec<usize>,
        acts: Vec<String>,
    },
    Transformer(TCfg),
    OpenThai2(crate::ocr::Graph),
    Unknown,
}

pub struct NetData {
    pub name: String,
    pub arch: Arch,
    pub parts: IndexMap<String, PartData>,
    pub meta: IndexMap<String, Value>,
    pub source: Option<String>,
    /// Selected file tensor names; saving this net writes a tensor bundle, not a full checkpoint.
    pub selection: Option<Vec<String>>,
    /// A natively compiled forward pass (set by `compile`).
    pub native: Option<Rc<wisp_llvm::jit::NativeForward>>,
    pub decoder: Option<crate::decoder::Model>,
    pub head_binding: Option<crate::decoder::HeadBinding>,
}

impl Clone for NetData {
    fn clone(&self) -> Self {
        NetData {
            name: self.name.clone(),
            arch: self.arch.clone(),
            parts: self.parts.clone(),
            meta: self.meta.clone(),
            source: self.source.clone(),
            selection: self.selection.clone(),
            native: None,
            decoder: self.decoder.clone(),
            head_binding: self.head_binding.clone(),
        }
    }
}

pub enum Resolved {
    Part,
    Prefix,
    Meta(Value),
    Missing,
}

impl NetData {
    pub fn new(name: &str, arch: Arch) -> Self {
        NetData {
            name: name.into(),
            arch,
            parts: IndexMap::new(),
            meta: IndexMap::new(),
            source: None,
            selection: None,
            native: None,
            decoder: None,
            head_binding: None,
        }
    }

    pub fn into_ref(self) -> NetRef {
        Rc::new(RefCell::new(self))
    }

    pub fn resolve(&self, path: &str) -> Resolved {
        if self.parts.contains_key(path) {
            return Resolved::Part;
        }
        let dot = format!("{path}.");
        let br = format!("{path}[");
        if self
            .parts
            .keys()
            .any(|k| k.starts_with(&dot) || k.starts_with(&br))
        {
            return Resolved::Prefix;
        }
        if !path.contains('.') && !path.contains('[') {
            if let Some(v) = self.meta.get(path) {
                return Resolved::Meta(v.clone());
            }
        }
        Resolved::Missing
    }

    /// Indices `i` such that `prefix[i]` exists.
    pub fn sub_indices(&self, prefix: &str) -> Vec<usize> {
        let br = format!("{prefix}[");
        let mut v: Vec<usize> = self
            .parts
            .keys()
            .filter_map(|k| k.strip_prefix(&br))
            .filter_map(|rest| rest.split(']').next())
            .filter_map(|n| n.parse().ok())
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    pub fn snapshot_base(&mut self) {
        for p in self.parts.values_mut() {
            p.base = Some(p.data.clone());
            p.base_sha = Some(crate::io::part_sha(p, &p.data));
        }
    }

    pub fn total_weights(&self) -> usize {
        self.decoder.as_ref().map_or_else(
            || self.parts.values().map(|p| p.data.len()).sum(),
            |m| m.parameter_count(),
        )
    }

    pub fn writers(&self) -> Vec<String> {
        let mut v = Vec::new();
        for k in self.parts.keys() {
            let is = k == "embed"
                || k == "pos"
                || k.ends_with(".attn.o")
                || k.ends_with(".attn.o_b")
                || k.ends_with(".mlp.down")
                || k.ends_with(".mlp.down_b");
            if is {
                v.push(k.clone());
            }
        }
        v
    }

    pub fn readers(&self) -> Vec<String> {
        let mut v = Vec::new();
        for k in self.parts.keys() {
            let is = k == "unembed"
                || k.ends_with(".attn.q")
                || k.ends_with(".attn.k")
                || k.ends_with(".attn.v")
                || k.ends_with(".mlp.up")
                || k.ends_with(".mlp.gate");
            if is {
                v.push(k.clone());
            }
        }
        v
    }

    /// Which axis of a part holds the residual dimension (0 = out, 1 = in).
    pub fn resid_axis(&self, name: &str) -> Option<usize> {
        if self.writers().iter().any(|w| w == name) {
            Some(0)
        } else if self.readers().iter().any(|r| r == name) {
            Some(1)
        } else {
            None
        }
    }
}

pub fn mlp(sizes: &[usize], act: &str) -> NetData {
    let n = sizes.len();
    let mut acts = vec![act.to_string(); n.saturating_sub(2)];
    acts.push("linear".into());
    let mut net = NetData::new(
        "mlp",
        Arch::Mlp {
            sizes: sizes.to_vec(),
            acts,
        },
    );
    for i in 1..n {
        let w = format!("W{i}");
        let b = format!("b{i}");
        net.parts
            .insert(w.clone(), PartData::new(vec![sizes[i], sizes[i - 1]], &w));
        net.parts
            .insert(b.clone(), PartData::new(vec![sizes[i]], &b));
    }
    net.snapshot_base();
    net
}

pub fn transformer(cfg: TCfg, pos: bool) -> NetData {
    let (d, v, ff, ha) = (cfg.d, cfg.vocab.len(), cfg.ff, cfg.heads * cfg.head_dim);
    let mut net = NetData::new("transformer", Arch::Transformer(cfg.clone()));
    let mut add = |name: String, shape: Vec<usize>| {
        net.parts.insert(name.clone(), PartData::new(shape, &name));
    };
    add("embed".into(), vec![d, v]);
    if pos {
        add("pos".into(), vec![d, cfg.ctx]);
    }
    for l in 0..cfg.layers {
        add(format!("L[{l}].attn.q"), vec![ha, d]);
        add(format!("L[{l}].attn.k"), vec![ha, d]);
        add(format!("L[{l}].attn.v"), vec![ha, d]);
        add(format!("L[{l}].attn.o"), vec![d, ha]);
        add(format!("L[{l}].mlp.up"), vec![ff, d]);
        add(format!("L[{l}].mlp.up_b"), vec![ff]);
        add(format!("L[{l}].mlp.down"), vec![d, ff]);
        add(format!("L[{l}].mlp.down_b"), vec![d]);
    }
    add("unembed".into(), vec![v, d]);
    net.snapshot_base();
    net
}

// ---- half-precision conversions (no external crate)

pub fn f32_to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return 0x7fc0;
    }
    let round = ((b >> 16) & 1) + 0x7fff;
    ((b + round) >> 16) as u16
}
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}
pub fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) & 1) as u32;
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as u32;
    let v = if e == 0 {
        (m as f32) * 2f32.powi(-24)
    } else if e == 31 {
        if m == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        (1.0 + m as f32 / 1024.0) * 2f32.powi(e - 15)
    };
    if s == 1 {
        -v
    } else {
        v
    }
}
pub fn f32_to_f16(x: f32) -> u16 {
    // round-to-nearest via f64 arithmetic; adequate for weight files
    let s: u16 = if x.is_sign_negative() { 0x8000 } else { 0 };
    let a = x.abs();
    if a.is_nan() {
        return 0x7e00;
    }
    if a >= 65520.0 {
        return s | 0x7c00;
    }
    if a < 2f32.powi(-24) * 0.5 {
        return s;
    }
    let e = a.log2().floor() as i32;
    if e < -14 {
        let m = (a / 2f32.powi(-24)).round() as u16;
        return s | m;
    }
    let mut m = ((a / 2f32.powi(e) - 1.0) * 1024.0).round() as u32;
    let mut e = e;
    if m == 1024 {
        m = 0;
        e += 1;
    }
    s | (((e + 15) as u16) << 10) | m as u16
}
