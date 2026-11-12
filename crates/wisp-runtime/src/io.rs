//! Weight files (.safetensors, .npz), the metadata sidecar, and hash-keyed patches.

use crate::net::*;
use crate::tensor::Tag;
use serde_json::{json, Value as J};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

type LoadedTensor = (String, Vec<usize>, Dtype, Vec<f64>);

pub struct LoadOpts {
    pub in_out: bool,
    pub act: String,
}

/// Values of a part laid out as in its file (transposed back if the file is `[in, out]`).
pub fn file_layout(p: &PartData, data: &[f64]) -> (Vec<usize>, Vec<f64>) {
    if p.in_out && p.shape.len() == 2 {
        let (o, i) = (p.shape[0], p.shape[1]);
        let mut d = vec![0.0; o * i];
        for r in 0..o {
            for c in 0..i {
                d[c * o + r] = data[r * i + c];
            }
        }
        (vec![i, o], d)
    } else {
        (p.shape.clone(), data.to_vec())
    }
}

pub fn encode(dtype: Dtype, data: &[f64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    for &v in data {
        match dtype {
            Dtype::F32 => out.extend_from_slice(&(v as f32).to_le_bytes()),
            Dtype::F64 => out.extend_from_slice(&v.to_le_bytes()),
            Dtype::BF16 => out.extend_from_slice(&f32_to_bf16(v as f32).to_le_bytes()),
            Dtype::F16 => out.extend_from_slice(&f32_to_f16(v as f32).to_le_bytes()),
        }
    }
    out
}

fn decode(dtype: Dtype, bytes: &[u8]) -> Vec<f64> {
    match dtype {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64)
            .collect(),
        Dtype::F64 => bytes
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        Dtype::BF16 => bytes
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])) as f64)
            .collect(),
        Dtype::F16 => bytes
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])) as f64)
            .collect(),
    }
}

pub fn sha_hex(bytes: &[u8]) -> String {
    let h = Sha256::digest(bytes);
    h.iter().map(|b| format!("{b:02x}")).collect()
}

/// SHA-256 of a part's bytes as they are stored in its file (layout and dtype).
pub fn part_sha(p: &PartData, data: &[f64]) -> String {
    let (_, d) = file_layout(p, data);
    sha_hex(&encode(p.dtype, &d))
}

fn transpose_in(shape: &[usize], data: Vec<f64>) -> (Vec<usize>, Vec<f64>) {
    // file [in, out] -> canonical [out, in]
    let (i, o) = (shape[0], shape[1]);
    let mut d = vec![0.0; i * o];
    for r in 0..i {
        for c in 0..o {
            d[c * i + r] = data[r * o + c];
        }
    }
    (vec![o, i], d)
}

pub fn load(path: &str, opts: &LoadOpts) -> Result<NetData, String> {
    load_filtered(path, opts, None)
}

/// Load exact file tensor names without decoding the rest of a safetensors checkpoint.
pub fn load_selected(path: &str, opts: &LoadOpts, names: &[String]) -> Result<NetData, String> {
    if names.is_empty() {
        return Err("'parts' needs at least one file tensor name".into());
    }
    if !path.ends_with(".safetensors") {
        return Err("selective loading currently requires safetensors".into());
    }
    load_filtered(path, opts, Some(names))
}

fn load_filtered(
    path: &str,
    opts: &LoadOpts,
    selected: Option<&[String]>,
) -> Result<NetData, String> {
    let raw: Vec<(String, Vec<usize>, Dtype, Vec<f64>)> = if path.ends_with(".npz") {
        load_npz(path)?
    } else if path.ends_with(".safetensors") {
        load_safetensors(path, selected)?
    } else {
        return Err(format!("{path}: expected a .safetensors or .npz file"));
    };
    let sidecar = sidecar_path(path);
    let side: Option<J> = fs::read_to_string(&sidecar)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok());
    let mut net = NetData::new(
        Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("net"),
        Arch::Unknown,
    );
    net.source = Some(path.to_string());
    net.selection = selected.map(|s| s.to_vec()).or_else(|| {
        side.as_ref()
            .and_then(|s| s["selected_files"].as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
    });
    let mut file_to_canon: HashMap<String, (String, bool)> = HashMap::new();
    if let Some(s) = &side {
        if let Some(parts) = s.get("parts").and_then(|p| p.as_object()) {
            for (canon, info) in parts {
                let file = info
                    .get("file")
                    .and_then(|f| f.as_str())
                    .unwrap_or(canon)
                    .to_string();
                let io = info
                    .get("in_out")
                    .and_then(|b| b.as_bool())
                    .unwrap_or(false);
                file_to_canon.insert(file, (canon.clone(), io));
            }
        }
    }
    let hf: HashMap<String, String> = raw
        .iter()
        .filter_map(|(name, _, _, _)| hf_layer_name(name).map(|alias| (name.clone(), alias)))
        .collect();
    let aliases = if !hf.is_empty() {
        hf
    } else if file_to_canon.is_empty() && net.selection.is_none() {
        mlp_aliases(&raw)
    } else {
        HashMap::new()
    };
    for (fname, shape, dtype, data) in raw {
        let (canon, in_out) = match file_to_canon.get(&fname) {
            Some((c, io)) => (c.clone(), *io),
            None => (
                aliases
                    .get(&fname)
                    .cloned()
                    .unwrap_or_else(|| fname.clone()),
                opts.in_out && shape.len() == 2,
            ),
        };
        let (shape, data) = if in_out && shape.len() == 2 {
            transpose_in(&shape, data)
        } else {
            (shape, data)
        };
        let p = PartData {
            shape,
            data,
            file_name: fname,
            in_out,
            dtype,
            base: None,
            base_sha: None,
            max_tag: Tag::Lit,
        };
        if net.parts.insert(canon.clone(), p).is_some() {
            return Err(format!(
                "multiple file tensors map to canonical part `{canon}`"
            ));
        }
    }
    // architecture
    if net.selection.is_some() {
        net.arch = Arch::Unknown;
    } else if let Some(a) = side.as_ref().and_then(|s| s.get("arch")) {
        net.arch = arch_from_json(a)?;
    } else {
        net.arch = infer_mlp(&net, &opts.act).unwrap_or(Arch::Unknown);
    }
    if let Some(binding) = side
        .as_ref()
        .and_then(|s| s.get("head_binding"))
        .filter(|v| !v.is_null())
    {
        net.head_binding = Some(
            serde_json::from_value(binding.clone())
                .map_err(|e| format!("invalid head binding: {e}"))?,
        );
    }
    if let Some(binding) = &net.head_binding {
        net.meta.insert(
            "token_ids".into(),
            crate::value::Value::Arr(crate::tensor::Tensor::row(
                binding.tokens.iter().map(|&t| t as f64).collect(),
            )),
        );
    }
    net.snapshot_base();
    Ok(net)
}

/// Known dense language-model layer names only. Other tensor names stay untouched;
/// mapping names does not assert support for the model's forward architecture.
fn hf_layer_name(name: &str) -> Option<String> {
    let rest = name
        .strip_prefix("model.language_model.layers.")
        .or_else(|| name.strip_prefix("model.layers."))?;
    let (layer, part) = rest.split_once('.')?;
    let layer: usize = layer.parse().ok()?;
    let leaf = match part {
        "mlp.down_proj.weight" => "mlp.down",
        "mlp.up_proj.weight" => "mlp.up",
        "mlp.gate_proj.weight" => "mlp.gate",
        "self_attn.q_proj.weight" => "attn.q",
        "self_attn.k_proj.weight" => "attn.k",
        "self_attn.v_proj.weight" => "attn.v",
        "self_attn.o_proj.weight" => "attn.o",
        _ => return None,
    };
    Some(format!("L[{layer}].{leaf}"))
}

/// Recognise common MLP naming (Flax `a/dense1/kernel` + `bias`, PyTorch `fc1.weight` + `bias`)
/// and map it to canonical `W1, b1, W2, ...` in file order. File names stay as aliases.
fn mlp_aliases(raw: &[(String, Vec<usize>, Dtype, Vec<f64>)]) -> HashMap<String, String> {
    let mut out = HashMap::new();
    if raw.iter().any(|r| r.0 == "W1") {
        return out;
    }
    let mut layers: Vec<String> = Vec::new();
    for (name, shape, _, _) in raw {
        let (stem, leaf) = match name.rsplit_once(['/', '.']) {
            Some(x) => x,
            None => return HashMap::new(),
        };
        let is_w = matches!(leaf, "kernel" | "weight" | "w");
        let is_b = matches!(leaf, "bias" | "b");
        if !(is_w || is_b) || (is_w && shape.len() != 2) || (is_b && shape.len() != 1) {
            return HashMap::new();
        }
        let li = match layers.iter().position(|l| l == stem) {
            Some(i) => i,
            None => {
                layers.push(stem.to_string());
                layers.len() - 1
            }
        };
        out.insert(
            name.clone(),
            format!("{}{}", if is_w { "W" } else { "b" }, li + 1),
        );
    }
    out
}

fn infer_mlp(net: &NetData, act: &str) -> Option<Arch> {
    let mut sizes = Vec::new();
    let mut k = 1;
    while let Some(w) = net.parts.get(&format!("W{k}")) {
        if w.shape.len() != 2 {
            return None;
        }
        if k == 1 {
            sizes.push(w.shape[1]);
        }
        sizes.push(w.shape[0]);
        k += 1;
    }
    if sizes.len() < 2 {
        return None;
    }
    let mut acts = vec![act.to_string(); sizes.len() - 2];
    acts.push("linear".into());
    Some(Arch::Mlp { sizes, acts })
}

pub fn arch_to_json(a: &Arch) -> J {
    match a {
        Arch::Mlp { sizes, acts } => json!({"kind": "mlp", "sizes": sizes, "acts": acts}),
        Arch::Transformer(c) => json!({
            "kind": "transformer", "vocab": c.vocab, "ctx": c.ctx, "layers": c.layers,
            "heads": c.heads, "d": c.d, "ff": c.ff, "head_dim": c.head_dim, "act": c.act
        }),
        Arch::OpenThai2(g) => json!({"kind": "openthai2_rec", "graph": g}),
        Arch::Unknown => json!({"kind": "unknown"}),
    }
}

fn arch_from_json(a: &J) -> Result<Arch, String> {
    let us = |k: &str| {
        a.get(k)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .ok_or(format!("sidecar arch is missing `{k}`"))
    };
    match a.get("kind").and_then(|k| k.as_str()) {
        Some("openthai2_rec") => {
            let g: crate::ocr::Graph = serde_json::from_value(a["graph"].clone())
                .map_err(|e| format!("invalid OpenThai2 graph: {e}"))?;
            g.validate()?;
            Ok(Arch::OpenThai2(g))
        }
        Some("mlp") => {
            let sizes = a["sizes"]
                .as_array()
                .ok_or("sizes")?
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect();
            let acts = a["acts"]
                .as_array()
                .ok_or("acts")?
                .iter()
                .map(|v| v.as_str().unwrap_or("linear").to_string())
                .collect();
            Ok(Arch::Mlp { sizes, acts })
        }
        Some("transformer") => Ok(Arch::Transformer(TCfg {
            vocab: a["vocab"]
                .as_array()
                .ok_or("vocab")?
                .iter()
                .map(|v| v.as_str().unwrap_or("").to_string())
                .collect(),
            ctx: us("ctx")?,
            layers: us("layers")?,
            heads: us("heads")?,
            d: us("d")?,
            ff: us("ff")?,
            head_dim: us("head_dim")?,
            act: a
                .get("act")
                .and_then(|v| v.as_str())
                .unwrap_or("relu")
                .to_string(),
        })),
        _ => Ok(Arch::Unknown),
    }
}

pub fn sidecar_path(path: &str) -> String {
    format!("{path}.wisp.json")
}

pub(crate) fn safetensors_header(
    file: &mut fs::File,
    path: &str,
) -> Result<(safetensors::tensor::Metadata, u64), String> {
    let file_len = file.metadata().map_err(|e| e.to_string())?.len();
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix)
        .map_err(|e| format!("{path}: {e}"))?;
    let header_len = u64::from_le_bytes(prefix);
    if header_len > 100_000_000 || header_len > file_len.saturating_sub(8) {
        return Err(format!("{path}: invalid safetensors header length"));
    }
    let mut header = vec![0; header_len as usize];
    file.read_exact(&mut header).map_err(|e| e.to_string())?;
    let metadata: safetensors::tensor::Metadata =
        serde_json::from_slice(&header).map_err(|e| format!("{path}: {e}"))?;
    let mut entries: Vec<_> = metadata.tensors().into_iter().collect();
    entries.sort_by(|(a, x), (b, y)| x.data_offsets.cmp(&y.data_offsets).then(a.cmp(b)));
    // Validate every tensor's layout before reading even a selected subset.
    let mut end = 0usize;
    for (name, t) in &entries {
        let (start, stop) = t.data_offsets;
        let count = t
            .shape
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .ok_or("tensor shape overflow")?;
        let bytes = count
            .checked_mul(t.dtype.size())
            .ok_or("tensor byte size overflow")?;
        if start != end || stop < start || stop - start != bytes {
            return Err(format!("{path}: invalid offsets or shape for {name}"));
        }
        end = stop;
    }
    let data_start = 8 + header_len;
    if end as u64 != file_len - data_start {
        return Err(format!(
            "{path}: safetensors data length does not match its header"
        ));
    }
    Ok((metadata, data_start))
}

fn load_safetensors(path: &str, selected: Option<&[String]>) -> Result<Vec<LoadedTensor>, String> {
    let mut file = fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let (metadata, data_start) = safetensors_header(&mut file, path)?;
    let mut entries: Vec<_> = metadata.tensors().into_iter().collect();
    entries.sort_by(|(a, x), (b, y)| x.data_offsets.cmp(&y.data_offsets).then(a.cmp(b)));
    if let Some(names) = selected {
        for name in names {
            if metadata.info(name).is_none() {
                return Err(format!("{path}: no tensor named '{name}'"));
            }
        }
    }
    let mut out = Vec::new();
    for (name, t) in entries {
        if selected.is_some_and(|names| !names.contains(&name)) {
            continue;
        }
        let dtype = match t.dtype {
            safetensors::Dtype::F32 => Dtype::F32,
            safetensors::Dtype::F64 => Dtype::F64,
            safetensors::Dtype::BF16 => Dtype::BF16,
            safetensors::Dtype::F16 => Dtype::F16,
            other => {
                return Err(format!(
                    "{path}: tensor {name} has unsupported dtype {other:?}"
                ))
            }
        };
        file.seek(SeekFrom::Start(data_start + t.data_offsets.0 as u64))
            .map_err(|e| e.to_string())?;
        let mut bytes = vec![0; t.data_offsets.1 - t.data_offsets.0];
        file.read_exact(&mut bytes).map_err(|e| e.to_string())?;
        out.push((name, t.shape.clone(), dtype, decode(dtype, &bytes)));
    }
    Ok(out)
}

fn load_npz(path: &str) -> Result<Vec<LoadedTensor>, String> {
    let f = fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut z = zip::ZipArchive::new(f).map_err(|e| format!("{path}: {e}"))?;
    let mut out = Vec::new();
    for i in 0..z.len() {
        let mut e = z.by_index(i).map_err(|e| e.to_string())?;
        let name = e.name().trim_end_matches(".npy").to_string();
        let mut buf = Vec::new();
        e.read_to_end(&mut buf).map_err(|e| e.to_string())?;
        let (shape, dtype, data) = parse_npy(&buf).map_err(|m| format!("{path}:{name}: {m}"))?;
        out.push((name, shape, dtype, data));
    }
    Ok(out)
}

fn parse_npy(b: &[u8]) -> Result<(Vec<usize>, Dtype, Vec<f64>), String> {
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err("not a .npy array".into());
    }
    let major = b[6];
    if b[7] != 0 {
        return Err("unsupported .npy minor version".into());
    }
    let (hlen, hstart): (usize, usize) = match major {
        1 => (u16::from_le_bytes([b[8], b[9]]) as usize, 10),
        2 | 3 => {
            let bytes = b.get(8..12).ok_or("truncated .npy prefix")?;
            (
                u32::from_le_bytes(bytes.try_into().map_err(|_| "bad prefix")?) as usize,
                12,
            )
        }
        _ => return Err("unsupported .npy version".into()),
    };
    let data_start = hstart.checked_add(hlen).ok_or("header length overflow")?;
    let header = std::str::from_utf8(b.get(hstart..data_start).ok_or("truncated .npy header")?)
        .map_err(|_| "bad header")?;
    let descr = header
        .split("'descr':")
        .nth(1)
        .and_then(|s| s.split('\'').nth(1))
        .ok_or("no descr")?;
    let fortran = header.contains("'fortran_order': True");
    let shape_s = header
        .split("'shape':")
        .nth(1)
        .and_then(|s| s.split('(').nth(1))
        .and_then(|s| s.split(')').next())
        .ok_or("no shape")?;
    let shape: Vec<usize> = shape_s
        .split(',')
        .map(|x| x.trim())
        .filter(|x| !x.is_empty())
        .map(|x| x.parse::<usize>().map_err(|_| "invalid shape dimension"))
        .collect::<Result<_, _>>()?;
    let dtype = match descr {
        "<f4" => Dtype::F32,
        "<f8" => Dtype::F64,
        "<f2" => Dtype::F16,
        other => return Err(format!("unsupported dtype {other}")),
    };
    let n = shape
        .iter()
        .try_fold(1usize, |n, d| n.checked_mul(*d))
        .ok_or("shape overflow")?;
    let width = match dtype {
        Dtype::F16 | Dtype::BF16 => 2,
        Dtype::F32 => 4,
        Dtype::F64 => 8,
    };
    let expected = n.checked_mul(width).ok_or("data size overflow")?;
    if b.len() - data_start != expected {
        return Err("data length does not match shape".into());
    }
    if fortran && shape.len() > 2 {
        return Err("Fortran-order arrays above rank 2 are unsupported".into());
    }
    let mut data = decode(dtype, &b[data_start..]);
    let shape = if shape.is_empty() { vec![1] } else { shape };
    if fortran && shape.len() == 2 {
        let (r, c) = (shape[0], shape[1]);
        let mut d = vec![0.0; r * c];
        for j in 0..c {
            for i in 0..r {
                d[i * c + j] = data[j * r + i];
            }
        }
        data = d;
    }
    Ok((shape, dtype, data))
}

pub struct SaveOpts {
    pub obstacle: Option<String>,
    pub preserve: Option<String>,
    pub budget: Option<f64>,
    pub protect: Vec<String>,
}

pub struct SaveReport {
    pub changed_parts: usize,
    pub changed_entries: usize,
    pub share: f64,
    pub patch_path: String,
}

/// Write the model, its sidecar and its patch. Applies the protection and global-knob lints.
pub fn save(net: &NetData, path: &str, opts: &SaveOpts) -> Result<SaveReport, (String, String)> {
    if net.decoder.is_some() {
        return Err(("E-UNSUPPORTED".into(), "save token_head output rows, then reload with with_head; full decoder export is not supported".into()));
    }
    // ---- patch first, so lints refuse before anything is written
    let mut parts_j = serde_json::Map::new();
    let mut changed_total = 0usize;
    let total = net.total_weights().max(1);
    for (canon, p) in &net.parts {
        let base = match &p.base {
            Some(b) => b,
            None => continue,
        };
        let (_, fb) = file_layout(p, base);
        let (fshape, fe) = file_layout(p, &p.data);
        let mut idx = Vec::new();
        let mut val = Vec::new();
        for (k, (&a, &b)) in fb.iter().zip(&fe).enumerate() {
            let rb = p.dtype.round(b);
            if p.dtype.round(a).to_bits() != rb.to_bits() {
                idx.push(k);
                val.push(rb);
            }
        }
        if idx.is_empty() {
            continue;
        }
        if opts.protect.iter().any(|x| x == canon || x == &p.file_name) {
            return Err((
                "E-PROTECTED".into(),
                format!(
                    "part `{canon}` is protected by the task, but {} entries changed",
                    idx.len()
                ),
            ));
        }
        if let Some(c) = global_scale(base, &p.data) {
            return Err((
                "E-GLOBAL-KNOB".into(),
                format!(
                    "part `{canon}` was only multiplied by {} as a whole; global knobs are refused",
                    crate::tensor::fmt_num(c)
                ),
            ));
        }
        let bn: f64 = base.iter().map(|x| x * x).sum::<f64>().sqrt();
        let dn: f64 = base
            .iter()
            .zip(&p.data)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        changed_total += idx.len();
        parts_j.insert(
            canon.clone(),
            json!({
                "file_name": p.file_name,
                "file_shape": fshape,
                "dtype": p.dtype.name(),
                "base_sha256": p.base_sha.clone().unwrap_or_default(),
                "edited_sha256": part_sha(p, &p.data),
                "changed_entries": idx.len(),
                "relative_change": if bn > 0.0 { dn / bn } else { f64::INFINITY },
                "provenance": p.max_tag.name(),
                "form": "sparse",
                "idx": idx,
                "val": val,
            }),
        );
    }
    let share = changed_total as f64 / total as f64;
    if let Some(b) = opts.budget {
        if share > b {
            return Err((
                "E-EDIT-BUDGET".into(),
                format!(
                    "the edit changes {:.4}% of weights, more than the stated budget of {:.4}%",
                    share * 100.0,
                    b * 100.0
                ),
            ));
        }
    }
    // ---- model file
    let mut tensors: Vec<(String, Vec<usize>, Vec<u8>, safetensors::Dtype)> = Vec::new();
    let mut parts_meta = serde_json::Map::new();
    for (canon, p) in &net.parts {
        let (fshape, fd) = file_layout(p, &p.data);
        let st_dtype = match p.dtype {
            Dtype::F32 => safetensors::Dtype::F32,
            Dtype::F64 => safetensors::Dtype::F64,
            Dtype::BF16 => safetensors::Dtype::BF16,
            Dtype::F16 => safetensors::Dtype::F16,
        };
        tensors.push((p.file_name.clone(), fshape, encode(p.dtype, &fd), st_dtype));
        parts_meta.insert(
            canon.clone(),
            json!({"file": p.file_name, "in_out": p.in_out}),
        );
    }
    let views: Vec<(String, safetensors::tensor::TensorView)> = tensors
        .iter()
        .map(|(n, s, b, d)| {
            (
                n.clone(),
                safetensors::tensor::TensorView::new(*d, s.clone(), b).unwrap(),
            )
        })
        .collect();
    let out_path = if path.ends_with(".safetensors") {
        path.to_string()
    } else {
        format!("{path}.safetensors")
    };
    safetensors::serialize_to_file(views, &None, Path::new(&out_path))
        .map_err(|e| ("E-IO".to_string(), e.to_string()))?;
    let side = json!({"arch": arch_to_json(&net.arch), "parts": parts_meta, "selected_files": net.selection, "head_binding": net.head_binding});
    fs::write(
        sidecar_path(&out_path),
        serde_json::to_string_pretty(&side).unwrap(),
    )
    .map_err(|e| ("E-IO".to_string(), e.to_string()))?;
    let patch_path = out_path.trim_end_matches(".safetensors").to_string() + ".patch.json";
    let layout = parts_meta.clone();
    let patch = json!({
        "wisp_patch": 1,
        "scope": if net.head_binding.is_some() { "output_rows" } else if net.selection.is_some() { "selected_parts" } else { "full_model" },
        "head_binding": net.head_binding,
        "selected_files": net.selection,
        "base": net.source.clone().unwrap_or_else(|| format!("constructor:{}", net.name)),
        "obstacle": opts.obstacle,
        "preserve": opts.preserve,
        "edit_budget": opts.budget,
        "changed_parts": parts_j.len(),
        "changed_entries": changed_total,
        "share_of_weights": share,
        "layout": layout,
        "parts": parts_j,
    });
    fs::write(&patch_path, serde_json::to_string_pretty(&patch).unwrap())
        .map_err(|e| ("E-IO".to_string(), e.to_string()))?;
    Ok(SaveReport {
        changed_parts: patch["changed_parts"].as_u64().unwrap() as usize,
        changed_entries: changed_total,
        share,
        patch_path,
    })
}

/// Some(c) when `new` is exactly `c * base` for one constant c != 1 over a non-zero base.
fn global_scale(base: &[f64], new: &[f64]) -> Option<f64> {
    let mut c: Option<f64> = None;
    let mut any = false;
    for (&a, &b) in base.iter().zip(new) {
        if a == 0.0 {
            if b != 0.0 {
                return None;
            }
            continue;
        }
        any = true;
        let r = b / a;
        match c {
            None => c = Some(r),
            Some(c0) => {
                if (r - c0).abs() > 1e-9 * c0.abs().max(1.0) {
                    return None;
                }
            }
        }
    }
    match c {
        Some(c) if any && (c - 1.0).abs() > 1e-12 => Some(c),
        _ => None,
    }
}

/// Rebuild an edited model from a base file and a patch, checking both hashes.
pub fn apply_patch(base_path: &str, patch_path: &str, out_path: &str) -> Result<String, String> {
    let patch: J =
        serde_json::from_str(&fs::read_to_string(patch_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let opts = LoadOpts {
        in_out: false,
        act: "tanh".into(),
    };
    let mut net = if patch["scope"].as_str() == Some("selected_parts") {
        let names: Vec<String> = patch["selected_files"]
            .as_array()
            .ok_or("selected_parts patch has no selected_files")?
            .iter()
            .map(|v| {
                v.as_str()
                    .map(String::from)
                    .ok_or("invalid selected file name")
            })
            .collect::<Result<_, _>>()?;
        load_selected(base_path, &opts, &names)?
    } else {
        load(base_path, &opts)?
    };
    if patch["scope"].as_str() == Some("output_rows") {
        let binding: crate::decoder::HeadBinding =
            serde_json::from_value(patch["head_binding"].clone())
                .map_err(|e| format!("invalid patch head binding: {e}"))?;
        if net.head_binding.as_ref() != Some(&binding) {
            return Err("output-row patch binding does not match its base bundle".into());
        }
    }
    if let Some(layout) = patch.get("layout").and_then(|l| l.as_object()) {
        let mut parts = indexmap::IndexMap::new();
        for (_, mut p) in std::mem::take(&mut net.parts) {
            let entry = layout
                .iter()
                .find(|(_, v)| v["file"].as_str() == Some(p.file_name.as_str()));
            let canon = entry
                .map(|(k, _)| k.clone())
                .unwrap_or_else(|| p.file_name.clone());
            let io = entry
                .and_then(|(_, v)| v["in_out"].as_bool())
                .unwrap_or(false);
            if io && p.shape.len() == 2 && !p.in_out {
                let (shape, data) = transpose_in(&p.shape, std::mem::take(&mut p.data));
                p.shape = shape;
                p.data = data;
                p.in_out = true;
            }
            p.base_sha = Some(part_sha(&p, &p.data));
            parts.insert(canon, p);
        }
        net.parts = parts;
    }
    let mut log = String::new();
    let parts = patch["parts"].as_object().ok_or("patch has no parts")?;
    for (canon, info) in parts {
        let p = net
            .parts
            .get_mut(canon)
            .ok_or(format!("base has no part `{canon}`"))?;
        let want_base = info["base_sha256"].as_str().unwrap_or("");
        let have_base = part_sha(p, &p.data);
        if want_base != have_base {
            return Err(format!(
                "{canon}: the base tensor differs from the one the patch was made on"
            ));
        }
        let (fshape, mut fd) = file_layout(p, &p.data);
        let idx = info["idx"].as_array().ok_or("idx")?;
        let val = info["val"].as_array().ok_or("val")?;
        for (i, v) in idx.iter().zip(val) {
            fd[i.as_u64().unwrap() as usize] = v.as_f64().unwrap();
        }
        // back to canonical layout
        let data = if p.in_out && fshape.len() == 2 {
            transpose_in(&fshape, fd).1
        } else {
            fd
        };
        p.data = data;
        let got = part_sha(p, &p.data);
        if got != info["edited_sha256"].as_str().unwrap_or("") {
            return Err(format!(
                "{canon}: the rebuilt tensor does not match the patch's edited hash"
            ));
        }
        log += &format!("{canon}: rebuilt, {} entries changed, hash OK\n", idx.len());
        p.max_tag = Tag::Lit;
    }
    let mut copy = net.clone();
    for p in copy.parts.values_mut() {
        p.base = None;
    }
    save(
        &copy,
        out_path,
        &SaveOpts {
            obstacle: None,
            preserve: None,
            budget: None,
            protect: vec![],
        },
    )
    .map_err(|(c, m)| format!("{c}: {m}"))?;
    Ok(log)
}

#[cfg(test)]
mod npy_tests {
    use super::parse_npy;

    fn fixture(version: u8, shape: &str, fortran: bool, values: &[f32]) -> Vec<u8> {
        let order = if fortran { "True" } else { "False" };
        let header =
            format!("{{'descr': '<f4', 'fortran_order': {order}, 'shape': ({shape}), }}\n");
        let mut bytes = b"\x93NUMPY".to_vec();
        bytes.extend([version, 0]);
        if version == 1 {
            bytes.extend((header.len() as u16).to_le_bytes());
        } else {
            bytes.extend((header.len() as u32).to_le_bytes());
        }
        bytes.extend(header.bytes());
        for value in values {
            bytes.extend(value.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn all_truncated_prefixes_headers_and_payloads_are_errors_not_panics() {
        for version in [1, 2, 3] {
            let bytes = fixture(version, "2, 3", false, &[1., 2., 3., 4., 5., 6.]);
            for end in 0..bytes.len() {
                assert!(
                    parse_npy(&bytes[..end]).is_err(),
                    "version {version}, length {end}"
                );
            }
            let (shape, _, data) = parse_npy(&bytes).unwrap();
            assert_eq!(shape, [2, 3]);
            assert_eq!(data, [1., 2., 3., 4., 5., 6.]);
        }
    }

    #[test]
    fn invalid_or_overflowing_dimensions_are_rejected() {
        for shape in [
            "-1, 3".to_string(),
            "oops, 2".to_string(),
            format!("{}, 2", usize::MAX),
            format!("{},", usize::MAX),
        ] {
            assert!(parse_npy(&fixture(1, &shape, false, &[])).is_err());
        }
    }

    #[test]
    fn exact_payload_size_is_required() {
        assert!(parse_npy(&fixture(1, "2, 3", true, &[1.])).is_err());
        assert!(parse_npy(&fixture(1, "1,", false, &[1., 2.])).is_err());
        let mut bytes = fixture(1, "1,", false, &[1.]);
        bytes.push(0);
        assert!(parse_npy(&bytes).is_err());
    }

    #[test]
    fn empty_arrays_are_distinct_from_scalars() {
        let (shape, _, data) = parse_npy(&fixture(1, "0, 3", false, &[])).unwrap();
        assert_eq!(shape, [0, 3]);
        assert!(data.is_empty());
        let (shape, _, data) = parse_npy(&fixture(1, "", false, &[7.])).unwrap();
        assert_eq!(shape, [1]);
        assert_eq!(data, [7.]);
    }

    #[test]
    fn fortran_order_is_converted_or_explicitly_refused() {
        let (_, _, data) = parse_npy(&fixture(1, "2, 3", true, &[1., 4., 2., 5., 3., 6.])).unwrap();
        assert_eq!(data, [1., 2., 3., 4., 5., 6.]);
        assert!(parse_npy(&fixture(1, "1, 1, 1", true, &[1.])).is_err());
    }

    #[test]
    fn unsupported_versions_and_oversized_headers_are_rejected() {
        let mut bytes = fixture(2, "1,", false, &[1.]);
        bytes[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(parse_npy(&bytes).is_err());
        assert!(parse_npy(&fixture(4, "1,", false, &[1.])).is_err());
        let mut bytes = fixture(1, "1,", false, &[1.]);
        bytes[7] = 1;
        assert!(parse_npy(&bytes).is_err());
    }
}
