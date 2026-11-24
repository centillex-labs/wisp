//! GGUF containers: header parsing, tensor decoding, and a streaming rewrite that copies
//! every untouched tensor byte for byte.
//!
//! Wisp decodes only the plain and Q8_0 tensor types. Any other type (K-quants, NVFP4, ...)
//! is opaque: it can be listed and copied, never read as numbers.

use crate::builtins::{arr, cell_strings, opts, string};
use crate::interp::{err, Interp, R};
use crate::net::{bf16_to_f32, f16_to_f32, f32_to_bf16};
use crate::tensor::{Tag, Tensor};
use crate::value::Value;
use indexmap::IndexMap;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::ops::Range;
use wisp_syntax::Span;

const MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;
const MAX_HEADER: usize = 1 << 30;
const MAX_DIMS: u32 = 8;

pub const T_F32: u32 = 0;
pub const T_F16: u32 = 1;
pub const T_Q8_0: u32 = 8;
pub const T_I8: u32 = 24;
pub const T_I16: u32 = 25;
pub const T_I32: u32 = 26;
pub const T_I64: u32 = 27;
pub const T_F64: u32 = 28;
pub const T_BF16: u32 = 30;

/// Names of ggml tensor types. Types without a name here print as `T<number>`.
const TYPE_NAMES: &[(u32, &str)] = &[
    (0, "F32"),
    (1, "F16"),
    (2, "Q4_0"),
    (3, "Q4_1"),
    (6, "Q5_0"),
    (7, "Q5_1"),
    (8, "Q8_0"),
    (9, "Q8_1"),
    (10, "Q2_K"),
    (11, "Q3_K"),
    (12, "Q4_K"),
    (13, "Q5_K"),
    (14, "Q6_K"),
    (15, "Q8_K"),
    (16, "IQ2_XXS"),
    (17, "IQ2_XS"),
    (18, "IQ3_XXS"),
    (19, "IQ1_S"),
    (20, "IQ4_NL"),
    (21, "IQ3_S"),
    (22, "IQ2_S"),
    (23, "IQ4_XS"),
    (24, "I8"),
    (25, "I16"),
    (26, "I32"),
    (27, "I64"),
    (28, "F64"),
    (29, "IQ1_M"),
    (30, "BF16"),
    (34, "TQ1_0"),
    (35, "TQ2_0"),
];

pub fn type_name(ty: u32) -> String {
    TYPE_NAMES
        .iter()
        .find(|(t, _)| *t == ty)
        .map(|(_, n)| n.to_string())
        .unwrap_or_else(|| format!("T{ty}"))
}

pub fn type_from_name(name: &str) -> Option<u32> {
    if let Some((t, _)) = TYPE_NAMES.iter().find(|(_, n)| *n == name) {
        return Some(*t);
    }
    name.strip_prefix('T').and_then(|n| n.parse().ok())
}

/// Bytes needed to store `count` values of a type Wisp can encode or decode.
fn exact_bytes(ty: u32, count: u64) -> Option<u64> {
    let per = match ty {
        T_I8 => 1,
        T_F16 | T_BF16 | T_I16 => 2,
        T_F32 | T_I32 => 4,
        T_F64 | T_I64 => 8,
        T_Q8_0 => {
            return if count % 32 == 0 {
                (count / 32).checked_mul(34)
            } else {
                None
            }
        }
        _ => return None,
    };
    count.checked_mul(per)
}

#[derive(Clone, Debug)]
pub enum Meta {
    Num(f64),
    Bool(bool),
    Str(String),
    /// Element type, element count, and the byte range of the elements in the header.
    Array(u32, u64, Range<usize>),
}

#[derive(Clone, Debug)]
pub struct Kv {
    pub key: String,
    pub ty: u32,
    pub value: Meta,
    /// The whole entry (key, type and value) in the header.
    pub raw: Range<usize>,
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    /// Dimensions as stored in the file (innermost first).
    pub dims: Vec<u64>,
    pub ty: u32,
    /// Offset from the start of the tensor data section.
    pub offset: u64,
    /// Bytes owned by this tensor: exact for types Wisp understands, otherwise everything up to
    /// the next tensor (which includes alignment padding).
    pub nbytes: u64,
}

impl TensorInfo {
    pub fn count(&self) -> u64 {
        self.dims.iter().product()
    }
    /// Shape in row-major order (outermost first), as a Wisp array has it.
    pub fn shape(&self) -> Vec<usize> {
        let mut s: Vec<usize> = self.dims.iter().rev().map(|&d| d as usize).collect();
        if s.len() == 1 {
            s.push(1);
        }
        if s.is_empty() {
            s = vec![1, 1];
        }
        s
    }
}

pub struct Gguf {
    pub path: String,
    pub version: u32,
    pub alignment: u64,
    pub kvs: Vec<Kv>,
    pub tensors: Vec<TensorInfo>,
    pub data_start: u64,
    pub file_len: u64,
    header: Vec<u8>,
}

/// Reads the header on demand, keeping every byte so entries can be copied verbatim.
struct Cursor {
    file: fs::File,
    buf: Vec<u8>,
    pos: usize,
    file_len: u64,
}

impl Cursor {
    fn need(&mut self, n: usize) -> Result<(), String> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or("GGUF header length overflow")?;
        if end as u64 > self.file_len {
            return Err("GGUF header runs past the end of the file".into());
        }
        if end > MAX_HEADER {
            return Err("GGUF header is larger than 1 GiB".into());
        }
        while self.buf.len() < end {
            let want = (end - self.buf.len()).max(1 << 20);
            let left = (self.file_len - self.buf.len() as u64).min(want as u64) as usize;
            let at = self.buf.len();
            self.buf.resize(at + left, 0);
            self.file
                .read_exact(&mut self.buf[at..])
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
    fn take(&mut self, n: usize) -> Result<&[u8], String> {
        self.need(n)?;
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn len(&mut self) -> Result<usize, String> {
        let n = self.u64()?;
        if n > self.file_len {
            return Err("GGUF length field is larger than the file".into());
        }
        Ok(n as usize)
    }
    fn string(&mut self) -> Result<String, String> {
        let n = self.len()?;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| "GGUF string is not UTF-8".into())
    }
}

fn scalar_width(ty: u32) -> Option<usize> {
    Some(match ty {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        10..=12 => 8,
        _ => return None,
    })
}

fn scalar_value(ty: u32, b: &[u8]) -> Meta {
    match ty {
        0 => Meta::Num(b[0] as f64),
        1 => Meta::Num(b[0] as i8 as f64),
        2 => Meta::Num(u16::from_le_bytes([b[0], b[1]]) as f64),
        3 => Meta::Num(i16::from_le_bytes([b[0], b[1]]) as f64),
        4 => Meta::Num(u32::from_le_bytes(b.try_into().unwrap()) as f64),
        5 => Meta::Num(i32::from_le_bytes(b.try_into().unwrap()) as f64),
        6 => Meta::Num(f32::from_le_bytes(b.try_into().unwrap()) as f64),
        7 => Meta::Bool(b[0] != 0),
        10 => Meta::Num(u64::from_le_bytes(b.try_into().unwrap()) as f64),
        11 => Meta::Num(i64::from_le_bytes(b.try_into().unwrap()) as f64),
        _ => Meta::Num(f64::from_le_bytes(b.try_into().unwrap())),
    }
}

fn skip_value(c: &mut Cursor, ty: u32, depth: u32) -> Result<Meta, String> {
    if let Some(w) = scalar_width(ty) {
        return Ok(scalar_value(ty, c.take(w)?));
    }
    match ty {
        8 => Ok(Meta::Str(c.string()?)),
        9 => {
            if depth > 4 {
                return Err("GGUF arrays are nested too deeply".into());
            }
            let elem = c.u32()?;
            let count = c.u64()?;
            let start = c.pos;
            if let Some(w) = scalar_width(elem) {
                let bytes = (count as usize)
                    .checked_mul(w)
                    .ok_or("GGUF array length overflow")?;
                c.take(bytes)?;
            } else {
                for _ in 0..count {
                    skip_value(c, elem, depth + 1)?;
                }
            }
            Ok(Meta::Array(elem, count, start..c.pos))
        }
        other => Err(format!("unknown GGUF metadata type {other}")),
    }
}

impl Gguf {
    pub fn open(path: &str) -> Result<Gguf, String> {
        Self::parse(path).map_err(|m| format!("{path}: {m}"))
    }

    fn parse(path: &str) -> Result<Gguf, String> {
        let file = fs::File::open(path).map_err(|e| e.to_string())?;
        let file_len = file.metadata().map_err(|e| e.to_string())?.len();
        let mut c = Cursor {
            file,
            buf: Vec::new(),
            pos: 0,
            file_len,
        };
        if c.take(4)? != MAGIC {
            return Err("not a GGUF file".into());
        }
        let version = c.u32()?;
        if version != 2 && version != 3 {
            return Err(format!(
                "unsupported GGUF version {version} (big-endian files are not supported)"
            ));
        }
        let tensor_count = c.len()?;
        let kv_count = c.len()?;
        let mut kvs = Vec::new();
        let mut alignment = DEFAULT_ALIGNMENT;
        let mut keys = HashSet::new();
        for _ in 0..kv_count {
            let start = c.pos;
            let key = c.string()?;
            let ty = c.u32()?;
            let value = skip_value(&mut c, ty, 0)?;
            if key == "general.alignment" {
                match value {
                    Meta::Num(a) if a >= 1.0 && (a as u64).is_power_of_two() => {
                        alignment = a as u64
                    }
                    _ => return Err("invalid general.alignment".into()),
                }
            }
            if !keys.insert(key.clone()) {
                return Err(format!("duplicate metadata key '{key}'"));
            }
            kvs.push(Kv {
                key,
                ty,
                value,
                raw: start..c.pos,
            });
        }
        let mut tensors = Vec::new();
        let mut names = HashSet::new();
        for _ in 0..tensor_count {
            let name = c.string()?;
            let nd = c.u32()?;
            if nd > MAX_DIMS {
                return Err(format!("tensor '{name}' has {nd} dimensions"));
            }
            let mut dims = Vec::new();
            let mut count = 1u64;
            for _ in 0..nd {
                let d = c.u64()?;
                count = count
                    .checked_mul(d)
                    .ok_or_else(|| format!("tensor '{name}' shape overflows"))?;
                dims.push(d);
            }
            let ty = c.u32()?;
            let offset = c.u64()?;
            if !names.insert(name.clone()) {
                return Err(format!("duplicate tensor name '{name}'"));
            }
            tensors.push(TensorInfo {
                name,
                dims,
                ty,
                offset,
                nbytes: 0,
            });
        }
        let data_start = (c.pos as u64).div_ceil(alignment) * alignment;
        if data_start > file_len {
            return Err("tensor data section starts past the end of the file".into());
        }
        let data_len = file_len - data_start;
        // Size each tensor: exact where the type is understood, otherwise up to the next offset.
        let mut order: Vec<usize> = (0..tensors.len()).collect();
        order.sort_by_key(|&i| tensors[i].offset);
        for (k, &i) in order.iter().enumerate() {
            let next = order
                .get(k + 1)
                .map(|&j| tensors[j].offset)
                .unwrap_or(data_len);
            let t = &tensors[i];
            if t.offset % alignment != 0 || next < t.offset || next > data_len {
                return Err(format!("tensor '{}' has an invalid offset", t.name));
            }
            let span = next - t.offset;
            let nbytes = match exact_bytes(t.ty, t.count()) {
                Some(n) if n <= span => n,
                Some(_) => {
                    return Err(format!(
                        "tensor '{}' does not fit before the next tensor",
                        t.name
                    ))
                }
                None if matches!(
                    t.ty,
                    T_F32 | T_F16 | T_BF16 | T_F64 | T_Q8_0 | T_I8 | T_I16 | T_I32 | T_I64
                ) =>
                {
                    return Err(format!("tensor '{}' has an invalid shape", t.name))
                }
                None => span,
            };
            tensors[i].nbytes = nbytes;
        }
        let pos = c.pos;
        let mut header = c.buf;
        header.truncate(pos);
        Ok(Gguf {
            path: path.to_string(),
            version,
            alignment,
            kvs,
            tensors,
            data_start,
            file_len,
            header,
        })
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo, String> {
        self.tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| format!("{}: no tensor named '{name}'", self.path))
    }

    pub fn kv(&self, key: &str) -> Option<&Kv> {
        self.kvs.iter().find(|k| k.key == key)
    }

    fn raw(&self, t: &TensorInfo) -> Result<Vec<u8>, String> {
        let mut f = fs::File::open(&self.path).map_err(|e| e.to_string())?;
        f.seek(SeekFrom::Start(self.data_start + t.offset))
            .map_err(|e| e.to_string())?;
        let mut b = vec![0u8; t.nbytes as usize];
        f.read_exact(&mut b).map_err(|e| e.to_string())?;
        Ok(b)
    }

    /// Decode one tensor to float64 values in row-major order.
    pub fn read(&self, name: &str) -> Result<Tensor, String> {
        let t = self.tensor(name)?;
        let bytes = self.raw(t)?;
        let data = decode(t.ty, &bytes).ok_or_else(|| {
            format!(
                "{}: tensor '{name}' has type {}, which Wisp can copy but not decode",
                self.path,
                type_name(t.ty)
            )
        })?;
        Ok(Tensor::new(t.shape(), data, Tag::W))
    }

    /// Elements of an array-valued metadata entry.
    fn array(&self, elem: u32, count: u64, range: &Range<usize>) -> Result<Value, String> {
        let b = &self.header[range.clone()];
        if let Some(w) = scalar_width(elem) {
            let v: Vec<f64> = b
                .chunks_exact(w)
                .map(|c| match scalar_value(elem, c) {
                    Meta::Num(x) => x,
                    Meta::Bool(x) => x as u8 as f64,
                    _ => unreachable!(),
                })
                .collect();
            return Ok(Value::Arr(Tensor::row(v)));
        }
        if elem != 8 {
            return Err("nested metadata arrays cannot be read".into());
        }
        let mut out = Vec::with_capacity(count as usize);
        let mut at = 0usize;
        for _ in 0..count {
            let n = u64::from_le_bytes(b[at..at + 8].try_into().unwrap()) as usize;
            at += 8;
            out.push(Value::str(String::from_utf8_lossy(&b[at..at + n])));
            at += n;
        }
        Ok(Value::cell(out))
    }

    pub fn meta_value(&self, kv: &Kv) -> Result<Value, String> {
        Ok(match &kv.value {
            Meta::Num(x) => Value::num(*x),
            Meta::Bool(b) => Value::boolean(*b),
            Meta::Str(s) => Value::str(s),
            Meta::Array(elem, count, range) => self.array(*elem, *count, range)?,
        })
    }
}

// ---- tensor codecs

/// Round-to-nearest-even float32 to float16, bit-exact with IEEE 754 (and numpy).
pub fn f32_to_f16_exact(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x007f_ffff;
    if exp == 255 {
        return sign | 0x7c00 | if man != 0 { 0x200 | (man >> 13) as u16 } else { 0 };
    }
    let e = exp - 112;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rem = m & ((1u32 << shift) - 1);
        let mut r = m >> shift;
        if rem > half || (rem == half && r & 1 == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut h = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    sign | h as u16
}

fn decode(ty: u32, b: &[u8]) -> Option<Vec<f64>> {
    Some(match ty {
        T_F32 => b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64)
            .collect(),
        T_F64 => b
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        T_F16 => b
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])) as f64)
            .collect(),
        T_BF16 => b
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])) as f64)
            .collect(),
        T_I8 => b.iter().map(|&x| x as i8 as f64).collect(),
        T_I16 => b
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f64)
            .collect(),
        T_I32 => b
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f64)
            .collect(),
        T_I64 => b
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f64)
            .collect(),
        T_Q8_0 => {
            let mut out = Vec::with_capacity(b.len() / 34 * 32);
            for blk in b.chunks_exact(34) {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                out.extend(blk[2..].iter().map(|&q| (d * (q as i8 as f32)) as f64));
            }
            out
        }
        _ => return None,
    })
}

/// Encode values for storage. Q8_0 follows llama.cpp's reference quantizer in float32:
/// per 32-value block, scale = max|x| / 127 stored as float16, values rounded half away from zero.
pub fn encode(ty: u32, data: &[f64]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(exact_bytes(ty, data.len() as u64)? as usize);
    match ty {
        T_F32 => data
            .iter()
            .for_each(|&v| out.extend_from_slice(&(v as f32).to_le_bytes())),
        T_F64 => data
            .iter()
            .for_each(|&v| out.extend_from_slice(&v.to_le_bytes())),
        T_F16 => data
            .iter()
            .for_each(|&v| out.extend_from_slice(&f32_to_f16_exact(v as f32).to_le_bytes())),
        T_BF16 => data
            .iter()
            .for_each(|&v| out.extend_from_slice(&f32_to_bf16(v as f32).to_le_bytes())),
        T_Q8_0 => {
            for blk in data.chunks_exact(32) {
                let mut x = [0f32; 32];
                let mut top = 0f32;
                for (o, &v) in x.iter_mut().zip(blk) {
                    *o = v as f32;
                    top = top.max(o.abs());
                }
                let d = top / 127.0;
                let id = if d == 0.0 { 0.0 } else { 1.0 / d };
                out.extend_from_slice(&f32_to_f16_exact(d).to_le_bytes());
                for v in x {
                    let p = v * id;
                    let a = p.abs();
                    let fl = a.floor();
                    let r = fl + (2.0 * (a - fl)).floor();
                    let q = if p < 0.0 { -r } else { r };
                    out.push(q as i8 as u8);
                }
            }
        }
        _ => return None,
    }
    Some(out)
}

// ---- streaming rewrite

pub struct Plan {
    /// Tensor name -> new storage type, for tensors supplied by the caller.
    pub edits: IndexMap<String, u32>,
    pub drop: HashSet<String>,
    /// String metadata to replace (existing key) or append (new key), in order.
    pub meta: Vec<(String, String)>,
}

pub struct Report {
    pub edited: usize,
    pub dropped: usize,
    pub copied: usize,
    pub bytes: u64,
    pub sha256: String,
}

pub enum Fail<E> {
    Msg(String),
    Inner(E),
}

impl<E> From<String> for Fail<E> {
    fn from(m: String) -> Self {
        Fail::Msg(m)
    }
}

struct Sink {
    w: BufWriter<fs::File>,
    sha: Sha256,
    n: u64,
}

impl Sink {
    fn put(&mut self, b: &[u8]) -> Result<(), String> {
        self.sha.update(b);
        self.n += b.len() as u64;
        self.w.write_all(b).map_err(|e| e.to_string())
    }
    fn pad(&mut self, alignment: u64) -> Result<(), String> {
        let rem = (self.n % alignment) as usize;
        if rem != 0 {
            self.put(&vec![0u8; alignment as usize - rem])?;
        }
        Ok(())
    }
}

fn put_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn string_kv(key: &str, value: &str) -> Vec<u8> {
    let mut b = Vec::new();
    put_string(&mut b, key);
    b.extend_from_slice(&8u32.to_le_bytes());
    put_string(&mut b, value);
    b
}

/// Write `out` as a copy of `src` in which the planned tensors are replaced by what `produce`
/// returns. Tensors are produced and written one at a time, in file order.
pub fn rewrite<E>(
    src: &Gguf,
    out: &str,
    plan: &Plan,
    mut produce: impl FnMut(&TensorInfo) -> Result<Tensor, E>,
) -> Result<Report, Fail<E>> {
    for name in plan.edits.keys().chain(plan.drop.iter()) {
        src.tensor(name)?;
    }
    if let Some(n) = plan.edits.keys().find(|n| plan.drop.contains(*n)) {
        return Err(format!("tensor '{n}' is both edited and dropped").into());
    }
    if let (Ok(a), Ok(b)) = (fs::canonicalize(&src.path), fs::canonicalize(out)) {
        if a == b {
            return Err("the output must not be the source file".to_string().into());
        }
    }
    // ---- metadata
    let mut kv_bytes: Vec<Vec<u8>> = Vec::new();
    let mut replaced = HashSet::new();
    for kv in &src.kvs {
        match plan.meta.iter().rev().find(|(k, _)| *k == kv.key) {
            Some((k, v)) => {
                if kv.ty != 8 {
                    return Err(format!("metadata '{k}' is not a string in the source").into());
                }
                if k == "general.alignment" {
                    return Err("general.alignment cannot be changed".to_string().into());
                }
                replaced.insert(k.clone());
                kv_bytes.push(string_kv(k, v));
            }
            None => kv_bytes.push(src.header[kv.raw.clone()].to_vec()),
        }
    }
    let mut added = HashSet::new();
    for (k, v) in &plan.meta {
        if replaced.contains(k) {
            continue;
        }
        if !added.insert(k.clone()) {
            return Err(format!("metadata '{k}' is set twice").into());
        }
        kv_bytes.push(string_kv(k, v));
    }
    // ---- tensor table
    let kept: Vec<&TensorInfo> = src
        .tensors
        .iter()
        .filter(|t| !plan.drop.contains(&t.name))
        .collect();
    let mut head = Vec::new();
    head.extend_from_slice(MAGIC);
    head.extend_from_slice(&src.version.to_le_bytes());
    head.extend_from_slice(&(kept.len() as u64).to_le_bytes());
    head.extend_from_slice(&(kv_bytes.len() as u64).to_le_bytes());
    for b in &kv_bytes {
        head.extend_from_slice(b);
    }
    let mut sizes = Vec::with_capacity(kept.len());
    let mut offset = 0u64;
    for t in &kept {
        let (ty, nbytes) = match plan.edits.get(&t.name) {
            Some(&ty) => {
                let n = exact_bytes(ty, t.count()).ok_or_else(|| {
                    format!(
                        "tensor '{}' cannot be written as {} (Wisp writes F32, F64, F16, BF16 and Q8_0; Q8_0 needs a multiple of 32 values)",
                        t.name,
                        type_name(ty)
                    )
                })?;
                if ty == T_Q8_0 && t.dims.first().is_some_and(|d| d % 32 != 0) {
                    return Err(format!(
                        "tensor '{}' cannot be Q8_0: its row length is not a multiple of 32",
                        t.name
                    )
                    .into());
                }
                (ty, n)
            }
            None => (t.ty, t.nbytes),
        };
        put_string(&mut head, &t.name);
        head.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            head.extend_from_slice(&d.to_le_bytes());
        }
        head.extend_from_slice(&ty.to_le_bytes());
        head.extend_from_slice(&offset.to_le_bytes());
        sizes.push((ty, nbytes));
        offset = (offset + nbytes).div_ceil(src.alignment) * src.alignment;
    }
    // ---- data
    let partial = format!("{out}.partial");
    let file = fs::File::create(&partial).map_err(|e| format!("{partial}: {e}"))?;
    let mut sink = Sink {
        w: BufWriter::with_capacity(8 << 20, file),
        sha: Sha256::new(),
        n: 0,
    };
    let result = (|| -> Result<(usize, usize), Fail<E>> {
        sink.put(&head)?;
        sink.pad(src.alignment)?;
        let mut input = fs::File::open(&src.path).map_err(|e| e.to_string())?;
        let mut buf = vec![0u8; 8 << 20];
        let (mut edited, mut copied) = (0, 0);
        for (t, &(ty, nbytes)) in kept.iter().zip(&sizes) {
            if plan.edits.contains_key(&t.name) {
                let x = produce(t).map_err(Fail::Inner)?;
                if x.data.len() as u64 != t.count() {
                    return Err(format!(
                        "edit of '{}' returned {} values, the tensor has {}",
                        t.name,
                        x.data.len(),
                        t.count()
                    )
                    .into());
                }
                if x.data.iter().any(|v| !v.is_finite()) {
                    return Err(format!("edit of '{}' returned a non-finite value", t.name).into());
                }
                let bytes = encode(ty, &x.data).ok_or("unsupported output type".to_string())?;
                debug_assert_eq!(bytes.len() as u64, nbytes);
                drop(x);
                sink.put(&bytes)?;
                edited += 1;
            } else {
                input
                    .seek(SeekFrom::Start(src.data_start + t.offset))
                    .map_err(|e| e.to_string())?;
                let mut left = nbytes;
                while left > 0 {
                    let n = left.min(buf.len() as u64) as usize;
                    input
                        .read_exact(&mut buf[..n])
                        .map_err(|e| format!("{}: {e}", src.path))?;
                    sink.put(&buf[..n])?;
                    left -= n as u64;
                }
                copied += 1;
            }
            sink.pad(src.alignment)?;
        }
        sink.w.flush().map_err(|e| e.to_string())?;
        Ok((edited, copied))
    })();
    let (edited, copied) = match result {
        Ok(v) => v,
        Err(e) => {
            drop(sink);
            let _ = fs::remove_file(&partial);
            return Err(e);
        }
    };
    let bytes = sink.n;
    let sha256: String = sink
        .sha
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    drop(sink.w);
    fs::rename(&partial, out).map_err(|e| format!("{out}: {e}"))?;
    Ok(Report {
        edited,
        dropped: src.tensors.len() - kept.len(),
        copied,
        bytes,
        sha256,
    })
}

// ---- builtins

pub const NAMES: &[&str] = &[
    "gguf_names",
    "gguf_type",
    "gguf_size",
    "gguf_meta",
    "gguf_read",
    "gguf_rewrite",
];

fn io(span: Span) -> impl Fn(String) -> wisp_syntax::Diag {
    move |m| err("E-IO", m, span)
}

pub(crate) fn call(it: &mut Interp, name: &str, a: &[Value], span: Span) -> R<Vec<Value>> {
    let want = match name {
        "gguf_names" => 1,
        "gguf_rewrite" => 5,
        _ => 2,
    };
    if a.len() < want {
        return Err(err(
            "E-ARGS",
            format!("`{name}` needs at least {want} argument(s)"),
            span,
        ));
    }
    let path = string(&a[0], span, "the GGUF file")?;
    let g = Gguf::open(&path).map_err(io(span))?;
    let one = |v: Value| Ok(vec![v]);
    match name {
        "gguf_names" => one(Value::cell(
            g.tensors.iter().map(|t| Value::str(&t.name)).collect(),
        )),
        "gguf_type" => {
            let t = g
                .tensor(&string(&a[1], span, "the tensor name")?)
                .map_err(io(span))?;
            one(Value::str(type_name(t.ty)))
        }
        "gguf_size" => {
            let t = g
                .tensor(&string(&a[1], span, "the tensor name")?)
                .map_err(io(span))?;
            one(Value::Arr(Tensor::row(
                t.shape().iter().map(|&d| d as f64).collect(),
            )))
        }
        "gguf_meta" => {
            let key = string(&a[1], span, "the metadata key")?;
            match (g.kv(&key), a.get(2)) {
                (Some(kv), _) => one(g.meta_value(kv).map_err(io(span))?),
                (None, Some(default)) => one(default.clone()),
                (None, None) => Err(err(
                    "E-IO",
                    format!("{path}: no metadata key '{key}'"),
                    span,
                )),
            }
        }
        "gguf_read" => {
            let t = g
                .read(&string(&a[1], span, "the tensor name")?)
                .map_err(io(span))?;
            one(Value::Arr(t))
        }
        "gguf_rewrite" => rewrite_call(it, &g, a, span),
        _ => unreachable!(),
    }
}

/// gguf_rewrite(src, out, names, types, @edit, 'drop', {...}, 'meta', {key, value, ...})
fn rewrite_call(it: &mut Interp, g: &Gguf, a: &[Value], span: Span) -> R<Vec<Value>> {
    let out = string(&a[1], span, "the output file")?;
    let names = cell_strings(&a[2], span, "the tensor names")?;
    let types = match &a[3] {
        Value::Str(s) => vec![s.to_string(); names.len()],
        other => cell_strings(other, span, "the tensor types")?,
    };
    if types.len() != names.len() {
        return Err(err(
            "E-ARGS",
            "gguf_rewrite needs one type per tensor name, or a single type",
            span,
        ));
    }
    let Value::Func(edit) = &a[4] else {
        return Err(err(
            "E-TYPE",
            "the fifth argument of gguf_rewrite must be a function of the tensor name",
            span,
        ));
    };
    let mut plan = Plan {
        edits: IndexMap::new(),
        drop: HashSet::new(),
        meta: Vec::new(),
    };
    for (n, t) in names.iter().zip(&types) {
        let ty = type_from_name(t)
            .ok_or_else(|| err("E-ARGS", format!("unknown tensor type '{t}'"), span))?;
        if plan.edits.insert(n.clone(), ty).is_some() {
            return Err(err("E-ARGS", format!("tensor '{n}' is listed twice"), span));
        }
    }
    for (key, value) in opts(a, 5, span)? {
        match key.as_str() {
            "drop" => plan.drop = cell_strings(&value, span, "'drop'")?.into_iter().collect(),
            "meta" => {
                let flat = cell_strings(&value, span, "'meta'")?;
                if flat.len() % 2 != 0 {
                    return Err(err(
                        "E-ARGS",
                        "'meta' must be a cell of key, value string pairs",
                        span,
                    ));
                }
                plan.meta = flat.chunks(2).map(|p| (p[0].clone(), p[1].clone())).collect();
            }
            other => {
                return Err(err(
                    "E-ARGS",
                    format!("unsupported gguf_rewrite option '{other}'"),
                    span,
                ))
            }
        }
    }
    let edit = edit.clone();
    let report = rewrite(g, &out, &plan, |t| {
        let v = it.call_value(&edit, vec![Value::str(&t.name)], 1, span)?;
        let v = v.into_iter().next().ok_or_else(|| {
            err(
                "E-TYPE",
                format!("the edit function returned nothing for '{}'", t.name),
                span,
            )
        })?;
        let x = arr(it, &v, span)?;
        if t.dims.len() >= 2 && x.shape != t.shape() {
            return Err(err(
                "E-SHAPE",
                format!(
                    "edit of '{}' has shape {}, the tensor is {:?}",
                    t.name,
                    x.shape_str(),
                    t.shape()
                ),
                span,
            ));
        }
        Ok(x)
    })
    .map_err(|f| match f {
        Fail::Msg(m) => err("E-IO", m, span),
        Fail::Inner(d) => d,
    })?;
    it.print(&format!(
        "gguf_rewrite: {out}: {} edited, {} dropped, {} copied byte for byte; {} bytes\n",
        report.edited, report.dropped, report.copied, report.bytes
    ));
    let mut s = IndexMap::new();
    s.insert("edited".to_string(), Value::num(report.edited as f64));
    s.insert("dropped".to_string(), Value::num(report.dropped as f64));
    s.insert("copied".to_string(), Value::num(report.copied as f64));
    s.insert("bytes".to_string(), Value::num(report.bytes as f64));
    s.insert("sha256".to_string(), Value::str(report.sha256));
    Ok(vec![Value::strukt(s)])
}
