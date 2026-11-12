//! Native CPU FP32 executor for the bounded OpenThai2 recognizer graph.
//! No ONNX runtime, subprocesses, or cached activations. Each call reads live parts.
use crate::{
    net::NetData,
    tensor::{Tag, Tensor},
};
use candle_core::{Device, Tensor as C};
use indexmap::IndexMap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value as J;
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Graph {
    pub version: u32,
    pub input: String,
    pub output: String,
    pub nodes: Vec<Node>,
    pub weights: BTreeMap<String, Weight>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Weight {
    pub shape: Vec<usize>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Node {
    pub op: String,
    #[serde(rename = "in")]
    pub inputs: Vec<String>,
    pub out: String,
    pub attrs: J,
}
const MAX_ELEMENTS: usize = 64 * 1024 * 1024;
fn count(shape: &[usize]) -> Result<usize, String> {
    let n = shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or("OCR shape overflow")?;
    if n == 0 || n > MAX_ELEMENTS {
        return Err("OCR tensor exceeds element limit".into());
    }
    Ok(n)
}
impl Graph {
    pub fn validate(&self) -> Result<(), String> {
        if self.version != 1
            || self.nodes.is_empty()
            || self.nodes.len() > 256
            || self.weights.len() > 256
        {
            return Err("unsupported OCR graph version/size".into());
        }
        let mut seen = HashSet::from([self.input.clone()]);
        for (name, w) in &self.weights {
            count(&w.shape)?;
            if w.shape.len() > 4 || !seen.insert(name.clone()) {
                return Err("invalid OCR weight declaration".into());
            }
        }
        for n in &self.nodes {
            let (lo, hi, allowed): (usize, usize, &[&str]) = match n.op.as_str() {
                "Conv" => (
                    2,
                    3,
                    &["kernel_shape", "strides", "pads", "dilations", "group"],
                ),
                "Add" | "Mul" | "Div" | "MatMul" => (2, 2, &[]),
                "Erf" | "Relu" | "HardSwish" | "GlobalAveragePool" => (1, 1, &[]),
                "HardSigmoid" => (1, 1, &["alpha", "beta"]),
                "AveragePool" => (
                    1,
                    1,
                    &[
                        "kernel_shape",
                        "strides",
                        "pads",
                        "ceil_mode",
                        "count_include_pad",
                    ],
                ),
                "Squeeze" => (1, 1, &["axes"]),
                "Transpose" => (1, 1, &["perm"]),
                "Softmax" => (1, 1, &["axis"]),
                "Cast" => (1, 1, &["to"]),
                _ => return Err(format!("unsupported OCR operator {}", n.op)),
            };
            let attrs = n.attrs.as_object().ok_or("OCR attrs must be an object")?;
            if attrs.keys().any(|k| !allowed.contains(&k.as_str()))
                || n.inputs.len() < lo
                || n.inputs.len() > hi
            {
                return Err(format!("invalid OCR attributes/arity: {}", n.op));
            }
            if n.inputs.iter().any(|i| !seen.contains(i)) || !seen.insert(n.out.clone()) {
                return Err("OCR graph is not topological or has duplicate values".into());
            }
        }
        if !self.nodes.iter().any(|n| n.out == self.output) {
            return Err("missing OCR output".into());
        }
        Ok(())
    }
}
fn ints(a: &J, key: &str, default: Vec<usize>) -> candle_core::Result<Vec<usize>> {
    match a.get(key) {
        None => Ok(default),
        Some(v) => serde_json::from_value(v.clone())
            .map_err(|e| candle_core::Error::Msg(format!("{key}: {e}"))),
    }
}
fn integer(a: &J, key: &str, default: usize) -> candle_core::Result<usize> {
    match a.get(key) {
        None => Ok(default),
        Some(v) => v
            .as_u64()
            .and_then(|x| usize::try_from(x).ok())
            .ok_or_else(|| candle_core::Error::Msg(format!("invalid {key}"))),
    }
}
fn check(ok: bool, message: &str) -> candle_core::Result<()> {
    if !ok {
        candle_core::bail!("{message}");
    }
    Ok(())
}
fn bounded(shape: &[usize]) -> candle_core::Result<usize> {
    count(shape).map_err(candle_core::Error::Msg)
}
fn flat(t: &C) -> candle_core::Result<Vec<f32>> {
    t.flatten_all()?.to_vec1::<f32>()
}

// Explicit NCHW grouped kernels support unequal spatial strides, unlike
// Candle's scalar-stride Conv2d API. Accumulation and stored results are FP32.
fn conv(x: &C, w: &C, bias: Option<&C>, a: &J) -> candle_core::Result<C> {
    let rank = x.rank().checked_sub(2).unwrap_or(0);
    check(
        (rank == 1 || rank == 2) && w.rank() == rank + 2,
        "Conv rank",
    )?;
    let k = ints(a, "kernel_shape", vec![])?;
    let s = ints(a, "strides", vec![1; rank])?;
    let p = ints(a, "pads", vec![0; rank * 2])?;
    let d = ints(a, "dilations", vec![1; rank])?;
    let groups = integer(a, "group", 1)?;
    check(
        k.len() == rank && s.len() == rank && p.len() == rank * 2 && d.len() == rank,
        "Conv attributes",
    )?;
    check(
        k.iter().all(|&v| v > 0 && v <= 15)
            && s.iter().all(|&v| v > 0 && v <= 16)
            && d.iter().all(|&v| v > 0 && v <= 16)
            && p.iter().all(|&v| v <= 32),
        "Conv limits",
    )?;
    let (n, ci, co) = (x.dims()[0], x.dims()[1], w.dims()[0]);
    check(
        groups > 0
            && ci % groups == 0
            && co % groups == 0
            && w.dims()[1] == ci / groups
            && w.dims()[2..] == k,
        "Conv channels/kernel",
    )?;
    let spatial = &x.dims()[2..];
    let mut out = vec![n, co];
    for i in 0..rank {
        let extent = d[i] * (k[i] - 1) + 1;
        check(
            spatial[i] + p[i] + p[i + rank] >= extent,
            "Conv kernel exceeds input",
        )?;
        out.push((spatial[i] + p[i] + p[i + rank] - extent) / s[i] + 1);
    }
    let size = bounded(&out)?;
    let (ih, iw, oh, ow, kh, kw, sy, sx, py, px, dy, dx) = if rank == 2 {
        (
            spatial[0], spatial[1], out[2], out[3], k[0], k[1], s[0], s[1], p[0], p[1], d[0], d[1],
        )
    } else {
        (1, spatial[0], 1, out[2], 1, k[0], 1, s[0], 0, p[0], 1, d[0])
    };
    let xv = flat(x)?;
    let wv = flat(w)?;
    let bv = match bias {
        Some(b) => {
            check(b.dims() == [co], "Conv bias shape")?;
            flat(b)?
        }
        None => vec![0.; co],
    };
    let mut y = vec![0f32; size];
    y.par_chunks_mut(oh * ow)
        .enumerate()
        .for_each(|(channel, plane)| {
            let batch = channel / co;
            let oc = channel % co;
            let ic0 = (oc / (co / groups)) * (ci / groups);
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut sum = 0f32;
                    for ic in 0..ci / groups {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = oy * sy + ky * dy;
                                let ix = ox * sx + kx * dx;
                                if iy >= py && ix >= px && iy - py < ih && ix - px < iw {
                                    sum += xv
                                        [((batch * ci + ic0 + ic) * ih + iy - py) * iw + ix - px]
                                        * wv[((oc * (ci / groups) + ic) * kh + ky) * kw + kx];
                                }
                            }
                        }
                    }
                    plane[oy * ow + ox] = sum + bv[oc];
                }
            }
        });
    C::from_vec(y, out, &Device::Cpu)
}
fn execute(n: &Node, v: &HashMap<String, C>) -> candle_core::Result<C> {
    let x = &v[&n.inputs[0]];
    let a = &n.attrs;
    let b = || &v[&n.inputs[1]];
    match n.op.as_str() {
        "Conv" => conv(x, b(), n.inputs.get(2).map(|i| &v[i]), a),
        "Add" | "Mul" | "Div" => {
            let other = b();
            let rank = x.rank().max(other.rank());
            let mut shape = vec![1; rank];
            for (tensor, offset) in [(x, rank - x.rank()), (other, rank - other.rank())] {
                for (i, &d) in tensor.dims().iter().enumerate() {
                    let current = &mut shape[offset + i];
                    check(
                        *current == 1 || d == 1 || *current == d,
                        "incompatible broadcast",
                    )?;
                    *current = (*current).max(d);
                }
            }
            bounded(&shape)?;
            match n.op.as_str() {
                "Add" => x.broadcast_add(other),
                "Mul" => x.broadcast_mul(other),
                _ => x.broadcast_div(other),
            }
        }
        "Erf" => x.erf(),
        "Relu" => x.relu(),
        "Cast" => {
            check(integer(a, "to", 0)? == 1, "only FP32 Cast supported")?;
            Ok(x.clone())
        }
        "HardSigmoid" => {
            let alpha = a
                .get("alpha")
                .map_or(Some(0.2), J::as_f64)
                .ok_or_else(|| candle_core::Error::Msg("invalid alpha".into()))?;
            let beta = a
                .get("beta")
                .map_or(Some(0.5), J::as_f64)
                .ok_or_else(|| candle_core::Error::Msg("invalid beta".into()))?;
            x.affine(alpha, beta)?.clamp(0f32, 1f32)
        }
        "HardSwish" => x.mul(&x.affine(1., 3.)?.clamp(0f32, 6f32)?.affine(1. / 6., 0.)?),
        "GlobalAveragePool" => {
            check(x.rank() == 4, "GlobalAveragePool rank")?;
            x.mean_keepdim((2, 3))
        }
        "AveragePool" => {
            let k = ints(a, "kernel_shape", vec![])?;
            let s = ints(a, "strides", vec![1, 1])?;
            check(
                x.rank() == 4 && k.len() == 2 && s.len() == 2 && k.iter().chain(&s).all(|&i| i > 0),
                "AveragePool dimensions",
            )?;
            check(
                ints(a, "pads", vec![0; 4])? == [0; 4]
                    && integer(a, "ceil_mode", 0)? == 0
                    && integer(a, "count_include_pad", 0)? <= 1,
                "AveragePool padding/mode",
            )?;
            check(
                k[0] <= x.dims()[2] && k[1] <= x.dims()[3],
                "AveragePool kernel exceeds input",
            )?;
            x.avg_pool2d_with_stride((k[0], k[1]), (s[0], s[1]))
        }
        "Squeeze" => {
            check(
                x.rank() == 4 && ints(a, "axes", vec![])? == [2],
                "Squeeze axes",
            )?;
            x.squeeze(2)
        }
        "Transpose" => {
            check(
                x.rank() == 3 && ints(a, "perm", vec![])? == [0, 2, 1],
                "Transpose permutation",
            )?;
            x.transpose(1, 2)?.contiguous()
        }
        "MatMul" => {
            check(x.rank() == 3 && b().rank() == 2, "MatMul rank")?;
            let (batch, time, inner) = x.dims3()?;
            let (wi, wo) = b().dims2()?;
            check(inner == wi, "MatMul inner dimension")?;
            bounded(&[batch, time, wo])?;
            x.reshape((batch * time, inner))?
                .matmul(b())?
                .reshape((batch, time, wo))
        }
        "Softmax" => {
            let axis = match a.get("axis") {
                None => Some(-1),
                Some(v) => v.as_i64(),
            };
            check(
                x.rank() == 3 && matches!(axis, Some(-1 | 2)),
                "Softmax axis",
            )?;
            candle_nn::ops::softmax(x, 2)
        }
        _ => candle_core::bail!("unsupported OCR op {}", n.op),
    }
}
fn observed(t: &C) -> Result<Tensor, String> {
    Ok(Tensor::new(
        t.dims().to_vec(),
        flat(t)
            .map_err(|e| e.to_string())?
            .into_iter()
            .map(f64::from)
            .collect(),
        Tag::Act,
    ))
}

pub fn forward(
    net: &NetData,
    graph: &Graph,
    input: &Tensor,
    capture: bool,
) -> Result<(Tensor, IndexMap<String, Tensor>), String> {
    graph.validate()?;
    if input.shape.len() != 4
        || !(1..=16).contains(&input.shape[0])
        || input.shape[1] != 3
        || input.shape[2] != 48
        || !(16..=3200).contains(&input.shape[3])
    {
        return Err("OpenThai2 expects normalized NCHW [N,3,48,W], N=1..16, W=16..3200".into());
    }
    if count(&input.shape)? != input.data.len()
        || input
            .data
            .iter()
            .any(|x| !x.is_finite() || !(*x as f32).is_finite())
    {
        return Err("invalid OCR input data".into());
    }
    let mut values = HashMap::new();
    let make = |data: Vec<f32>, shape: &[usize]| {
        C::from_vec(data, shape, &Device::Cpu).map_err(|e| e.to_string())
    };
    values.insert(
        graph.input.clone(),
        make(input.data.iter().map(|&x| x as f32).collect(), &input.shape)?,
    );
    for (name, w) in &graph.weights {
        let part = net
            .parts
            .get(name)
            .ok_or_else(|| format!("missing OCR weight {name}"))?;
        if part.shape != w.shape
            || part.data.len() != count(&w.shape)?
            || part
                .data
                .iter()
                .any(|v| !v.is_finite() || !(*v as f32).is_finite())
        {
            return Err(format!("invalid OCR weight {name}"));
        }
        values.insert(
            name.clone(),
            make(part.data.iter().map(|&x| x as f32).collect(), &w.shape)?,
        );
    }
    let mut uses: HashMap<String, usize> = HashMap::new();
    for n in &graph.nodes {
        for i in &n.inputs {
            *uses.entry(i.clone()).or_default() += 1;
        }
    }
    let mut states = IndexMap::new();
    let mut captured = 0;
    for n in &graph.nodes {
        let y = execute(n, &values).map_err(|e| format!("OCR {} ({}): {e}", n.out, n.op))?;
        count(y.dims())?;
        if capture {
            captured += y.elem_count();
            if captured > MAX_ELEMENTS {
                return Err("OCR capture exceeds element limit; use smaller inputs".into());
            }
            states.insert(n.out.clone(), observed(&y)?);
        }
        values.insert(n.out.clone(), y);
        for i in &n.inputs {
            let left = uses.get_mut(i).unwrap();
            *left -= 1;
            if *left == 0 && i != &graph.output {
                values.remove(i);
            }
        }
    }
    let y = &values[&graph.output];
    if y.rank() != 3 || y.dims()[0] != input.shape[0] || y.dims()[2] != 530 {
        return Err("invalid OpenThai2 output shape".into());
    }
    Ok((observed(y)?, states))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn grouped_conv_unequal_stride_and_bias() {
        let x = C::from_vec(
            (0..24).map(|x| x as f32).collect::<Vec<_>>(),
            (1, 2, 3, 4),
            &Device::Cpu,
        )
        .unwrap();
        let w = C::ones((2, 1, 1, 1), candle_core::DType::F32, &Device::Cpu).unwrap();
        let b = C::from_vec(vec![1f32, -1.], 2, &Device::Cpu).unwrap();
        let y = conv(
            &x,
            &w,
            Some(&b),
            &serde_json::json!({"kernel_shape":[1,1],"strides":[2,1],"group":2}),
        )
        .unwrap();
        assert_eq!(y.dims(), [1, 2, 2, 4]);
        assert_eq!(
            flat(&y).unwrap(),
            vec![1., 2., 3., 4., 9., 10., 11., 12., 11., 12., 13., 14., 19., 20., 21., 22.]
        );
    }
    #[test]
    fn depthwise_conv1d_padding() {
        let x = C::from_vec(vec![1f32, 2., 3., 4., 5., 6.], (1, 2, 3), &Device::Cpu).unwrap();
        let w = C::ones((2, 1, 3), candle_core::DType::F32, &Device::Cpu).unwrap();
        let y = conv(
            &x,
            &w,
            None,
            &serde_json::json!({"kernel_shape":[3],"pads":[1,1],"group":2}),
        )
        .unwrap();
        assert_eq!(flat(&y).unwrap(), vec![3., 6., 5., 9., 15., 11.]);
    }
}
