//! 0-based indexing with the region-shape rule from the specification.

use crate::tensor::{strides, Tensor};

#[derive(Clone, Debug)]
pub enum Idx {
    /// `:` over a dimension of the given size.
    All(usize),
    Scalar(usize),
    /// Positions with the shape of the index array that produced them.
    List(Vec<usize>, Vec<usize>),
}

impl Idx {
    pub fn len(&self) -> usize {
        match self {
            Idx::All(n) => *n,
            Idx::Scalar(_) => 1,
            Idx::List(p, _) => p.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn positions(&self) -> Vec<usize> {
        match self {
            Idx::All(n) => (0..*n).collect(),
            Idx::Scalar(i) => vec![*i],
            Idx::List(p, _) => p.clone(),
        }
    }
    pub fn max_pos(&self) -> Option<usize> {
        match self {
            Idx::All(n) => n.checked_sub(1),
            Idx::Scalar(i) => Some(*i),
            Idx::List(p, _) => p.iter().copied().max(),
        }
    }

    /// Convert an index value for a dimension of size `dim`.
    /// `allow_beyond` permits positions past the end (assignment that grows an array).
    pub fn from_tensor(t: &Tensor, dim: usize, allow_beyond: bool) -> Result<Idx, String> {
        if t.logical {
            if t.numel() != dim {
                return Err(format!(
                    "a logical index must have {dim} elements, got {}",
                    t.numel()
                ));
            }
            let p: Vec<usize> = t
                .data
                .iter()
                .enumerate()
                .filter(|(_, &v)| v != 0.0)
                .map(|(i, _)| i)
                .collect();
            let n = p.len();
            return Ok(Idx::List(p, vec![1, n]));
        }
        let mut p = Vec::with_capacity(t.numel());
        for &v in &t.data {
            if v < 0.0 || v != v.trunc() || !v.is_finite() {
                return Err(format!(
                    "index {} is not a non-negative integer (indices start at 0)",
                    crate::tensor::fmt_num(v)
                ));
            }
            let i = v as usize;
            if !allow_beyond && i >= dim {
                return Err(if dim == 0 {
                    format!("index {i} is out of range: the dimension is empty")
                } else {
                    format!("index {i} is out of range (valid: 0..{})", dim - 1)
                });
            }
            p.push(i);
        }
        if p.len() == 1 && t.numel() == 1 {
            return Ok(Idx::Scalar(p[0]));
        }
        Ok(Idx::List(p, t.shape.clone()))
    }
}

/// Shape of the region selected by `idxs`.
pub fn region_shape(idxs: &[Idx]) -> Vec<usize> {
    if idxs.len() == 1 {
        return match &idxs[0] {
            Idx::Scalar(_) => vec![1, 1],
            Idx::All(n) => vec![*n, 1],
            Idx::List(_, s) => norm_shape(s),
        };
    }
    let lists: Vec<&Idx> = idxs.iter().filter(|i| matches!(i, Idx::List(..))).collect();
    let alls = idxs.iter().filter(|i| matches!(i, Idx::All(_))).count();
    if lists.len() == 1 && alls == 0 {
        if let Idx::List(_, s) = lists[0] {
            return norm_shape(s);
        }
    }
    idxs.iter().map(|i| i.len()).collect()
}

fn norm_shape(s: &[usize]) -> Vec<usize> {
    match s.len() {
        0 => vec![1, 1],
        1 => vec![1, s[0]],
        _ => s.to_vec(),
    }
}

/// Flat positions in a container of shape `shape`, in row-major order over the index tuple.
pub fn flat_positions(idxs: &[Idx], shape: &[usize]) -> Result<Vec<usize>, String> {
    if idxs.len() == 1 {
        return Ok(idxs[0].positions());
    }
    // Allow trailing scalar zeros beyond the rank, and fewer indices than rank only when the rest are size 1.
    let r = shape.len();
    let k = idxs.len();
    let mut eff_shape = shape.to_vec();
    if k > r {
        for i in &idxs[r..] {
            if !matches!(i, Idx::Scalar(0)) {
                return Err(format!("{k} indices for a {r}-D array"));
            }
        }
        eff_shape.extend(std::iter::repeat_n(1, k - r));
    } else if k < r {
        if shape[k..].iter().any(|&d| d != 1) {
            return Err(format!(
                "{k} indices for a {r}-D array of shape {}",
                crate::tensor::shape_str(shape)
            ));
        }
        eff_shape.truncate(k);
    }
    let st = strides(&eff_shape);
    let lists: Vec<Vec<usize>> = idxs.iter().map(|i| i.positions()).collect();
    let total: usize = lists.iter().map(|l| l.len()).product();
    let mut out = Vec::with_capacity(total);
    if total == 0 {
        return Ok(out);
    }
    let mut c = vec![0usize; k];
    loop {
        let mut off = 0;
        for d in 0..k {
            off += lists[d][c[d]] * st[d];
        }
        out.push(off);
        let mut d = k;
        loop {
            if d == 0 {
                return Ok(out);
            }
            d -= 1;
            c[d] += 1;
            if c[d] < lists[d].len() {
                break;
            }
            c[d] = 0;
        }
    }
}

/// Read the region selected by `idxs` from `t`.
pub fn read(t: &Tensor, idxs: &[Idx]) -> Result<Tensor, String> {
    let pos = flat_positions(idxs, &t.shape)?;
    let shape = region_shape(idxs);
    let data = pos.iter().map(|&p| t.data[p]).collect();
    let mut out = Tensor::new(shape, data, t.tag);
    out.logical = t.logical;
    Ok(out)
}

/// Values to write into a region: broadcast, or same element count in row-major order.
pub fn values_for_region(v: &Tensor, region: &[usize]) -> Result<Vec<f64>, String> {
    let n: usize = region.iter().product();
    if let Some(d) = v.broadcast_to(region) {
        return Ok(d);
    }
    if v.numel() == n {
        return Ok(v.data.clone());
    }
    Err(format!(
        "cannot assign a {} value to a {} region",
        v.shape_str(),
        crate::tensor::shape_str(region)
    ))
}

/// Write `v` into `t` at `idxs`; `t` must already be large enough.
pub fn write(t: &mut Tensor, idxs: &[Idx], v: &Tensor) -> Result<(), String> {
    let pos = flat_positions(idxs, &t.shape)?;
    let region = region_shape(idxs);
    let vals = values_for_region(v, &region)?;
    for (p, x) in pos.iter().zip(vals) {
        t.data[*p] = x;
    }
    t.tag = t.tag.max(v.tag);
    Ok(())
}

/// Grow an array so that `idxs` fits (MATLAB-style growth on assignment).
pub fn grow_for(t: &mut Tensor, idxs: &[Idx]) -> Result<(), String> {
    if idxs.len() == 1 {
        let need = match idxs[0].max_pos() {
            Some(m) => m + 1,
            None => return Ok(()),
        };
        if need <= t.numel() {
            return Ok(());
        }
        let new_shape = if t.is_empty() || t.shape == vec![1, t.numel()] {
            vec![1, need]
        } else if t.shape.len() == 2 && t.shape[1] == 1 {
            vec![need, 1]
        } else {
            return Err(format!(
                "linear index {} is past the end of a {} array (it can grow only a vector)",
                need - 1,
                t.shape_str()
            ));
        };
        let mut d = t.data.clone();
        d.resize(need, 0.0);
        t.shape = new_shape;
        t.data = d;
        return Ok(());
    }
    let mut new_shape = t.shape.clone();
    while new_shape.len() < idxs.len() {
        new_shape.push(if t.is_empty() { 0 } else { 1 });
    }
    let mut changed = false;
    for (k, i) in idxs.iter().enumerate() {
        if let Some(m) = i.max_pos() {
            if m + 1 > new_shape[k] {
                new_shape[k] = m + 1;
                changed = true;
            }
        }
    }
    if !changed && new_shape.len() == t.shape.len() {
        return Ok(());
    }
    let mut nt = Tensor::zeros(new_shape.clone());
    nt.tag = t.tag;
    if !t.is_empty() {
        let old_idx: Vec<Idx> = t.shape.iter().map(|&d| Idx::All(d)).collect();
        let mut padded = old_idx;
        while padded.len() < new_shape.len() {
            padded.push(Idx::Scalar(0));
        }
        let pos = flat_positions(&padded, &new_shape)?;
        for (p, x) in pos.iter().zip(&t.data) {
            nt.data[*p] = *x;
        }
    }
    *t = nt;
    Ok(())
}
