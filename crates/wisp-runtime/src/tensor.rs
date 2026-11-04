//! Dense row-major float64 arrays with a provenance tag.

use std::fmt;

/// How the numbers in an array were obtained. Ordered from least to most observed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tag {
    Lit,
    W,
    Alg,
    Act,
    Grad,
}

impl Tag {
    /// Tag of a value computed from inputs with tags `a` and `b`.
    pub fn combine(a: Tag, b: Tag) -> Tag {
        let m = a.max(b);
        if m == Tag::W {
            Tag::Alg
        } else {
            m
        }
    }
    /// Tag after any operation on a single input.
    pub fn derived(a: Tag) -> Tag {
        if a == Tag::W {
            Tag::Alg
        } else {
            a
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Tag::Lit => "Lit",
            Tag::W => "W",
            Tag::Alg => "Alg",
            Tag::Act => "Act",
            Tag::Grad => "Grad",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
    pub tag: Tag,
    pub logical: bool,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f64>, tag: Tag) -> Self {
        debug_assert_eq!(shape.iter().product::<usize>(), data.len());
        Tensor {
            shape,
            data,
            tag,
            logical: false,
        }
    }
    pub fn scalar(v: f64) -> Self {
        Tensor::new(vec![1, 1], vec![v], Tag::Lit)
    }
    pub fn boolean(b: bool) -> Self {
        let mut t = Tensor::scalar(if b { 1.0 } else { 0.0 });
        t.logical = true;
        t
    }
    pub fn zeros(shape: Vec<usize>) -> Self {
        let n = shape.iter().product();
        Tensor::new(shape, vec![0.0; n], Tag::Lit)
    }
    pub fn empty() -> Self {
        Tensor::new(vec![0, 0], vec![], Tag::Lit)
    }
    pub fn row(v: Vec<f64>) -> Self {
        Tensor::new(vec![1, v.len()], v, Tag::Lit)
    }
    pub fn col(v: Vec<f64>) -> Self {
        Tensor::new(vec![v.len(), 1], v, Tag::Lit)
    }
    pub fn with_tag(mut self, tag: Tag) -> Self {
        self.tag = tag;
        self
    }
    pub fn numel(&self) -> usize {
        self.data.len()
    }
    pub fn is_scalar(&self) -> bool {
        self.data.len() == 1
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    pub fn rows(&self) -> usize {
        self.shape.first().copied().unwrap_or(1)
    }
    pub fn cols(&self) -> usize {
        if self.shape.len() >= 2 {
            self.shape[1..].iter().product()
        } else {
            1
        }
    }
    pub fn is_vector(&self) -> bool {
        self.shape.len() == 2 && (self.shape[0] == 1 || self.shape[1] == 1)
    }
    pub fn scalar_value(&self) -> Option<f64> {
        if self.data.len() == 1 {
            Some(self.data[0])
        } else {
            None
        }
    }
    pub fn shape_str(&self) -> String {
        shape_str(&self.shape)
    }

    pub fn map(&self, f: impl Fn(f64) -> f64) -> Tensor {
        Tensor::new(
            self.shape.clone(),
            self.data.iter().map(|&x| f(x)).collect(),
            Tag::derived(self.tag),
        )
    }

    pub fn transpose(&self) -> Result<Tensor, String> {
        if self.shape.len() != 2 {
            return Err(format!(
                "transpose needs a 2-D array, got {}",
                self.shape_str()
            ));
        }
        let (r, c) = (self.shape[0], self.shape[1]);
        let mut d = vec![0.0; r * c];
        for i in 0..r {
            for j in 0..c {
                d[j * r + i] = self.data[i * c + j];
            }
        }
        let mut t = Tensor::new(vec![c, r], d, Tag::derived(self.tag));
        t.logical = self.logical;
        Ok(t)
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Result<Tensor, String> {
        let n: usize = shape.iter().product();
        if n != self.numel() {
            return Err(format!(
                "cannot reshape {} elements into {}",
                self.numel(),
                shape_str(&shape)
            ));
        }
        let mut t = Tensor::new(shape, self.data.clone(), self.tag);
        t.logical = self.logical;
        Ok(t)
    }

    /// Elementwise binary operation with NumPy broadcasting (right-aligned).
    pub fn zip(&self, other: &Tensor, f: impl Fn(f64, f64) -> f64) -> Result<Tensor, String> {
        let tag = Tag::combine(self.tag, other.tag);
        if self.shape == other.shape {
            let d = self
                .data
                .iter()
                .zip(&other.data)
                .map(|(&a, &b)| f(a, b))
                .collect();
            return Ok(Tensor::new(self.shape.clone(), d, tag));
        }
        if other.is_scalar() {
            let b = other.data[0];
            return Ok(Tensor::new(
                self.shape.clone(),
                self.data.iter().map(|&a| f(a, b)).collect(),
                tag,
            ));
        }
        if self.is_scalar() {
            let a = self.data[0];
            return Ok(Tensor::new(
                other.shape.clone(),
                other.data.iter().map(|&b| f(a, b)).collect(),
                tag,
            ));
        }
        let shape = broadcast_shape(&self.shape, &other.shape).ok_or_else(|| {
            format!(
                "shapes {} and {} do not broadcast",
                self.shape_str(),
                other.shape_str()
            )
        })?;
        let a = self.broadcast_to(&shape).unwrap();
        let b = other.broadcast_to(&shape).unwrap();
        let d = a.iter().zip(&b).map(|(&x, &y)| f(x, y)).collect();
        Ok(Tensor::new(shape, d, tag))
    }

    /// The data broadcast to `shape`, or None if impossible.
    pub fn broadcast_to(&self, shape: &[usize]) -> Option<Vec<f64>> {
        if self.shape == shape {
            return Some(self.data.clone());
        }
        let n: usize = shape.iter().product();
        if self.is_scalar() {
            return Some(vec![self.data[0]; n]);
        }
        let r = shape.len();
        if self.shape.len() > r {
            // allow extra leading singleton dimensions
            let extra = self.shape.len() - r;
            if self.shape[..extra].iter().any(|&d| d != 1) {
                return None;
            }
            let t = Tensor::new(self.shape[extra..].to_vec(), self.data.clone(), self.tag);
            return t.broadcast_to(shape);
        }
        let mut src = vec![1usize; r - self.shape.len()];
        src.extend_from_slice(&self.shape);
        for k in 0..r {
            if src[k] != shape[k] && src[k] != 1 {
                return None;
            }
        }
        let sstr = strides(&src);
        let mut out = Vec::with_capacity(n);
        let mut idx = vec![0usize; r];
        for _ in 0..n {
            let mut off = 0;
            for k in 0..r {
                if src[k] != 1 {
                    off += idx[k] * sstr[k];
                }
            }
            out.push(self.data[off]);
            // increment
            for k in (0..r).rev() {
                idx[k] += 1;
                if idx[k] < shape[k] {
                    break;
                }
                idx[k] = 0;
            }
        }
        Some(out)
    }

    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, String> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(format!(
                "matrix product needs 2-D arrays, got {} and {}",
                self.shape_str(),
                other.shape_str()
            ));
        }
        let (m, k) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);
        if k != k2 {
            return Err(format!(
                "inner dimensions must agree: {} * {} (use .* for elementwise products)",
                self.shape_str(),
                other.shape_str()
            ));
        }
        let mut d = vec![0.0; m * n];
        for i in 0..m {
            for p in 0..k {
                let a = self.data[i * k + p];
                if a == 0.0 {
                    continue;
                }
                let brow = &other.data[p * n..(p + 1) * n];
                let drow = &mut d[i * n..(i + 1) * n];
                for j in 0..n {
                    drow[j] += a * brow[j];
                }
            }
        }
        Ok(Tensor::new(
            vec![m, n],
            d,
            Tag::combine(self.tag, other.tag),
        ))
    }

    /// Solve A x = B for square A by Gaussian elimination with partial pivoting.
    pub fn left_divide(&self, b: &Tensor) -> Result<Tensor, String> {
        if self.shape.len() != 2 || self.shape[0] != self.shape[1] {
            return Err(format!("A \\ B needs a square A, got {}", self.shape_str()));
        }
        let n = self.shape[0];
        if b.shape.len() != 2 || b.shape[0] != n {
            return Err(format!(
                "A \\ B needs B with {} rows, got {}",
                n,
                b.shape_str()
            ));
        }
        let m = b.shape[1];
        let mut a = self.data.clone();
        let mut x = b.data.clone();
        for col in 0..n {
            let mut piv = col;
            for r in col + 1..n {
                if a[r * n + col].abs() > a[piv * n + col].abs() {
                    piv = r;
                }
            }
            if a[piv * n + col].abs() < 1e-300 {
                return Err("matrix is singular".into());
            }
            if piv != col {
                for j in 0..n {
                    a.swap(col * n + j, piv * n + j);
                }
                for j in 0..m {
                    x.swap(col * m + j, piv * m + j);
                }
            }
            let p = a[col * n + col];
            for r in col + 1..n {
                let f = a[r * n + col] / p;
                if f == 0.0 {
                    continue;
                }
                for j in col..n {
                    a[r * n + j] -= f * a[col * n + j];
                }
                for j in 0..m {
                    x[r * m + j] -= f * x[col * m + j];
                }
            }
        }
        for col in (0..n).rev() {
            let p = a[col * n + col];
            for j in 0..m {
                let mut s = x[col * m + j];
                for k in col + 1..n {
                    s -= a[col * n + k] * x[k * m + j];
                }
                x[col * m + j] = s / p;
            }
        }
        Ok(Tensor::new(vec![n, m], x, Tag::combine(self.tag, b.tag)))
    }
}

pub fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for k in (0..shape.len().saturating_sub(1)).rev() {
        s[k] = s[k + 1] * shape[k + 1];
    }
    s
}

pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let r = a.len().max(b.len());
    let mut out = vec![0; r];
    for k in 0..r {
        let da = if k + a.len() >= r {
            a[k + a.len() - r]
        } else {
            1
        };
        let db = if k + b.len() >= r {
            b[k + b.len() - r]
        } else {
            1
        };
        out[k] = if da == db {
            da
        } else if da == 1 {
            db
        } else if db == 1 {
            da
        } else {
            return None;
        };
    }
    Some(out)
}

pub fn shape_str(s: &[usize]) -> String {
    s.iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join("\u{d7}")
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_scalar() {
            return write!(f, "{}", fmt_num(self.data[0]));
        }
        if self.shape.len() != 2 {
            return write!(f, "[{} array, tag {}]", self.shape_str(), self.tag.name());
        }
        let (r, c) = (self.shape[0], self.shape[1]);
        if r * c > 400 {
            return write!(f, "[{} array, tag {}]", self.shape_str(), self.tag.name());
        }
        for i in 0..r {
            let row: Vec<String> = (0..c).map(|j| fmt_num(self.data[i * c + j])).collect();
            writeln!(f, "  {}", row.join("  "))?;
        }
        Ok(())
    }
}

pub fn fmt_num(v: f64) -> String {
    if v == v.trunc() && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        let s = format!("{:.6}", v);
        let s = s.trim_end_matches('0').trim_end_matches('.').to_string();
        if s == "-0" {
            "0".into()
        } else {
            s
        }
    }
}
