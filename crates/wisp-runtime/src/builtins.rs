//! The standard library.

use crate::forward;
use crate::index;
use crate::interp::{err, Interp, R};
use crate::io;
use crate::net::{self, Arch, NetData, NetRef, TCfg};
use crate::tensor::{fmt_num, Tag, Tensor};
use crate::value::{FuncVal, Value};
use indexmap::IndexMap;
use std::rc::Rc;
use wisp_syntax::Span;

pub const NAMES: &[&str] = &[
    // numerics
    "size",
    "numel",
    "length",
    "rows",
    "cols",
    "zeros",
    "ones",
    "eye",
    "reshape",
    "sum",
    "prod",
    "cumsum",
    "mean",
    "max",
    "min",
    "abs",
    "sqrt",
    "exp",
    "log",
    "tanh",
    "relu",
    "sigmoid",
    "floor",
    "ceil",
    "round",
    "mod",
    "rem",
    "sign",
    "norm",
    "dot",
    "kron",
    "sort",
    "find",
    "maxk",
    "mink",
    "any",
    "all",
    "isempty",
    "ind2sub",
    "sub2ind",
    "linspace",
    "isequal",
    "logical",
    "double",
    "isnan",
    "randn",
    "rand",
    "pi",
    "Inf",
    "NaN",
    "tic",
    "toc",
    // strings, structs, cells
    "sprintf",
    "fprintf",
    "disp",
    "error",
    "assert",
    "startsWith",
    "strcat",
    "getenv",
    "isfile",
    "hf_download",
    "hf_model",
    "capture",
    "token_head",
    "with_head",
    "with_writers",
    "read_weights",
    "score",
    "repmat",
    "dec2bin",
    "num2str",
    "count",
    "struct",
    "cell",
    "isfield",
    "fieldnames",
    "last",
    // networks
    "wload",
    "wsave",
    "mlp",
    "transformer",
    "copy",
    "parts",
    "writers",
    "readers",
    "names",
    "layout",
    "channels",
    "tok",
    "keep_top",
    "remove",
    "delta",
    "wdiff",
    "wcanon",
    "walign",
    "baseline",
    "compile",
    // observation
    "budget",
    "dataset",
    "protect",
    "activations",
    "run",
    "gen",
    "tf",
    "attn",
    "rollout",
    "pair",
    "cause",
    "confirm",
    "grad_solve",
];

pub fn exists(name: &str) -> bool {
    NAMES.contains(&name)
}

pub(crate) fn arr(interp: &Interp, v: &Value, span: Span) -> R<Tensor> {
    interp.to_arr(v.clone(), span)
}

pub(crate) fn num(interp: &Interp, v: &Value, span: Span, what: &str) -> R<f64> {
    let t = arr(interp, v, span)?;
    t.scalar_value().ok_or_else(|| {
        err(
            "E-TYPE",
            format!("{what} must be a scalar, got {}", t.shape_str()),
            span,
        )
    })
}

pub(crate) fn usize_arg(interp: &Interp, v: &Value, span: Span, what: &str) -> R<usize> {
    let x = num(interp, v, span, what)?;
    if x < 0.0 || x != x.trunc() {
        return Err(err(
            "E-TYPE",
            format!("{what} must be a non-negative integer"),
            span,
        ));
    }
    Ok(x as usize)
}

pub(crate) fn string(v: &Value, span: Span, what: &str) -> R<String> {
    match v {
        Value::Str(s) => Ok(s.to_string()),
        other => Err(err(
            "E-TYPE",
            format!("{what} must be a string, got a {}", other.kind()),
            span,
        )),
    }
}

pub(crate) fn net_of(v: &Value, span: Span) -> R<NetRef> {
    match v {
        Value::Net(n) => Ok(n.clone()),
        other => Err(err(
            "E-TYPE",
            format!("expected a net, got a {}", other.kind()),
            span,
        )),
    }
}

/// Name-value options after the first `skip` positional arguments.
pub(crate) fn opts(args: &[Value], skip: usize, span: Span) -> R<IndexMap<String, Value>> {
    let mut m = IndexMap::new();
    let rest = &args[skip.min(args.len())..];
    if !rest.len().is_multiple_of(2) {
        return Err(err(
            "E-ARGS",
            "options must come in 'name', value pairs",
            span,
        ));
    }
    for pair in rest.chunks(2) {
        let k = string(&pair[0], span, "an option name")?;
        m.insert(k, pair[1].clone());
    }
    Ok(m)
}

fn one(v: Value) -> R<Vec<Value>> {
    Ok(vec![v])
}

fn t1(t: Tensor) -> R<Vec<Value>> {
    Ok(vec![Value::Arr(t)])
}

pub(crate) fn cell_strings(v: &Value, span: Span, what: &str) -> R<Vec<String>> {
    match v {
        Value::Cell(c) => c.iter().map(|x| string(x, span, what)).collect(),
        Value::Str(s) => Ok(vec![s.to_string()]),
        other => Err(err(
            "E-TYPE",
            format!("{what} must be a cell of strings, got a {}", other.kind()),
            span,
        )),
    }
}

/// Reduce along a 0-based dimension, MATLAB-style default (vectors reduce fully, matrices along dim 0).
fn reduce(
    t: &Tensor,
    dim: Option<usize>,
    init: f64,
    f: impl Fn(f64, f64) -> f64,
    post: impl Fn(f64, usize) -> f64,
) -> Tensor {
    let tag = Tag::derived(t.tag);
    let (r, c) = (t.rows(), t.cols());
    let dim = match dim {
        Some(d) => d,
        None => {
            if t.is_vector() || t.is_scalar() {
                let mut acc = init;
                for &x in &t.data {
                    acc = f(acc, x);
                }
                return Tensor::scalar(post(acc, t.numel())).with_tag(tag);
            }
            0
        }
    };
    if dim == 0 {
        let mut out = vec![init; c];
        for i in 0..r {
            for (j, value) in out.iter_mut().enumerate() {
                *value = f(*value, t.data[i * c + j]);
            }
        }
        Tensor::new(
            vec![1, c],
            out.into_iter().map(|a| post(a, r)).collect(),
            tag,
        )
    } else {
        let mut out = vec![init; r];
        for (i, value) in out.iter_mut().enumerate() {
            for j in 0..c {
                *value = f(*value, t.data[i * c + j]);
            }
        }
        Tensor::new(
            vec![r, 1],
            out.into_iter().map(|a| post(a, c)).collect(),
            tag,
        )
    }
}

/// MATLAB dimension convention: dim 1 reduces rows (result is one row),
/// dim 2 reduces columns (result is one column).
fn dim_arg(it: &Interp, v: &Value, span: Span) -> R<usize> {
    match usize_arg(it, v, span, "the dimension")? {
        1 => Ok(0),
        2 => Ok(1),
        d => Err(err(
            "E-ARGS",
            format!("dimension must be 1 or 2, got {d}"),
            span,
        )),
    }
}

/// max/min with positions along a dimension.
fn extreme(t: &Tensor, dim: Option<usize>, want_max: bool) -> (Tensor, Tensor) {
    let better = |a: f64, b: f64| if want_max { a > b } else { a < b };
    let tag = Tag::derived(t.tag);
    let (r, c) = (t.rows(), t.cols());
    let full = dim.is_none() && (t.is_vector() || t.is_scalar());
    if full {
        let mut bi = 0;
        for (i, &x) in t.data.iter().enumerate() {
            if better(x, t.data[bi]) || t.data[bi].is_nan() {
                bi = i;
            }
        }
        return (
            Tensor::scalar(t.data[bi]).with_tag(tag),
            Tensor::scalar(bi as f64).with_tag(tag),
        );
    }
    let dim = dim.unwrap_or(0);
    if dim == 0 {
        let mut v = vec![0.0; c];
        let mut p = vec![0.0; c];
        for j in 0..c {
            let mut bi = 0;
            for i in 0..r {
                if better(t.data[i * c + j], t.data[bi * c + j]) {
                    bi = i;
                }
            }
            v[j] = t.data[bi * c + j];
            p[j] = bi as f64;
        }
        (
            Tensor::new(vec![1, c], v, tag),
            Tensor::new(vec![1, c], p, tag),
        )
    } else {
        let mut v = vec![0.0; r];
        let mut p = vec![0.0; r];
        for i in 0..r {
            let mut bj = 0;
            for j in 0..c {
                if better(t.data[i * c + j], t.data[i * c + bj]) {
                    bj = j;
                }
            }
            v[i] = t.data[i * c + bj];
            p[i] = bj as f64;
        }
        (
            Tensor::new(vec![r, 1], v, tag),
            Tensor::new(vec![r, 1], p, tag),
        )
    }
}

fn elementwise(interp: &Interp, args: &[Value], span: Span, f: fn(f64) -> f64) -> R<Vec<Value>> {
    if args.len() != 1 {
        return Err(err("E-ARGS", "takes one argument", span));
    }
    t1(arr(interp, &args[0], span)?.map(f))
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545F4914F6CDD1D);
        (v >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.next().max(1e-300);
        let u2 = self.next();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn dims_from(interp: &Interp, args: &[Value], span: Span) -> R<(Vec<usize>, usize)> {
    // leading numeric args are dimensions; returns (shape, number consumed)
    let mut dims = Vec::new();
    let mut k = 0;
    while k < args.len() {
        match &args[k] {
            Value::Arr(t) if t.is_scalar() => {
                dims.push(usize_arg(interp, &args[k], span, "a size")?);
                k += 1;
            }
            Value::Arr(t) if k == 0 && t.is_vector() => {
                for &x in &t.data {
                    dims.push(x as usize);
                }
                k += 1;
                break;
            }
            _ => break,
        }
    }
    if dims.len() == 1 {
        dims.push(dims[0]);
    }
    if dims.is_empty() {
        dims = vec![1, 1];
    }
    Ok((dims, k))
}

pub fn format(interp: &Interp, fmt: &str, args: &[Value], span: Span) -> R<String> {
    // flatten arguments into a queue of numbers / strings
    let mut queue: Vec<Value> = Vec::new();
    for a in args {
        match a {
            Value::Arr(t) => {
                for &x in &t.data {
                    queue.push(Value::num(x));
                }
            }
            Value::Part(..) => {
                for &x in &arr(interp, a, span)?.data {
                    queue.push(Value::num(x));
                }
            }
            other => queue.push(other.clone()),
        }
    }
    let mut out = String::new();
    let mut qi = 0;
    let chars: Vec<char> = fmt.chars().collect();
    loop {
        let mut i = 0;
        let mut used = false;
        while i < chars.len() {
            let c = chars[i];
            if c == '\\' && i + 1 < chars.len() {
                match chars[i + 1] {
                    'n' => out.push('\n'),
                    't' => out.push('\t'),
                    '\\' => out.push('\\'),
                    other => {
                        out.push('\\');
                        out.push(other);
                    }
                }
                i += 2;
                continue;
            }
            if c == '%' && i + 1 < chars.len() {
                if chars[i + 1] == '%' {
                    out.push('%');
                    i += 2;
                    continue;
                }
                let mut j = i + 1;
                while j < chars.len() && "+-0# .123456789".contains(chars[j]) {
                    j += 1;
                }
                if j >= chars.len() {
                    out.push(c);
                    i += 1;
                    continue;
                }
                let spec: String = chars[i + 1..j].iter().collect();
                let conv = chars[j];
                let arg = queue.get(qi).cloned();
                qi += 1;
                used = true;
                let (width, prec) = {
                    let s = spec.trim_start_matches(['+', '-', '0', '#', ' ']);
                    let mut parts = s.split('.');
                    let w: usize = parts.next().unwrap_or("").parse().unwrap_or(0);
                    let p: Option<usize> = parts.next().map(|p| p.parse().unwrap_or(0));
                    (w, p)
                };
                let left = spec.starts_with('-');
                let s = match (conv, &arg) {
                    (_, None) => String::new(),
                    ('s', Some(Value::Str(s))) => s.to_string(),
                    ('s', Some(Value::Arr(t))) => fmt_num(t.data[0]),
                    ('d' | 'i', Some(Value::Arr(t))) => {
                        let v = t.data[0];
                        if v == v.trunc() {
                            format!("{}", v as i64)
                        } else {
                            format!("{v}")
                        }
                    }
                    ('f', Some(Value::Arr(t))) => format!("{:.*}", prec.unwrap_or(6), t.data[0]),
                    ('e', Some(Value::Arr(t))) => format!("{:.*e}", prec.unwrap_or(6), t.data[0]),
                    ('g', Some(Value::Arr(t))) => match prec {
                        Some(p) => {
                            let v = t.data[0];
                            let s = format!("{:.*}", p, v);
                            s.trim_end_matches('0').trim_end_matches('.').to_string()
                        }
                        None => fmt_num(t.data[0]),
                    },
                    (_, Some(Value::Str(s))) => s.to_string(),
                    (_, Some(other)) => format!("<{}>", other.kind()),
                };
                let s = if s.len() < width {
                    if left {
                        format!("{s:<width$}")
                    } else {
                        format!("{s:>width$}")
                    }
                } else {
                    s
                };
                out.push_str(&s);
                i = j + 1;
                continue;
            }
            out.push(c);
            i += 1;
        }
        if !used || qi >= queue.len() {
            break;
        }
    }
    Ok(out)
}

pub fn display(interp: &Interp, v: &Value) -> String {
    match v {
        Value::Arr(t) => {
            if t.is_scalar() {
                format!("{}\n", t)
            } else if t.is_empty() {
                "[]\n".into()
            } else {
                format!("{}", t)
            }
        }
        Value::Str(s) => format!("{s}\n"),
        Value::Cell(c) => {
            let items: Vec<String> = c.iter().map(|x| short(interp, x)).collect();
            format!("{{{}}}\n", items.join(", "))
        }
        Value::Struct(s) => {
            let mut o = String::new();
            for (k, v) in s.iter() {
                o += &format!("  {k}: {}\n", short(interp, v));
            }
            o
        }
        Value::Net(n) => format!("{}\n", n.borrow().summary()),
        Value::Part(n, p) => {
            let nb = n.borrow();
            let pd = &nb.parts[p.as_ref()];
            format!(
                "part `{p}` {} (file `{}`)\n",
                crate::tensor::shape_str(&pd.shape),
                pd.file_name
            )
        }
        Value::Path(_, p) => format!("net path `{p}`\n"),
        Value::Group(g) => format!(
            "group of {} parts: {}\n",
            g.len(),
            compress(&g.iter().map(|x| x.1.clone()).collect::<Vec<_>>())
        ),
        Value::Func(_) => "function\n".into(),
        Value::Nothing => "\n".into(),
    }
}

fn short(interp: &Interp, v: &Value) -> String {
    match v {
        Value::Str(s) => format!("'{s}'"),
        Value::Arr(t) if t.is_scalar() => fmt_num(t.data[0]),
        Value::Arr(t) => format!("[{}]", t.shape_str()),
        other => display(interp, other).trim_end().to_string(),
    }
}

/// `L[0].attn.o, L[1].attn.o, ...` -> `L[0..23].attn.o`
pub fn compress(names: &[String]) -> String {
    let mut groups: IndexMap<String, Vec<usize>> = IndexMap::new();
    for n in names {
        if let (Some(a), Some(b)) = (n.find('['), n.find(']')) {
            if let Ok(i) = n[a + 1..b].parse::<usize>() {
                let key = format!("{}[*]{}", &n[..a], &n[b + 1..]);
                groups.entry(key).or_default().push(i);
                continue;
            }
        }
        groups.entry(n.clone()).or_default();
    }
    let mut out = Vec::new();
    for (k, idx) in groups {
        if idx.is_empty() {
            out.push(k);
            continue;
        }
        let (lo, hi) = (*idx.iter().min().unwrap(), *idx.iter().max().unwrap());
        let r = if idx.len() == hi - lo + 1 {
            if lo == hi {
                format!("{lo}")
            } else {
                format!("{lo}..{hi}")
            }
        } else {
            idx.iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        out.push(k.replace("[*]", &format!("[{r}]")));
    }
    out.join(", ")
}

fn glob(pat: &str, s: &str) -> bool {
    let p: Vec<char> = pat.chars().collect();
    let t: Vec<char> = s.chars().collect();
    fn m(p: &[char], t: &[char]) -> bool {
        match p.first() {
            None => t.is_empty(),
            Some('*') => (0..=t.len()).any(|k| m(&p[1..], &t[k..])),
            Some(c) => !t.is_empty() && t[0] == *c && m(&p[1..], &t[1..]),
        }
    }
    m(&p, &t)
}

/// Expand `[15,19]` lists into separate patterns.
fn expand_pattern(pat: &str) -> Vec<String> {
    if let (Some(a), Some(b)) = (pat.find('['), pat.find(']')) {
        let inner = &pat[a + 1..b];
        if inner.contains(',') {
            let mut out = Vec::new();
            for piece in inner.split(',') {
                let p = format!("{}[{}]{}", &pat[..a], piece.trim(), &pat[b + 1..]);
                out.extend(expand_pattern(&p));
            }
            return out;
        }
        // recurse on the rest
        let rest = expand_pattern(&pat[b + 1..]);
        return rest
            .into_iter()
            .map(|r| format!("{}{}", &pat[..b + 1], r))
            .collect();
    }
    vec![pat.to_string()]
}

/// Edit one weight: remove direction `u` on `axis` with gain `g`, optionally only top-k writers.
fn remove_dir(
    data: &mut [f64],
    shape: &[usize],
    u: &[f64],
    g: f64,
    axis: usize,
    top: Option<usize>,
) -> Result<(), String> {
    let (m, n) = if shape.len() == 1 {
        (shape[0], 1)
    } else {
        (shape[0], shape[1])
    };
    let norm: f64 = u.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 {
        return Err("the direction is all zeros".into());
    }
    let u: Vec<f64> = u.iter().map(|x| x / norm).collect();
    if axis == 0 {
        if u.len() != m {
            return Err(format!(
                "the direction has {} entries but the out side has {m}",
                u.len()
            ));
        }
        let mut c = vec![0.0; n];
        for i in 0..m {
            for j in 0..n {
                c[j] += u[i] * data[i * n + j];
            }
        }
        if let Some(k) = top {
            keep_top_abs(&mut c, k);
        }
        for i in 0..m {
            for j in 0..n {
                data[i * n + j] -= g * u[i] * c[j];
            }
        }
    } else {
        if u.len() != n {
            return Err(format!(
                "the direction has {} entries but the in side has {n}",
                u.len()
            ));
        }
        let mut c = vec![0.0; m];
        for i in 0..m {
            for j in 0..n {
                c[i] += data[i * n + j] * u[j];
            }
        }
        if let Some(k) = top {
            keep_top_abs(&mut c, k);
        }
        for i in 0..m {
            for j in 0..n {
                data[i * n + j] -= g * c[i] * u[j];
            }
        }
    }
    Ok(())
}

fn keep_top_abs(v: &mut [f64], k: usize) {
    if k >= v.len() {
        return;
    }
    let mut order: Vec<usize> = (0..v.len()).collect();
    order.sort_by(|&a, &b| v[b].abs().partial_cmp(&v[a].abs()).unwrap().then(a.cmp(&b)));
    for &i in &order[k..] {
        v[i] = 0.0;
    }
}

fn layer_of(name: &str) -> Option<usize> {
    let a = name.find('[')?;
    let b = name.find(']')?;
    name[a + 1..b].parse().ok()
}

pub fn call(
    it: &mut Interp,
    name: &str,
    args: Vec<Value>,
    nargout: usize,
    span: Span,
) -> R<Vec<Value>> {
    let a = &args;
    if matches!(
        name,
        "names" | "parts" | "writers" | "readers" | "delta" | "remove"
    ) && a.iter().any(|v| crate::decoder_api::is_decoder(Some(v)))
    {
        return Err(err(
            "E-UNSUPPORTED",
            "use token_head and with_head to inspect and edit native output rows",
            span,
        ));
    }
    let need = |n: usize| -> R<()> {
        if a.len() < n {
            Err(err(
                "E-ARGS",
                format!("`{name}` needs at least {n} argument(s)"),
                span,
            ))
        } else {
            Ok(())
        }
    };
    match name {
        // ------------------------------------------------ constants & time
        "pi" => t1(Tensor::scalar(std::f64::consts::PI)),
        "Inf" => t1(Tensor::scalar(f64::INFINITY)),
        "NaN" => t1(Tensor::scalar(f64::NAN)),
        "tic" => {
            it.timer = Some(std::time::Instant::now());
            Ok(vec![])
        }
        "toc" => {
            let s = it.timer.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0);
            t1(Tensor::scalar(s))
        }
        // ------------------------------------------------ shapes
        "size" => {
            need(1)?;
            let shape: Vec<usize> = match &a[0] {
                Value::Arr(t) => t.shape.clone(),
                Value::Part(..) => arr(it, &a[0], span)?.shape,
                Value::Str(s) => vec![1, s.chars().count()],
                Value::Cell(c) => vec![1, c.len()],
                other => return Err(err("E-TYPE", format!("size of a {}", other.kind()), span)),
            };
            if a.len() > 1 {
                let d = usize_arg(it, &a[1], span, "the dimension")?;
                return t1(Tensor::scalar(*shape.get(d).unwrap_or(&1) as f64));
            }
            if nargout >= 2 {
                return Ok(shape
                    .iter()
                    .take(nargout)
                    .map(|&d| Value::num(d as f64))
                    .collect());
            }
            t1(Tensor::row(shape.iter().map(|&d| d as f64).collect()))
        }
        "numel" | "length" => {
            need(1)?;
            let n = match &a[0] {
                Value::Arr(t) => {
                    if name == "length" {
                        *t.shape.iter().max().unwrap_or(&0)
                    } else {
                        t.numel()
                    }
                }
                Value::Part(..) => arr(it, &a[0], span)?.numel(),
                Value::Str(s) => s.chars().count(),
                Value::Cell(c) => c.len(),
                Value::Group(g) => g.len(),
                Value::Net(n) => n.borrow().total_weights(),
                Value::Path(n, p) => n.borrow().sub_indices(p).len(),
                other => return Err(err("E-TYPE", format!("numel of a {}", other.kind()), span)),
            };
            t1(Tensor::scalar(n as f64))
        }
        "rows" => {
            need(1)?;
            t1(Tensor::scalar(arr(it, &a[0], span)?.rows() as f64))
        }
        "cols" => {
            need(1)?;
            t1(Tensor::scalar(arr(it, &a[0], span)?.cols() as f64))
        }
        "last" => {
            need(1)?;
            let n = match &a[0] {
                Value::Arr(t) => t.numel(),
                Value::Part(..) => arr(it, &a[0], span)?.numel(),
                Value::Str(s) => s.chars().count(),
                Value::Cell(c) => c.len(),
                Value::Group(g) => g.len(),
                Value::Path(n, p) => n.borrow().sub_indices(p).last().map(|x| x + 1).unwrap_or(0),
                other => return Err(err("E-TYPE", format!("last of a {}", other.kind()), span)),
            };
            t1(Tensor::scalar(n as f64 - 1.0))
        }
        "zeros" | "ones" => {
            let (dims, _) = dims_from(it, a, span)?;
            let v = if name == "ones" { 1.0 } else { 0.0 };
            let n = dims.iter().product();
            t1(Tensor::new(dims, vec![v; n], Tag::Lit))
        }
        "eye" => {
            let (dims, _) = dims_from(it, a, span)?;
            let mut t = Tensor::zeros(dims.clone());
            for i in 0..dims[0].min(dims[1]) {
                t.data[i * dims[1] + i] = 1.0;
            }
            t1(t)
        }
        "reshape" => {
            need(2)?;
            let t = arr(it, &a[0], span)?;
            let (dims, _) = dims_from(it, &a[1..], span)?;
            t1(t.reshape(dims).map_err(|m| err("E-SHAPE", m, span))?)
        }
        "randn" | "rand" => {
            let (dims, k) = dims_from(it, a, span)?;
            let o = opts(a, k, span)?;
            let seed = match o.get("seed") {
                Some(v) => num(it, v, span, "the seed")? as u64,
                None => 0x9E3779B97F4A7C15,
            };
            let mut rng = Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1);
            let n: usize = dims.iter().product();
            let d = (0..n)
                .map(|_| {
                    if name == "randn" {
                        rng.normal()
                    } else {
                        rng.next()
                    }
                })
                .collect();
            t1(Tensor::new(dims, d, Tag::Lit))
        }
        "linspace" => {
            need(3)?;
            let (lo, hi, n) = (
                num(it, &a[0], span, "start")?,
                num(it, &a[1], span, "end")?,
                usize_arg(it, &a[2], span, "count")?,
            );
            let d = (0..n)
                .map(|k| {
                    if n == 1 {
                        hi
                    } else {
                        lo + (hi - lo) * k as f64 / (n - 1) as f64
                    }
                })
                .collect();
            t1(Tensor::row(d))
        }
        // ------------------------------------------------ reductions
        "sum" | "prod" | "mean" | "any" | "all" => {
            need(1)?;
            let t = arr(it, &a[0], span)?;
            let dim = if a.len() > 1 {
                Some(dim_arg(it, &a[1], span)?)
            } else {
                None
            };
            let mut r = match name {
                "sum" => reduce(&t, dim, 0.0, |x, y| x + y, |x, _| x),
                "prod" => reduce(&t, dim, 1.0, |x, y| x * y, |x, _| x),
                "mean" => reduce(&t, dim, 0.0, |x, y| x + y, |x, n| x / n as f64),
                "any" => reduce(
                    &t,
                    dim,
                    0.0,
                    |x, y| if x != 0.0 || y != 0.0 { 1.0 } else { 0.0 },
                    |x, _| x,
                ),
                _ => reduce(
                    &t,
                    dim,
                    1.0,
                    |x, y| if x != 0.0 && y != 0.0 { 1.0 } else { 0.0 },
                    |x, _| x,
                ),
            };
            if name == "any" || name == "all" {
                r.logical = true;
            }
            t1(r)
        }
        "cumsum" => {
            need(1)?;
            let t = arr(it, &a[0], span)?;
            let mut acc = 0.0;
            let d = t.data.iter().map(|x| {
                acc += x;
                acc
            });
            t1(Tensor::new(
                t.shape.clone(),
                d.collect(),
                Tag::derived(t.tag),
            ))
        }
        "max" | "min" => {
            need(1)?;
            let t = arr(it, &a[0], span)?;
            if a.len() >= 2 && !matches!(&a[1], Value::Arr(x) if x.is_empty()) {
                let u = arr(it, &a[1], span)?;
                let f = if name == "max" { f64::max } else { f64::min };
                return t1(t.zip(&u, f).map_err(|m| err("E-SHAPE", m, span))?);
            }
            let dim = if a.len() >= 3 {
                Some(dim_arg(it, &a[2], span)?)
            } else {
                None
            };
            let (v, p) = extreme(&t, dim, name == "max");
            Ok(vec![Value::Arr(v), Value::Arr(p)])
        }
        "maxk" | "mink" => {
            need(2)?;
            let t = arr(it, &a[0], span)?;
            let k = usize_arg(it, &a[1], span, "k")?.min(t.numel());
            let mut order: Vec<usize> = (0..t.numel()).collect();
            if name == "maxk" {
                order.sort_by(|&x, &y| {
                    t.data[y]
                        .partial_cmp(&t.data[x])
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(x.cmp(&y))
                });
            } else {
                order.sort_by(|&x, &y| {
                    t.data[x]
                        .partial_cmp(&t.data[y])
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(x.cmp(&y))
                });
            }
            order.truncate(k);
            let tag = Tag::derived(t.tag);
            let v = Tensor::row(order.iter().map(|&i| t.data[i]).collect()).with_tag(tag);
            let p = Tensor::row(order.iter().map(|&i| i as f64).collect()).with_tag(tag);
            Ok(vec![Value::Arr(v), Value::Arr(p)])
        }
        "sort" => {
            need(1)?;
            let t = arr(it, &a[0], span)?;
            let desc = a
                .get(1)
                .and_then(|v| v.as_str())
                .map(|s| s == "descend")
                .unwrap_or(false);
            let mut order: Vec<usize> = (0..t.numel()).collect();
            order.sort_by(|&x, &y| {
                let o = t.data[x]
                    .partial_cmp(&t.data[y])
                    .unwrap_or(std::cmp::Ordering::Equal);
                (if desc { o.reverse() } else { o }).then(x.cmp(&y))
            });
            let tag = Tag::derived(t.tag);
            let v = Tensor::new(
                t.shape.clone(),
                order.iter().map(|&i| t.data[i]).collect(),
                tag,
            );
            let p = Tensor::new(
                t.shape.clone(),
                order.iter().map(|&i| i as f64).collect(),
                tag,
            );
            Ok(vec![Value::Arr(v), Value::Arr(p)])
        }
        "find" => {
            need(1)?;
            let t = arr(it, &a[0], span)?;
            let p: Vec<f64> = t
                .data
                .iter()
                .enumerate()
                .filter(|(_, &x)| x != 0.0)
                .map(|(i, _)| i as f64)
                .collect();
            t1(Tensor::row(p))
        }
        // ------------------------------------------------ elementwise
        "abs" => elementwise(it, a, span, f64::abs),
        "sqrt" => elementwise(it, a, span, f64::sqrt),
        "exp" => elementwise(it, a, span, f64::exp),
        "log" => elementwise(it, a, span, f64::ln),
        "tanh" => elementwise(it, a, span, f64::tanh),
        "relu" => elementwise(it, a, span, |x| x.max(0.0)),
        "sigmoid" => elementwise(it, a, span, |x| 1.0 / (1.0 + (-x).exp())),
        "floor" => elementwise(it, a, span, f64::floor),
        "ceil" => elementwise(it, a, span, f64::ceil),
        "round" => elementwise(it, a, span, f64::round),
        "sign" => elementwise(it, a, span, |x| {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }),
        "isnan" => {
            need(1)?;
            let mut t = arr(it, &a[0], span)?.map(|x| if x.is_nan() { 1.0 } else { 0.0 });
            t.logical = true;
            t1(t)
        }
        "logical" => {
            need(1)?;
            let mut t = arr(it, &a[0], span)?.map(|x| if x != 0.0 { 1.0 } else { 0.0 });
            t.logical = true;
            t1(t)
        }
        "double" => {
            need(1)?;
            let mut t = arr(it, &a[0], span)?;
            t.logical = false;
            t1(t)
        }
        "mod" | "rem" => {
            need(2)?;
            let (x, y) = (arr(it, &a[0], span)?, arr(it, &a[1], span)?);
            let f = if name == "mod" {
                |p: f64, q: f64| if q == 0.0 { p } else { p - (p / q).floor() * q }
            } else {
                |p: f64, q: f64| if q == 0.0 { f64::NAN } else { p % q }
            };
            t1(x.zip(&y, f).map_err(|m| err("E-SHAPE", m, span))?)
        }
        "norm" => {
            need(1)?;
            let t = arr(it, &a[0], span)?;
            let s: f64 = t.data.iter().map(|x| x * x).sum::<f64>().sqrt();
            t1(Tensor::scalar(s).with_tag(Tag::derived(t.tag)))
        }
        "dot" => {
            need(2)?;
            let (x, y) = (arr(it, &a[0], span)?, arr(it, &a[1], span)?);
            if x.numel() != y.numel() {
                return Err(err("E-SHAPE", "dot needs vectors of the same length", span));
            }
            let s: f64 = x.data.iter().zip(&y.data).map(|(p, q)| p * q).sum();
            t1(Tensor::scalar(s).with_tag(Tag::combine(x.tag, y.tag)))
        }
        "kron" => {
            need(2)?;
            let (x, y) = (arr(it, &a[0], span)?, arr(it, &a[1], span)?);
            let (r1, c1, r2, c2) = (x.rows(), x.cols(), y.rows(), y.cols());
            let mut d = vec![0.0; r1 * r2 * c1 * c2];
            let w = c1 * c2;
            for i in 0..r1 {
                for j in 0..c1 {
                    for k in 0..r2 {
                        for l in 0..c2 {
                            d[(i * r2 + k) * w + j * c2 + l] =
                                x.data[i * c1 + j] * y.data[k * c2 + l];
                        }
                    }
                }
            }
            t1(Tensor::new(vec![r1 * r2, w], d, Tag::combine(x.tag, y.tag)))
        }
        "isempty" => {
            need(1)?;
            let e = match &a[0] {
                Value::Arr(t) => t.is_empty(),
                Value::Str(s) => s.is_empty(),
                Value::Cell(c) => c.is_empty(),
                _ => false,
            };
            one(Value::boolean(e))
        }
        "isequal" => {
            need(2)?;
            let eq = match (&a[0], &a[1]) {
                (Value::Str(x), Value::Str(y)) => x == y,
                _ => {
                    let (x, y) = (arr(it, &a[0], span)?, arr(it, &a[1], span)?);
                    x.shape == y.shape && x.data == y.data
                }
            };
            one(Value::boolean(eq))
        }
        "ind2sub" => {
            need(2)?;
            let shape = arr(it, &a[0], span)?;
            let k = num(it, &a[1], span, "the linear index")? as usize;
            let dims: Vec<usize> = shape.data.iter().map(|&x| x as usize).collect();
            let st = crate::tensor::strides(&dims);
            let mut rem = k;
            let mut out = Vec::new();
            for s in st {
                out.push(Value::num((rem / s) as f64));
                rem %= s;
            }
            Ok(out)
        }
        "sub2ind" => {
            need(2)?;
            let shape = arr(it, &a[0], span)?;
            let dims: Vec<usize> = shape.data.iter().map(|&x| x as usize).collect();
            let st = crate::tensor::strides(&dims);
            let mut k = 0usize;
            for (d, s) in a[1..].iter().zip(st) {
                k += num(it, d, span, "a subscript")? as usize * s;
            }
            t1(Tensor::scalar(k as f64))
        }
        // ------------------------------------------------ strings and output
        "sprintf" => {
            need(1)?;
            let f = string(&a[0], span, "the format")?;
            one(Value::str(format(it, &f, &a[1..], span)?))
        }
        "fprintf" => {
            need(1)?;
            let f = string(&a[0], span, "the format")?;
            let s = format(it, &f, &a[1..], span)?;
            it.print(&s);
            Ok(vec![])
        }
        "disp" => {
            need(1)?;
            let s = display(it, &a[0]);
            it.print(&s);
            Ok(vec![])
        }
        "error" => {
            need(1)?;
            let f = string(&a[0], span, "the message")?;
            Err(err("E-USER", format(it, &f, &a[1..], span)?, span))
        }
        "assert" => {
            need(1)?;
            let t = arr(it, &a[0], span)?;
            if t.is_empty() || t.data.contains(&0.0) {
                let msg = match a.get(1) {
                    Some(Value::Str(s)) => format(it, s, &a[2..], span)?,
                    _ => "assertion failed".into(),
                };
                return Err(err("E-ASSERT", msg, span));
            }
            Ok(vec![])
        }
        "startsWith" => {
            need(2)?;
            one(Value::boolean(
                string(&a[0], span, "the text")?.starts_with(&string(&a[1], span, "the prefix")?),
            ))
        }
        "strcat" => {
            let mut s = String::new();
            for v in a {
                s += &string(v, span, "each argument")?;
            }
            one(Value::str(s))
        }
        "repmat" => {
            need(3)?;
            let (r, c) = (
                usize_arg(it, &a[1], span, "rows")?,
                usize_arg(it, &a[2], span, "columns")?,
            );
            match &a[0] {
                Value::Str(s) => {
                    if r != 1 {
                        return Err(err("E-ARGS", "repmat of a string needs 1 row", span));
                    }
                    one(Value::str(s.repeat(c)))
                }
                v => {
                    let t = arr(it, v, span)?;
                    let (tr, tc) = (t.rows(), t.cols());
                    let (h, w) = (tr * r, tc * c);
                    let mut d = vec![0.0; h * w];
                    for i in 0..h {
                        for j in 0..w {
                            d[i * w + j] = t.data[(i % tr) * tc + (j % tc)];
                        }
                    }
                    t1(Tensor::new(vec![h, w], d, t.tag))
                }
            }
        }
        "dec2bin" => {
            need(1)?;
            let n = usize_arg(it, &a[0], span, "the number")?;
            let w = if a.len() > 1 {
                usize_arg(it, &a[1], span, "the width")?
            } else {
                0
            };
            one(Value::str(format!("{n:0w$b}")))
        }
        "num2str" => {
            need(1)?;
            let t = arr(it, &a[0], span)?;
            one(Value::str(
                t.data
                    .iter()
                    .map(|&x| fmt_num(x))
                    .collect::<Vec<_>>()
                    .join("  "),
            ))
        }
        "count" => {
            need(2)?;
            let (s, p) = (
                string(&a[0], span, "the text")?,
                string(&a[1], span, "the pattern")?,
            );
            if p.is_empty() {
                return Err(err("E-ARGS", "the pattern is empty", span));
            }
            t1(Tensor::scalar(s.matches(&p).count() as f64))
        }
        "struct" => {
            let mut m = IndexMap::new();
            if !a.len().is_multiple_of(2) {
                return Err(err("E-ARGS", "struct takes 'name', value pairs", span));
            }
            for p in a.chunks(2) {
                m.insert(string(&p[0], span, "a field name")?, p[1].clone());
            }
            one(Value::strukt(m))
        }
        "cell" => {
            let n = if a.is_empty() {
                0
            } else {
                usize_arg(it, &a[0], span, "the size")?
            };
            one(Value::cell(vec![Value::Arr(Tensor::empty()); n]))
        }
        "isfield" => {
            need(2)?;
            let f = string(&a[1], span, "the field")?;
            one(Value::boolean(
                matches!(&a[0], Value::Struct(s) if s.contains_key(&f)),
            ))
        }
        "fieldnames" => {
            need(1)?;
            match &a[0] {
                Value::Struct(s) => one(Value::cell(s.keys().map(Value::str).collect())),
                _ => Err(err("E-TYPE", "fieldnames needs a struct", span)),
            }
        }
        "getenv" => {
            if !(1..=2).contains(&a.len()) {
                return Err(err(
                    "E-ARGS",
                    "getenv needs a name and optional default string",
                    span,
                ));
            }
            let name = string(&a[0], span, "environment variable name")?;
            let fallback = if a.len() == 2 {
                string(&a[1], span, "environment default")?
            } else {
                String::new()
            };
            let value =
                crate::hf::environment(&name, &fallback).map_err(|m| err("E-ENV", m, span))?;
            one(Value::str(value))
        }
        "isfile" => {
            if a.len() != 1 {
                return Err(err("E-ARGS", "isfile needs exactly one path string", span));
            }
            let path = string(&a[0], span, "the path")?;
            one(Value::boolean(std::path::Path::new(&path).is_file()))
        }
        "hf_model" | "capture" | "token_head" | "with_head" | "with_writers" | "read_weights"
        | "score" => crate::decoder_api::call(it, name, a, span),
        "gen" if crate::decoder_api::is_decoder(a.first()) => {
            crate::decoder_api::call(it, name, a, span)
        }
        "run" if crate::decoder_api::is_decoder(a.first()) => {
            crate::decoder_api::call(it, name, a, span)
        }
        "hf_download" => {
            need(2)?;
            let repo = string(&a[0], span, "HF repository")?;
            let file = string(&a[1], span, "HF filename")?;
            let options = opts(a, 2, span)?;
            if options.len() != (a.len() - 2) / 2 {
                return Err(err("E-ARGS", "duplicate hf_download option", span));
            }
            let mut download = crate::hf::DownloadOptions::default();
            for (name, value) in options {
                match name.as_str() {
                    "revision" => download.revision = string(&value, span, "HF revision")?,
                    "sha256" => download.sha256 = Some(string(&value, span, "expected SHA-256")?),
                    "offline" => {
                        let value = num(it, &value, span, "offline")?;
                        if value != 0.0 && value != 1.0 {
                            return Err(err("E-ARGS", "offline must be boolean", span));
                        }
                        download.offline = value == 1.0;
                    }
                    "max_bytes" => {
                        let value = num(it, &value, span, "max_bytes")?;
                        if !value.is_finite()
                            || !(1.0..=9007199254740991.0).contains(&value)
                            || value != value.trunc()
                        {
                            return Err(err(
                                "E-ARGS",
                                "max_bytes must be a positive exact integer",
                                span,
                            ));
                        }
                        download.max_bytes = value as u64;
                    }
                    _ => return Err(err("E-ARGS", "unsupported hf_download option", span)),
                }
            }
            let path =
                crate::hf::download(&repo, &file, &download).map_err(|m| err("E-HF", m, span))?;
            let path = path
                .to_str()
                .ok_or_else(|| err("E-HF", "HF cache path is not UTF-8", span))?;
            one(Value::str(path))
        }
        // ------------------------------------------------ networks
        "wload" => {
            need(1)?;
            let path = string(&a[0], span, "the file")?;
            let o = opts(a, 1, span)?;
            let in_out = match o.get("orientation").and_then(|v| v.as_str()) {
                Some("in_out") => true,
                Some("out_in") | None => false,
                Some(other) => {
                    return Err(err(
                        "E-ARGS",
                        format!("orientation must be 'in_out' or 'out_in', got '{other}'"),
                        span,
                    ))
                }
            };
            let act = o
                .get("act")
                .and_then(|v| v.as_str())
                .unwrap_or("tanh")
                .to_string();
            let options = io::LoadOpts { in_out, act };
            let net = if let Some(parts) = o.get("parts") {
                let parts = cell_strings(parts, span, "'parts'")?;
                io::load_selected(&path, &options, &parts)
            } else {
                io::load(&path, &options)
            }
            .map_err(|m| err("E-IO", m, span))?;
            if net.selection.is_some() {
                it.print("wload: selected tensor bundle only; wsave/apply will not produce a complete checkpoint\n");
            }
            let conv = net.parts.values().any(|p| p.in_out);
            let renamed: Vec<String> = net
                .parts
                .iter()
                .filter(|(k, p)| **k != p.file_name)
                .map(|(k, p)| format!("{k} = {}", p.file_name))
                .collect();
            let msg = format!(
                "wload: {path}: {} parts, {} weights; matrices shown as [out, in]{}{}\n",
                net.parts.len(),
                net.total_weights(),
                if conv {
                    " (converted from the file's [in, out])"
                } else {
                    ""
                },
                if renamed.is_empty() {
                    String::new()
                } else {
                    format!("\n       names: {}", renamed.join(", "))
                }
            );
            it.print(&msg);
            one(Value::Net(net.into_ref()))
        }
        "wsave" => {
            need(2)?;
            let n = net_of(&a[0], span)?;
            let path = string(&a[1], span, "the file")?;
            let o = opts(a, 2, span)?;
            let obstacle = match o.get("obstacle") {
                Some(Value::Str(s)) => Some(s.to_string()),
                Some(Value::Struct(s)) => s
                    .get("name")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
                Some(_) => {
                    return Err(err(
                        "E-ARGS",
                        "'obstacle' must be a string or a claim from cause()",
                        span,
                    ))
                }
                None => None,
            };
            let preserve = o
                .get("preserve")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let budget = match o.get("budget") {
                Some(v) => Some(num(it, v, span, "the edit budget")?),
                None => None,
            };
            let protect = it.harness.protect.clone();
            let rep = io::save(
                &n.borrow(),
                &path,
                &io::SaveOpts {
                    obstacle,
                    preserve,
                    budget,
                    protect,
                },
            )
            .map_err(|(code, m)| err(static_code(&code), m, span))?;
            it.print(&format!(
                "wsave: {path}; patch {}: {} part(s), {} entries ({:.4}% of weights)\n",
                rep.patch_path,
                rep.changed_parts,
                rep.changed_entries,
                rep.share * 100.0
            ));
            Ok(vec![])
        }
        "mlp" => {
            need(1)?;
            let sizes: Vec<usize> = arr(it, &a[0], span)?
                .data
                .iter()
                .map(|&x| x as usize)
                .collect();
            if sizes.len() < 2 {
                return Err(err("E-ARGS", "mlp needs at least two sizes", span));
            }
            let act = a.get(1).and_then(|v| v.as_str()).unwrap_or("tanh");
            one(Value::Net(net::mlp(&sizes, act).into_ref()))
        }
        "transformer" => {
            let o = opts(a, 0, span)?;
            let g = |k: &str| -> R<usize> {
                match o.get(k) {
                    Some(v) => usize_arg(it, v, span, k),
                    None => Err(err("E-ARGS", format!("transformer needs '{k}'"), span)),
                }
            };
            let vocab = cell_strings(
                o.get("vocab")
                    .ok_or_else(|| err("E-ARGS", "transformer needs 'vocab'", span))?,
                span,
                "vocab",
            )?;
            let d = g("d")?;
            let heads = g("heads")?;
            let head_dim = match o.get("head_dim") {
                Some(v) => usize_arg(it, v, span, "head_dim")?,
                None => {
                    if d % heads != 0 {
                        return Err(err(
                            "E-ARGS",
                            "d must be divisible by heads, or give 'head_dim'",
                            span,
                        ));
                    }
                    d / heads
                }
            };
            let cfg = TCfg {
                vocab,
                ctx: g("ctx")?,
                layers: g("layers")?,
                heads,
                d,
                ff: g("ff")?,
                head_dim,
                act: o
                    .get("act")
                    .and_then(|v| v.as_str())
                    .unwrap_or("relu")
                    .to_string(),
            };
            let pos = matches!(o.get("pos"), Some(Value::Arr(t)) if t.data.first() == Some(&1.0));
            one(Value::Net(net::transformer(cfg, pos).into_ref()))
        }
        "copy" => {
            need(1)?;
            let n = net_of(&a[0], span)?;
            let c = n.borrow().clone();
            one(Value::Net(c.into_ref()))
        }
        "names" => {
            need(1)?;
            let n = net_of(&a[0], span)?;
            let v = n.borrow().parts.keys().map(Value::str).collect();
            one(Value::cell(v))
        }
        "parts" => {
            need(2)?;
            let n = net_of(&a[0], span)?;
            let pats = cell_strings(&a[1], span, "the pattern")?;
            let mut out = Vec::new();
            for p in pats {
                let exp = expand_pattern(&p);
                let nb = n.borrow();
                let mut found = false;
                for k in nb.parts.keys() {
                    if exp.iter().any(|e| glob(e, k)) {
                        out.push((n.clone(), k.clone()));
                        found = true;
                    }
                }
                if !found {
                    return Err(err("E-UNDEFINED", format!("no part matches `{p}`"), span));
                }
            }
            one(Value::Group(Rc::new(out)))
        }
        "writers" | "readers" => {
            need(1)?;
            let n = net_of(&a[0], span)?;
            let v = if name == "writers" {
                n.borrow().writers()
            } else {
                n.borrow().readers()
            };
            if v.is_empty() {
                return Err(err(
                    "E-TYPE",
                    format!("`{name}` needs a transformer with a residual stream"),
                    span,
                ));
            }
            one(Value::Group(Rc::new(
                v.into_iter().map(|p| (n.clone(), p)).collect(),
            )))
        }
        "layout" => {
            let mut dims = Vec::new();
            let mut names = None;
            for v in a {
                match v {
                    Value::Cell(_) => names = Some(cell_strings(v, span, "channel names")?),
                    _ => dims.push(usize_arg(it, v, span, "a layout size")?),
                }
            }
            let names =
                names.ok_or_else(|| err("E-ARGS", "layout needs a cell of channel names", span))?;
            let k = names.len();
            let (r, c) = match dims.as_slice() {
                [n] => (1, *n),
                [r, c] => (*r, *c),
                _ => return Err(err("E-ARGS", "layout takes one or two sizes", span)),
            };
            let mut m = IndexMap::new();
            for (ch, nm) in names.iter().enumerate() {
                let d: Vec<f64> = (0..r * c).map(|cell| (cell * k + ch) as f64).collect();
                m.insert(nm.clone(), Value::Arr(Tensor::new(vec![r, c], d, Tag::Lit)));
            }
            m.insert("size".into(), Value::num((r * c * k) as f64));
            one(Value::strukt(m))
        }
        "channels" => {
            need(1)?;
            let names = cell_strings(&a[0], span, "channel names")?;
            let mut m = IndexMap::new();
            for (i, n) in names.iter().enumerate() {
                m.insert(n.clone(), Value::num(i as f64));
            }
            one(Value::strukt(m))
        }
        "tok" => {
            need(2)?;
            let n = net_of(&a[0], span)?;
            let s = string(&a[1], span, "the token")?;
            let nb = n.borrow();
            match &nb.arch {
                Arch::Transformer(c) => match c.vocab.iter().position(|v| *v == s) {
                    Some(i) => t1(Tensor::scalar(i as f64)),
                    None => Err(err(
                        "E-UNDEFINED",
                        format!("token '{s}' is not in the vocabulary {:?}", c.vocab),
                        span,
                    )),
                },
                _ => Err(err("E-TYPE", "tok needs a transformer", span)),
            }
        }
        "keep_top" => {
            need(2)?;
            let mut t = arr(it, &a[0], span)?;
            let k = usize_arg(it, &a[1], span, "k")?;
            let by_abs = a.get(2).and_then(|v| v.as_str()) == Some("abs");
            if by_abs {
                keep_top_abs(&mut t.data, k);
            } else {
                let mut order: Vec<usize> = (0..t.numel()).collect();
                order.sort_by(|&x, &y| t.data[y].partial_cmp(&t.data[x]).unwrap().then(x.cmp(&y)));
                for &i in &order[k.min(order.len())..] {
                    t.data[i] = 0.0;
                }
            }
            t.tag = Tag::derived(t.tag);
            t1(t)
        }
        "remove" => remove(it, a, span),
        "delta" => {
            need(2)?;
            let (x, y) = (net_of(&a[0], span)?, net_of(&a[1], span)?);
            let o = opts(a, 2, span)?;
            let except = match o.get("except") {
                Some(v) => cell_strings(v, span, "'except'")?,
                None => vec![],
            };
            let mut out =
                crate::interp::net_combine(&x, &y, true).map_err(|m| err("E-SHAPE", m, span))?;
            out.parts.retain(|k, _| !except.contains(k));
            out.name = "delta".into();
            for p in out.parts.values_mut() {
                p.base = None;
            }
            one(Value::Net(out.into_ref()))
        }
        "wdiff" => {
            need(2)?;
            let (x, y) = (net_of(&a[0], span)?, net_of(&a[1], span)?);
            let (xb, yb) = (x.borrow(), y.borrow());
            let mut worst: f64 = 0.0;
            let mut lines = String::new();
            for (k, p) in &xb.parts {
                let q = match yb.parts.get(k) {
                    Some(q) => q,
                    None => {
                        lines += &format!("  {k}: missing on the right\n");
                        worst = f64::INFINITY;
                        continue;
                    }
                };
                if p.shape != q.shape {
                    lines += &format!("  {k}: shapes differ\n");
                    worst = f64::INFINITY;
                    continue;
                }
                let d = p
                    .data
                    .iter()
                    .zip(&q.data)
                    .map(|(u, v)| (u - v).abs())
                    .fold(0.0, f64::max);
                let same32 = p
                    .data
                    .iter()
                    .zip(&q.data)
                    .all(|(u, v)| (*u as f32).to_bits() == (*v as f32).to_bits());
                lines += &format!(
                    "  {k:<12} max |diff| {d:.3e}   float32-identical: {}\n",
                    if same32 { "yes" } else { "no" }
                );
                worst = worst.max(d);
            }
            it.print(&lines);
            t1(Tensor::scalar(worst))
        }
        "compile" => {
            need(1)?;
            let n = net_of(&a[0], span)?;
            let layers = {
                let nb = n.borrow();
                let (sizes, acts) = match &nb.arch {
                    Arch::Mlp { sizes, acts } => (sizes.clone(), acts.clone()),
                    _ => return Err(err("E-UNIMPLEMENTED", "the LLVM backend compiles MLP forward passes; transformers run interpreted", span)),
                };
                let mut layers = Vec::new();
                for l in 1..sizes.len() {
                    let w = nb
                        .parts
                        .get(&format!("W{l}"))
                        .ok_or_else(|| err("E-UNDEFINED", format!("missing W{l}"), span))?;
                    let b = nb
                        .parts
                        .get(&format!("b{l}"))
                        .map(|p| p.data.clone())
                        .unwrap_or(vec![0.0; sizes[l]]);
                    layers.push(wisp_llvm::ir::Dense {
                        w: w.data.clone(),
                        b,
                        n_out: sizes[l],
                        n_in: sizes[l - 1],
                        act: acts[l - 1].clone(),
                    });
                }
                layers
            };
            let ir = wisp_llvm::ir::mlp_forward(&layers).map_err(|m| err("E-LLVM", m, span))?;
            let lib = wisp_llvm::jit::compile_ir(&ir, "-O2").map_err(|m| err("E-LLVM", m, span))?;
            let nf = wisp_llvm::jit::NativeForward::load(
                lib,
                layers[0].n_in,
                layers.last().unwrap().n_out,
            )
            .map_err(|m| err("E-LLVM", m, span))?;
            n.borrow_mut().native = Some(Rc::new(nf));
            it.print(&format!("compile: forward pass compiled to native code with LLVM ({} layers, {} bytes of IR)\n", layers.len(), ir.len()));
            one(Value::Net(n))
        }
        "tf" | "attn" | "gen" | "pair" | "confirm" => {
            crate::observations::call(it, name, a, nargout, span)
        }
        "wcanon" | "walign" | "baseline" | "rollout" => Err(err(
            "E-UNIMPLEMENTED",
            format!("`{name}` is specified but not implemented in this prototype"),
            span,
        )),
        // ------------------------------------------------ observation
        "budget" => {
            if a.len() > 1 {
                return Err(err(
                    "E-ARGS",
                    "budget accepts an optional total pass limit",
                    span,
                ));
            }
            if let Some(value) = a.first() {
                let value = num(it, value, span, "observation budget")?;
                if !value.is_finite()
                    || !(0.0..=9007199254740991.0).contains(&value)
                    || value != value.trunc()
                {
                    return Err(err(
                        "E-ARGS",
                        "budget must be a non-negative exact integer",
                        span,
                    ));
                }
                let requested = value as u64;
                let limit = it
                    .harness
                    .passes
                    .map_or(requested, |cap| cap.min(requested));
                if limit < it.harness.used {
                    return Err(err(
                        "E-BUDGET",
                        "budget cannot be lower than passes already used",
                        span,
                    ));
                }
                it.harness.passes = Some(limit);
            }
            let left = it.harness.left().map(|x| x as f64).unwrap_or(f64::INFINITY);
            t1(Tensor::scalar(left))
        }
        "dataset" => {
            if a.len() != 2 && a.len() != 4 {
                return Err(err(
                    "E-ARGS",
                    "dataset needs a name, an input array or cell, and optionally 'heldout', true",
                    span,
                ));
            }
            let name = string(&a[0], span, "dataset name")?;
            if name.trim().is_empty() {
                return Err(err("E-ARGS", "dataset name must not be empty", span));
            }
            let heldout = if a.len() == 4 {
                let options = opts(a, 2, span)?;
                for key in options.keys() {
                    if key != "heldout" {
                        return Err(err(
                            "E-ARGS",
                            format!("unknown dataset option '{key}'"),
                            span,
                        ));
                    }
                }
                match options.get("heldout").map(|v| num(it, v, span, "heldout")) {
                    Some(Ok(0.0)) => false,
                    Some(Ok(1.0)) => true,
                    _ => return Err(err("E-ARGS", "heldout must be true or false", span)),
                }
            } else {
                false
            };
            it.harness
                .register_dataset(&name, a[1].clone(), heldout)
                .map_err(|m| err("E-DATASET", m, span))?;
            one(Value::Nothing)
        }
        "protect" => {
            if a.len() != 1 {
                return Err(err(
                    "E-ARGS",
                    "protect needs a part name or cell of part names",
                    span,
                ));
            }
            let names = cell_strings(&a[0], span, "protected parts")?;
            if names.iter().any(|s| s.trim().is_empty()) {
                return Err(err(
                    "E-ARGS",
                    "protected part names must not be empty",
                    span,
                ));
            }
            for name in names {
                if !it.harness.protect.contains(&name) {
                    it.harness.protect.push(name);
                }
            }
            one(Value::Nothing)
        }
        "activations" => {
            if a.len() != 1 {
                return Err(err(
                    "E-ARGS",
                    "activations needs one trusted cache name",
                    span,
                ));
            }
            let cache = string(&a[0], span, "activation cache name")?;
            let data = it
                .harness
                .resolve_data(&a[0])
                .map_err(|(c, m)| err(c, m, span))?;
            if !it.harness.is_activation_set(&cache) {
                return Err(err(
                    "E-FORBIDDEN",
                    "activation caches must be registered by the host",
                    span,
                ));
            }
            let loc = it.loc(span);
            match it.harness.observe("activations", 1, &loc, &data) {
                Ok(()) => {}
                Err(crate::harness::ObserveError::HeldOut) => return Err(err(
                    "E-HELDOUT",
                    "held-out data may only be observed by score; declare a separate fitting set",
                    span,
                )),
                Err(crate::harness::ObserveError::Budget(m)) => {
                    return Err(err("E-BUDGET", m, span))
                }
            }
            one(data.value)
        }
        "run" => {
            need(2)?;
            let n = net_of(&a[0], span)?;
            let loc = it.loc(span);
            let data = it
                .harness
                .resolve_data(&a[1])
                .map_err(|(c, m)| err(c, m, span))?;
            match it.harness.observe("run", 1, &loc, &data) {
                Ok(()) => {}
                Err(crate::harness::ObserveError::HeldOut) => return Err(err(
                    "E-HELDOUT",
                    "held-out data may only be observed by score; declare a separate fitting set",
                    span,
                )),
                Err(crate::harness::ObserveError::Budget(m)) => {
                    return Err(err("E-BUDGET", m, span))
                }
            }
            let nb = n.borrow();
            if let Arch::OpenThai2(graph) = &nb.arch {
                if a.len() != 2 || nargout > 2 {
                    return Err(err("E-ARGS", "OCR run takes (model, NCHW input) and returns probabilities and optional states", span));
                }
                let input = arr(it, &data.value, span)?;
                let (output, states) = crate::ocr::forward(&nb, graph, &input, nargout == 2)
                    .map_err(|m| err("E-OCR", m, span))?;
                let mut result = vec![Value::Arr(output)];
                if nargout == 2 {
                    result.push(Value::Struct(Rc::new(states.into_iter().map(|(k,v)| (k,Value::Arr(v))).collect())));
                }
                return Ok(result);
            }
            let out = match (&nb.arch, &data.value) {
                (Arch::Mlp { sizes, .. }, x) => {
                    let x = arr(it, x, span)?;
                    let (rows, n_in) = (x.rows(), x.cols());
                    if n_in != sizes[0] {
                        return Err(err("E-SHAPE", format!("the MLP takes rows of {} inputs, got {}", sizes[0], x.shape_str()), span));
                    }
                    let n_out = *sizes.last().unwrap();
                    let mut d = Vec::with_capacity(rows * n_out);
                    for r in 0..rows {
                        let row = &x.data[r * n_in..(r + 1) * n_in];
                        let y = match &nb.native {
                            Some(nf) => nf.run(row),
                            None => forward::mlp_one(&nb, row).map_err(|m| err("E-SHAPE", m, span))?,
                        };
                        d.extend(y);
                    }
                    Tensor::new(vec![rows, n_out], d, Tag::Act)
                }
                (Arch::Transformer(c), Value::Cell(seqs)) => {
                    let v = c.vocab.len();
                    let mut d = Vec::with_capacity(seqs.len() * v);
                    for s in seqs.iter() {
                        let toks = crate::observations::token_ids(c, s, span)?;
                        d.extend(forward::transformer_one(&nb, &toks).map_err(|m| err("E-SHAPE", m, span))?);
                    }
                    Tensor::new(vec![seqs.len(), v], d, Tag::Act)
                }
                _ => return Err(err("E-TYPE", "run takes an MLP with an input array, or a transformer with a cell of sequences", span)),
            };
            t1(out)
        }
        "cause" => {
            need(4)?;
            let cname = string(&a[0], span, "the claim name")?;
            let edit = match &a[1] {
                Value::Func(f) => f.clone(),
                _ => {
                    return Err(err(
                        "E-TYPE",
                        "cause needs an edit function, for example @(g) ...",
                        span,
                    ))
                }
            };
            let doses = arr(it, &a[2], span)?;
            if doses.numel() < 2 {
                return Err(err(
                    "E-SINGLE-DOSE",
                    "a causal claim needs a dose-response: give at least two doses",
                    span,
                )
                .hint("for example cause(name, @edit, [1 3 6], @metric)"));
            }
            let metric = match &a[3] {
                Value::Func(f) => f.clone(),
                _ => {
                    return Err(err(
                        "E-TYPE",
                        "cause needs a metric function that takes a net",
                        span,
                    ))
                }
            };
            let mut scores = Vec::new();
            for &d in &doses.data {
                let netv = it.call_value(&edit, vec![Value::num(d)], 1, span)?;
                let netv = netv
                    .into_iter()
                    .next()
                    .ok_or_else(|| err("E-ARGS", "the edit function returned nothing", span))?;
                let s = it.call_value(&metric, vec![netv], 1, span)?;
                let s = s
                    .into_iter()
                    .next()
                    .ok_or_else(|| err("E-ARGS", "the metric returned nothing", span))?;
                let sv = num(it, &s, span, "the metric")?;
                it.print(&format!(
                    "cause '{cname}': dose {} -> {}\n",
                    fmt_num(d),
                    fmt_num(sv)
                ));
                scores.push(sv);
            }
            let bi = (0..scores.len()).fold(0, |b, i| if scores[i] > scores[b] { i } else { b });
            let mut m = IndexMap::new();
            m.insert("name".into(), Value::str(&cname));
            m.insert("doses".into(), Value::Arr(doses.clone()));
            m.insert(
                "scores".into(),
                Value::Arr(Tensor::row(scores.clone()).with_tag(Tag::Act)),
            );
            m.insert("best".into(), Value::num(doses.data[bi]));
            m.insert(
                "best_score".into(),
                Value::Arr(Tensor::scalar(scores[bi]).with_tag(Tag::Act)),
            );
            m.insert(
                "monotone".into(),
                Value::boolean(
                    scores.windows(2).all(|w| w[1] >= w[0])
                        || scores.windows(2).all(|w| w[1] <= w[0]),
                ),
            );
            one(Value::strukt(m))
        }
        "grad_solve" => {
            if !it.harness.allows("grad") {
                return Err(err(
                    "E-FORBIDDEN",
                    "gradients are not allowed by this task",
                    span,
                )
                .hint("a task must list 'grad' in task.allow"));
            }
            Err(err(
                "E-UNIMPLEMENTED",
                "grad_solve is specified but not implemented in this prototype",
                span,
            ))
        }
        _ => {
            let _ = nargout;
            Err(err("E-UNDEFINED", format!("no built-in `{name}`"), span))
        }
    }
}

fn static_code(c: &str) -> &'static str {
    match c {
        "E-PROTECTED" => "E-PROTECTED",
        "E-UNSUPPORTED" => "E-UNSUPPORTED",
        "E-GLOBAL-KNOB" => "E-GLOBAL-KNOB",
        "E-EDIT-BUDGET" => "E-EDIT-BUDGET",
        _ => "E-IO",
    }
}

/// remove(net, group, u, g, ...) -> net, or remove(x, u, g, ...) -> array
fn remove(it: &mut Interp, a: &[Value], span: Span) -> R<Vec<Value>> {
    if a.len() < 3 {
        return Err(err(
            "E-ARGS",
            "remove(x, u, g, 'on', side) or remove(net, group, u, g, 'on', side)",
            span,
        ));
    }
    let group_form = matches!(&a[0], Value::Net(_)) && matches!(&a[1], Value::Group(_));
    let (k0, targets): (usize, Vec<(NetRef, String)>) = if group_form {
        match &a[1] {
            Value::Group(g) => (2, g.iter().cloned().collect()),
            _ => unreachable!(),
        }
    } else {
        (1, vec![])
    };
    let u_val = a[k0].clone();
    let g = num(it, &a[k0 + 1], span, "the gain")?;
    let o = opts(a, k0 + 2, span)?;
    let side = o
        .get("on")
        .and_then(|v| v.as_str())
        .unwrap_or("out")
        .to_string();
    let top = match o.get("top") {
        Some(v) => Some(usize_arg(it, v, span, "'top'")?),
        None => None,
    };
    let dir_for = |it: &Interp, part: &str| -> R<Vec<f64>> {
        match &u_val {
            Value::Cell(c) => {
                let l = layer_of(part).ok_or_else(|| {
                    err(
                        "E-ARGS",
                        format!("a cell of directions needs layer parts, got `{part}`"),
                        span,
                    )
                })?;
                let v = c.get(l).ok_or_else(|| {
                    err(
                        "E-INDEX-RANGE",
                        format!("no direction u{{{l}}} for `{part}`"),
                        span,
                    )
                })?;
                Ok(arr(it, v, span)?.data)
            }
            v => Ok(arr(it, v, span)?.data),
        }
    };
    let axis_for = |nb: &NetData, part: &str| -> R<usize> {
        match side.as_str() {
            "out" => Ok(0),
            "in" => Ok(1),
            "resid" => nb.resid_axis(part).ok_or_else(|| {
                err(
                    "E-TYPE",
                    format!("`{part}` does not touch the residual stream"),
                    span,
                )
            }),
            other => Err(err(
                "E-ARGS",
                format!("'on' must be 'out', 'in' or 'resid', got '{other}'"),
                span,
            )),
        }
    };
    if group_form {
        let base = net_of(&a[0], span)?;
        let mut out = base.borrow().clone();
        let mut touched = Vec::new();
        for (_, p) in &targets {
            let u = dir_for(it, p)?;
            let axis = axis_for(&out, p)?;
            let pd = out
                .parts
                .get_mut(p)
                .ok_or_else(|| err("E-UNDEFINED", format!("no part `{p}`"), span))?;
            let shape = pd.shape.clone();
            remove_dir(&mut pd.data, &shape, &u, g, axis, top)
                .map_err(|m| err("E-SHAPE", format!("{p}: {m}"), span))?;
            let ut = match &u_val {
                Value::Arr(t) => t.tag,
                _ => Tag::Act,
            };
            pd.max_tag = pd.max_tag.max(ut).max(Tag::Alg);
            touched.push(p.clone());
        }
        it.print(&format!(
            "remove: {} part(s) touched: {}\n",
            touched.len(),
            compress(&touched)
        ));
        return Ok(vec![Value::Net(out.into_ref())]);
    }
    // single array or part
    let (mut t, axis) = match &a[0] {
        Value::Part(n, p) => {
            let nb = n.borrow();
            (nb.parts[p.as_ref()].tensor(), axis_for(&nb, p)?)
        }
        v => {
            if side == "resid" {
                return Err(err(
                    "E-ARGS",
                    "'resid' needs a part of a net, not a plain array",
                    span,
                ));
            }
            (arr(it, v, span)?, if side == "in" { 1 } else { 0 })
        }
    };
    let u = dir_for(it, "")?;
    let shape = t.shape.clone();
    remove_dir(&mut t.data, &shape, &u, g, axis, top).map_err(|m| err("E-SHAPE", m, span))?;
    t.tag = Tag::Alg.max(t.tag);
    let _ = index::region_shape;
    let _ = FuncVal::Named(String::new());
    t1(t)
}
