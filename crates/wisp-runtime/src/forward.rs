//! Reference forward passes for the architectures in scope (no layer norm).

use crate::net::{Arch, NetData};

pub fn act(name: &str, x: f64) -> f64 {
    match name {
        "tanh" => x.tanh(),
        "relu" => x.max(0.0),
        "sigmoid" => 1.0 / (1.0 + (-x).exp()),
        "gelu" => 0.5 * x * (1.0 + (0.7978845608028654 * (x + 0.044715 * x * x * x)).tanh()),
        _ => x,
    }
}

fn part<'a>(net: &'a NetData, name: &str) -> Result<&'a [f64], String> {
    net.parts
        .get(name)
        .map(|p| p.data.as_slice())
        .ok_or(format!("the net has no part `{name}`"))
}

/// y = W x + b for W `[out, in]`.
fn dense(w: &[f64], b: Option<&[f64]>, x: &[f64], n_out: usize) -> Vec<f64> {
    let n_in = x.len();
    let mut y = vec![0.0; n_out];
    for o in 0..n_out {
        let row = &w[o * n_in..(o + 1) * n_in];
        let mut s = b.map(|b| b[o]).unwrap_or(0.0);
        for i in 0..n_in {
            s += row[i] * x[i];
        }
        y[o] = s;
    }
    y
}

/// MLP forward pass for one input vector.
pub fn mlp_one(net: &NetData, x: &[f64]) -> Result<Vec<f64>, String> {
    let (sizes, acts) = match &net.arch {
        Arch::Mlp { sizes, acts } => (sizes, acts),
        _ => return Err("not an MLP".into()),
    };
    if x.len() != sizes[0] {
        return Err(format!(
            "the MLP takes {} inputs, got {}",
            sizes[0],
            x.len()
        ));
    }
    let mut h = x.to_vec();
    for l in 1..sizes.len() {
        let w = part(net, &format!("W{l}"))?;
        let b = net.parts.get(&format!("b{l}")).map(|p| p.data.as_slice());
        let z = dense(w, b, &h, sizes[l]);
        let a = &acts[l - 1];
        h = z.into_iter().map(|v| act(a, v)).collect();
    }
    Ok(h)
}

pub struct TransformerTrace {
    pub hidden: Vec<Vec<f64>>,
    /// Row-major [layer, head, query, key], with masked entries zero.
    pub attention: Vec<f64>,
}

/// Transformer logits at the last position of one token sequence.
pub fn transformer_one(net: &NetData, toks: &[usize]) -> Result<Vec<f64>, String> {
    let trace = transformer_trace(net, toks, false)?;
    transformer_logits(net, trace.hidden.last().unwrap())
}

pub fn transformer_logits(net: &NetData, hidden: &[f64]) -> Result<Vec<f64>, String> {
    let Arch::Transformer(c) = &net.arch else {
        return Err("not a transformer".into());
    };
    let w = part(net, "unembed")?;
    if hidden.len() != c.d || w.len() != c.vocab.len() * c.d {
        return Err("unembed shape does not match the transformer config".into());
    }
    Ok(dense(w, None, hidden, c.vocab.len()))
}

pub fn transformer_trace(
    net: &NetData,
    toks: &[usize],
    capture_attention: bool,
) -> Result<TransformerTrace, String> {
    let c = match &net.arch {
        Arch::Transformer(c) => c,
        _ => return Err("not a transformer".into()),
    };
    let (d, v, hd, heads) = (c.d, c.vocab.len(), c.head_dim, c.heads);
    let t_len = toks.len();
    if t_len == 0 || t_len > c.ctx {
        return Err(format!("sequence length {t_len} is outside 1..{}", c.ctx));
    }
    if toks.iter().any(|&t| t >= v) {
        return Err("token ID is outside the vocabulary".into());
    }
    if d == 0 || heads == 0 || hd == 0 {
        return Err("transformer dimensions must be positive".into());
    }
    let mut attention = if capture_attention {
        vec![0.0; c.layers * heads * t_len * t_len]
    } else {
        vec![]
    };
    let embed = part(net, "embed")?; // [d, v]
    if embed.len() != d * v {
        return Err("embed shape does not match the transformer config".into());
    }
    let pos = net.parts.get("pos").map(|p| p.data.as_slice()); // [d, ctx]
    if pos.is_some_and(|p| p.len() != d * c.ctx) {
        return Err("position shape does not match the transformer config".into());
    }
    let mut x: Vec<Vec<f64>> = toks
        .iter()
        .enumerate()
        .map(|(t, &tok)| {
            (0..d)
                .map(|i| embed[i * v + tok] + pos.map(|p| p[i * c.ctx + t]).unwrap_or(0.0))
                .collect()
        })
        .collect();
    for l in 0..c.layers {
        let g = |n: &str| part(net, &format!("L[{l}].{n}"));
        let (wq, wk, wv, wo) = (g("attn.q")?, g("attn.k")?, g("attn.v")?, g("attn.o")?);
        let ha = heads * hd;
        if [wq, wk, wv].iter().any(|w| w.len() != ha * d) || wo.len() != d * ha {
            return Err(format!(
                "layer {l} attention shapes do not match the transformer config"
            ));
        }
        for (name, size) in [("attn.o_b", d), ("mlp.up_b", c.ff), ("mlp.down_b", d)] {
            if net
                .parts
                .get(&format!("L[{l}].{name}"))
                .is_some_and(|p| p.data.len() != size)
            {
                return Err(format!(
                    "layer {l} {name} shape does not match the transformer config"
                ));
            }
        }
        let q: Vec<Vec<f64>> = x.iter().map(|xt| dense(wq, None, xt, ha)).collect();
        let k: Vec<Vec<f64>> = x.iter().map(|xt| dense(wk, None, xt, ha)).collect();
        let vv: Vec<Vec<f64>> = x.iter().map(|xt| dense(wv, None, xt, ha)).collect();
        let scale = 1.0 / (hd as f64).sqrt();
        let mut newx = x.clone();
        for t in 0..t_len {
            let mut cat = vec![0.0; ha];
            for h in 0..heads {
                let r = h * hd..(h + 1) * hd;
                let scores: Vec<f64> = (0..=t)
                    .map(|s| {
                        q[t][r.clone()]
                            .iter()
                            .zip(&k[s][r.clone()])
                            .map(|(a, b)| a * b)
                            .sum::<f64>()
                            * scale
                    })
                    .collect();
                let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
                let z: f64 = e.iter().sum();
                for s in 0..=t {
                    let a = e[s] / z;
                    if capture_attention {
                        attention[((l * heads + h) * t_len + t) * t_len + s] = a;
                    }
                    for j in r.clone() {
                        cat[j] += a * vv[s][j];
                    }
                }
            }
            let o = dense(
                wo,
                net.parts
                    .get(&format!("L[{l}].attn.o_b"))
                    .map(|p| p.data.as_slice()),
                &cat,
                d,
            );
            for i in 0..d {
                newx[t][i] += o[i];
            }
        }
        x = newx;
        let up = g("mlp.up")?;
        let up_b = net
            .parts
            .get(&format!("L[{l}].mlp.up_b"))
            .map(|p| p.data.as_slice());
        let down = g("mlp.down")?;
        let down_b = net
            .parts
            .get(&format!("L[{l}].mlp.down_b"))
            .map(|p| p.data.as_slice());
        if up.len() != c.ff * d || down.len() != d * c.ff {
            return Err(format!(
                "layer {l} MLP shapes do not match the transformer config"
            ));
        }
        for state in x.iter_mut().take(t_len) {
            let h: Vec<f64> = dense(up, up_b, state, c.ff)
                .into_iter()
                .map(|z| act(&c.act, z))
                .collect();
            let o = dense(down, down_b, &h, d);
            for (value, delta) in state.iter_mut().zip(o) {
                *value += delta;
            }
        }
    }
    Ok(TransformerTrace {
        hidden: x,
        attention,
    })
}
