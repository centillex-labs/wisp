//! Budgeted observations over inline inputs or harness-owned named datasets.
use crate::builtins::{arr, net_of, num, opts, usize_arg};
use crate::forward;
use crate::harness::ObservationData;
use crate::interp::{err, Interp, R};
use crate::net::{Arch, TCfg};
use crate::tensor::{Tag, Tensor};
use crate::value::{FuncVal, Value};
use indexmap::IndexMap;
use std::rc::Rc;
use wisp_syntax::Span;

pub fn token_ids(c: &TCfg, value: &Value, span: Span) -> R<Vec<usize>> {
    let lookup = |s: &str| {
        c.vocab.iter().position(|t| t == s).ok_or_else(|| {
            err(
                "E-UNDEFINED",
                format!("token `{s}` is not in the vocabulary"),
                span,
            )
        })
    };
    match value {
        Value::Str(s) if c.vocab.iter().all(|t| t.chars().count() == 1) => s.chars().map(|ch| lookup(&ch.to_string())).collect(),
        Value::Cell(tokens) => tokens.iter().map(|v| {
            let s = v.as_str().ok_or_else(|| err("E-TYPE", "a token cell must contain strings", span))?;
            lookup(s)
        }).collect(),
        Value::Arr(t) if t.is_vector() => t.data.iter().map(|&x| {
            if !x.is_finite() || x < 0.0 || x.fract() != 0.0 || x >= c.vocab.len() as f64 {
                return Err(err("E-INDEX-RANGE", "token IDs must be integers within the vocabulary", span));
            }
            Ok(x as usize)
        }).collect(),
        _ => Err(err("E-TYPE", "a sequence must be a string for a character vocabulary, a cell of token strings, or an array of token IDs", span)),
    }
}

fn inputs(it: &Interp, v: &Value, span: Span) -> R<ObservationData> {
    it.harness
        .resolve_data(v)
        .map_err(|(code, msg)| err(code, msg, span))
}
fn charge(it: &mut Interp, name: &str, cost: u64, data: &ObservationData, span: Span) -> R<()> {
    let loc = it.loc(span);
    match it.harness.observe(name, cost, &loc, data) {
        Ok(()) => Ok(()),
        Err(crate::harness::ObserveError::HeldOut) => Err(err(
            "E-HELDOUT",
            "held-out data may only be observed by score; declare a separate fitting set",
            span,
        )),
        Err(crate::harness::ObserveError::Budget(m)) => Err(err("E-BUDGET", m, span)),
    }
}
fn sequences(c: &TCfg, value: &Value, span: Span) -> R<Vec<Vec<usize>>> {
    let Value::Cell(items) = value else {
        return Err(err(
            "E-TYPE",
            "transformer observations take a cell of sequences or a named dataset",
            span,
        ));
    };
    items
        .iter()
        .map(|item| {
            let ids = token_ids(c, item, span)?;
            if ids.is_empty() || ids.len() > c.ctx {
                return Err(err(
                    "E-SHAPE",
                    format!("sequence length {} is outside 1..{}", ids.len(), c.ctx),
                    span,
                ));
            }
            Ok(ids)
        })
        .collect()
}
fn function(v: &Value, span: Span) -> R<Rc<FuncVal>> {
    match v {
        Value::Func(f) => Ok(f.clone()),
        _ => Err(err("E-TYPE", "expected a metric function", span)),
    }
}
fn act_scalar(x: f64) -> Value {
    Value::Arr(Tensor::scalar(x).with_tag(Tag::Act))
}
fn count(value: &Value, span: Span) -> R<usize> {
    match value {
        Value::Arr(t) if t.shape.len() == 2 => Ok(t.rows()),
        Value::Cell(c) => Ok(c.len()),
        _ => Err(err(
            "E-TYPE",
            "items must be a matrix of input rows or a cell of sequences",
            span,
        )),
    }
}

pub fn call(it: &mut Interp, name: &str, a: &[Value], nargout: usize, span: Span) -> R<Vec<Value>> {
    let required = match name {
        "gen" => 3,
        "pair" | "confirm" => 4,
        _ => 2,
    };
    if a.len() < required || (name != "gen" && a.len() != required) {
        return Err(err(
            "E-ARGS",
            format!("{name} needs {required} positional arguments"),
            span,
        ));
    }
    if name == "pair" || name == "confirm" {
        return paired(it, name, a, span);
    }
    let net = net_of(&a[0], span)?;
    let cfg = match &net.borrow().arch {
        Arch::Transformer(c) => c.clone(),
        _ => return Err(err("E-TYPE", format!("{name} needs a transformer"), span)),
    };
    let data = inputs(it, &a[1], span)?;
    let seqs = sequences(&cfg, &data.value, span)?;
    if name == "gen" {
        return generate(it, a, net, &cfg, data, seqs, span);
    }
    let cost = if name == "tf" {
        (seqs.len() as u64).div_ceil(1024)
    } else {
        seqs.len() as u64
    };
    charge(it, name, cost, &data, span)?;
    let nb = net.borrow();
    let mut logprobs = Vec::new();
    let mut hidden = Vec::new();
    let mut maps = Vec::new();
    for ids in &seqs {
        let trace = forward::transformer_trace(&nb, ids, name == "attn")
            .map_err(|m| err("E-SHAPE", m, span))?;
        if name == "attn" {
            maps.push(Value::Arr(Tensor::new(
                vec![cfg.layers, cfg.heads, ids.len(), ids.len()],
                trace.attention,
                Tag::Act,
            )));
        } else {
            let mut lp = Vec::with_capacity(ids.len().saturating_sub(1));
            for (t, &next) in ids.iter().enumerate().skip(1) {
                let logits = forward::transformer_logits(&nb, &trace.hidden[t - 1])
                    .map_err(|m| err("E-SHAPE", m, span))?;
                let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let logz = logits.iter().map(|x| (x - m).exp()).sum::<f64>().ln();
                let value = (logits[next] - m) - logz;
                if !value.is_finite() {
                    return Err(err("E-SHAPE", "teacher-forced logits are not finite", span));
                }
                lp.push(value);
            }
            logprobs.push(Value::Arr(Tensor::row(lp).with_tag(Tag::Act)));
            hidden.push(Value::Arr(Tensor::new(
                vec![ids.len(), cfg.d],
                trace.hidden.into_iter().flatten().collect(),
                Tag::Act,
            )));
        }
    }
    if name == "attn" {
        return Ok(vec![Value::cell(maps)]);
    }
    let (lp, hidden) = (Value::cell(logprobs), Value::cell(hidden));
    if nargout >= 2 {
        return Ok(vec![lp, hidden]);
    }
    let mut out = IndexMap::new();
    out.insert("logprobs".into(), lp);
    out.insert("hidden".into(), hidden);
    Ok(vec![Value::strukt(out)])
}

fn metric_scores(
    it: &mut Interp,
    metric: &FuncVal,
    net: Value,
    items: Value,
    n: usize,
    span: Span,
) -> R<Tensor> {
    let before = it.harness.observation_count;
    let result = it.call_value(metric, vec![net, items], 1, span)?;
    let value = result
        .first()
        .ok_or_else(|| err("E-ARGS", "metric returned no scores", span))?;
    let scores = arr(it, value, span)?;
    if scores.numel() != n || !scores.is_vector() || scores.data.iter().any(|x| !x.is_finite()) {
        return Err(err(
            "E-SHAPE",
            format!("metric must return {n} finite per-item scores as a vector"),
            span,
        ));
    }
    if it.harness.observation_count == before || scores.tag < Tag::Act {
        return Err(err(
            "E-FORBIDDEN",
            "a paired metric must observe its supplied data and return observation-derived scores",
            span,
        ));
    }
    Ok(scores)
}

fn paired(it: &mut Interp, name: &str, a: &[Value], span: Span) -> R<Vec<Value>> {
    net_of(&a[0], span)?;
    net_of(&a[1], span)?;
    let metric = function(&a[3], span)?;
    if it.harness.covered_by.is_some() {
        return Err(err(
            "E-FORBIDDEN",
            "pair/confirm cannot be nested inside another paired metric",
            span,
        ));
    }
    if name == "confirm" && !matches!(a[2], Value::Str(_)) {
        return Err(err(
            "E-TYPE",
            "confirm requires a registered dataset name",
            span,
        ));
    }
    let data = inputs(it, &a[2], span)?;
    let n = count(&data.value, span)?;
    if n < 2 {
        return Err(err(
            "E-SHAPE",
            "paired estimates need at least two items",
            span,
        ));
    }
    if name == "confirm" {
        it.harness
            .require_fresh(&data)
            .map_err(|(c, m)| err(c, m, span))?;
    }
    charge(it, name, 1, &data, span)?;
    let old_cover = it.harness.covered_by.replace(name.into());
    let old_required = it.harness.required_data.replace(data.key.clone());
    // Always restore the enclosing harness state, even if the first metric fails.
    let result = (|| {
        let left = metric_scores(it, &metric, a[0].clone(), a[2].clone(), n, span)?;
        let right = metric_scores(it, &metric, a[1].clone(), a[2].clone(), n, span)?;
        let differences: Vec<f64> = right
            .data
            .iter()
            .zip(&left.data)
            .map(|(b, a)| b - a)
            .collect();
        let gain = differences.iter().sum::<f64>() / n as f64;
        let variance = differences.iter().map(|d| (d - gain).powi(2)).sum::<f64>() / (n - 1) as f64;
        let se = (variance / n as f64).sqrt();
        if !gain.is_finite() || !se.is_finite() {
            return Err(err(
                "E-SHAPE",
                "paired gain or uncertainty overflowed",
                span,
            ));
        }
        Ok((gain, se))
    })();
    it.harness.covered_by = old_cover;
    it.harness.required_data = old_required;
    let (gain, se) = result?;
    if name == "pair" {
        return Ok(vec![act_scalar(gain), act_scalar(se)]);
    }
    let ok = gain > 2.0 * se;
    it.print(&format!(
        "confirm on '{}': gain {gain} (se {se}) -> {}\n",
        data.name.unwrap(),
        if ok { "confirmed" } else { "not established" }
    ));
    let mut result = Tensor::boolean(ok);
    result.tag = Tag::Act;
    Ok(vec![Value::Arr(result)])
}

fn generate(
    it: &mut Interp,
    a: &[Value],
    net: crate::net::NetRef,
    cfg: &TCfg,
    data: ObservationData,
    seqs: Vec<Vec<usize>>,
    span: Span,
) -> R<Vec<Value>> {
    let steps = usize_arg(it, &a[2], span, "maximum new tokens")?;
    if steps > cfg.ctx || seqs.iter().any(|s| s.len() > cfg.ctx - steps) {
        return Err(err(
            "E-SHAPE",
            "prompt plus generated tokens exceeds context length",
            span,
        ));
    }
    let options = opts(a, 3, span)?;
    for k in options.keys() {
        if !["seed", "temperature", "grade"].contains(&k.as_str()) {
            return Err(err("E-ARGS", format!("unknown gen option '{k}'"), span));
        }
    }
    let seed = options
        .get("seed")
        .map(|v| usize_arg(it, v, span, "seed"))
        .transpose()?
        .unwrap_or(0) as u64;
    let temperature = options
        .get("temperature")
        .map(|v| num(it, v, span, "temperature"))
        .transpose()?
        .unwrap_or(1.0);
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(err(
            "E-ARGS",
            "temperature must be finite and nonnegative",
            span,
        ));
    }
    let grader = options
        .get("grade")
        .map(|v| function(v, span))
        .transpose()?;
    charge(it, "gen", (seqs.len() as u64).div_ceil(512), &data, span)?;
    let mut state = seed.wrapping_add(0x9e3779b97f4a7c15);
    let mut completions = Vec::new();
    let mut tokens = Vec::new();
    let mut scores = Vec::new();
    for mut ids in seqs {
        let mut generated = Vec::new();
        for _ in 0..steps {
            let logits = forward::transformer_one(&net.borrow(), &ids)
                .map_err(|m| err("E-SHAPE", m, span))?;
            if logits.iter().any(|x| !x.is_finite()) {
                return Err(err("E-SHAPE", "generation logits must be finite", span));
            }
            let mut best = 0;
            for i in 1..logits.len() {
                if logits[i] > logits[best] {
                    best = i;
                }
            }
            let token = if temperature == 0.0 {
                best
            } else {
                // SplitMix64: deterministic local sampling; never uses global RNG state.
                state = state.wrapping_add(0x9e3779b97f4a7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                z ^= z >> 31;
                let p: Vec<f64> = logits
                    .iter()
                    .map(|x| ((x - logits[best]) / temperature).exp())
                    .collect();
                let draw = ((z >> 11) as f64) / 9007199254740992.0 * p.iter().sum::<f64>();
                let mut total = 0.0;
                let mut chosen = p.len() - 1;
                for (i, p) in p.iter().enumerate() {
                    total += p;
                    if draw < total {
                        chosen = i;
                        break;
                    }
                }
                chosen
            };
            ids.push(token);
            generated.push(token);
        }
        let text: String = generated.iter().map(|&t| cfg.vocab[t].as_str()).collect();
        if let Some(grader) = &grader {
            let values = it.call_value(grader, vec![Value::str(&text)], 1, span)?;
            let v = values
                .first()
                .ok_or_else(|| err("E-ARGS", "grader returned nothing", span))?;
            let score = num(it, v, span, "grade")?;
            if !score.is_finite() {
                return Err(err("E-SHAPE", "grade must be finite", span));
            }
            scores.push(score);
        }
        completions.push(Value::str(text));
        tokens.push(Value::Arr(
            Tensor::row(generated.into_iter().map(|x| x as f64).collect()).with_tag(Tag::Act),
        ));
    }
    let mut result = IndexMap::new();
    result.insert("texts".into(), Value::cell(completions));
    result.insert("tokens".into(), Value::cell(tokens));
    if grader.is_some() {
        result.insert(
            "scores".into(),
            Value::Arr(Tensor::col(scores).with_tag(Tag::Act)),
        );
    }
    Ok(vec![Value::strukt(result)])
}
