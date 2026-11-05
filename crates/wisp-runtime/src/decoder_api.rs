use crate::{
    builtins::{cell_strings, net_of, num, opts, string, usize_arg},
    decoder::Model,
    hf::{self, DownloadOptions},
    interp::{err, Interp, R},
    net::{Arch, NetData},
    tensor::{Tag, Tensor},
    value::Value,
};
use indexmap::IndexMap;
use wisp_syntax::Span;

pub fn is_decoder(value: Option<&Value>) -> bool {
    matches!(value, Some(Value::Net(n)) if n.borrow().decoder.is_some())
}
fn array(shape: Vec<usize>, data: Vec<f64>) -> Value {
    Value::Arr(Tensor::new(shape, data, Tag::Act))
}
fn best(logits: &[f64]) -> usize {
    let mut top = 0;
    for i in 1..logits.len() {
        if logits[i] > logits[top] {
            top = i;
        }
    }
    top
}

pub fn call(it: &mut Interp, name: &str, a: &[Value], span: Span) -> R<Vec<Value>> {
    let fail = |m: String| err("E-MODEL", m, span);
    let count = match name {
        "hf_model" => 1,
        "gen" => 3,
        "score" => 4,
        _ => 2,
    };
    let variadic = matches!(name, "hf_model" | "gen" | "capture" | "read_weights");
    if a.len() < count || (!variadic && a.len() != count) {
        return Err(err(
            "E-ARGS",
            format!("{name} needs {count} positional arguments"),
            span,
        ));
    }
    if name == "hf_model" {
        let repo = string(&a[0], span, "model repository")?;
        let options = opts(a, 1, span)?;
        for key in options.keys() {
            if !["revision", "sha256", "offline"].contains(&key.as_str()) {
                return Err(err(
                    "E-ARGS",
                    format!("unknown hf_model option '{key}'"),
                    span,
                ));
            }
        }
        let revision = string(
            options
                .get("revision")
                .ok_or_else(|| err("E-ARGS", "hf_model requires a pinned revision", span))?,
            span,
            "revision",
        )?;
        let offline = match options
            .get("offline")
            .map(|v| num(it, v, span, "offline"))
            .transpose()?
        {
            None | Some(0.0) => false,
            Some(1.0) => true,
            _ => return Err(err("E-ARGS", "offline must be boolean", span)),
        };
        let sha256 = options
            .get("sha256")
            .map(|v| string(v, span, "SHA-256"))
            .transpose()?;
        let small = DownloadOptions {
            revision: revision.clone(),
            offline,
            max_bytes: 64 * 1024 * 1024,
            ..Default::default()
        };
        let config =
            hf::download(&repo, "config.json", &small).map_err(|m| err("E-HF", m, span))?;
        Model::validate_config(&config).map_err(fail)?;
        let tokenizer =
            hf::download(&repo, "tokenizer.json", &small).map_err(|m| err("E-HF", m, span))?;
        let weights = hf::download(
            &repo,
            "model.safetensors",
            &DownloadOptions {
                revision,
                sha256,
                offline,
                ..Default::default()
            },
        )
        .map_err(|m| err("E-HF", m, span))?;
        let model = Model::load(&weights, &config, &tokenizer).map_err(fail)?;
        let mut net = NetData::new(&repo, Arch::Unknown);
        net.source = Some(weights.display().to_string());
        net.meta.insert(
            "cache_dir".into(),
            Value::str(weights.parent().unwrap().display().to_string()),
        );
        net.meta.insert(
            "context".into(),
            Value::Arr(Tensor::scalar(model.context() as f64)),
        );
        net.meta
            .insert("identity".into(), Value::str(model.identity()));
        net.decoder = Some(model);
        return Ok(vec![Value::Net(net.into_ref())]);
    }
    let net = net_of(&a[0], span)?;
    let model = net
        .borrow()
        .decoder
        .clone()
        .ok_or_else(|| err("E-TYPE", "expected a native model from hf_model", span))?;
    if name == "token_head" {
        let texts = cell_strings(&a[1], span, "output tokens")?;
        let ids: Vec<_> = texts
            .iter()
            .map(|s| model.token_id(s))
            .collect::<Result<_, _>>()
            .map_err(fail)?;
        let mut head = model.token_head(&ids).map_err(fail)?;
        head.meta.insert(
            "token_ids".into(),
            Value::Arr(Tensor::row(ids.into_iter().map(f64::from).collect())),
        );
        return Ok(vec![Value::Net(head.into_ref())]);
    }
    if name == "with_head" {
        let head = net_of(&a[1], span)?;
        let edited = model.with_head(&head.borrow()).map_err(fail)?;
        let mut result = net.borrow().clone();
        result.decoder = Some(edited);
        return Ok(vec![Value::Net(result.into_ref())]);
    }
    if name == "read_weights" {
        let names = cell_strings(&a[1], span, "tensor names")?;
        if names.is_empty() {
            return Err(err(
                "E-ARGS",
                "read_weights needs at least one tensor name",
                span,
            ));
        }
        let source = net
            .borrow()
            .source
            .clone()
            .ok_or_else(|| fail("this model has no checkpoint path".to_string()))?;
        let options = crate::io::LoadOpts {
            in_out: false,
            act: "linear".into(),
        };
        let mut parts = crate::io::load_selected(&source, &options, &names).map_err(&fail)?;
        parts
            .meta
            .insert("model_identity".into(), Value::str(model.identity()));
        return Ok(vec![Value::Net(parts.into_ref())]);
    }
    if name == "with_writers" {
        let parts = net_of(&a[1], span)?;
        let edited = model.with_writers(&parts.borrow()).map_err(fail)?;
        let mut result = net.borrow().clone();
        result.decoder = Some(edited);
        result.meta.insert(
            "edited_parts".into(),
            Value::cell(
                result
                    .decoder
                    .as_ref()
                    .unwrap()
                    .edited_parts()
                    .iter()
                    .map(Value::str)
                    .collect(),
            ),
        );
        return Ok(vec![Value::Net(result.into_ref())]);
    }
    if name == "score" {
        let steps = usize_arg(it, &a[2], span, "maximum new tokens")?;
        let grader = match &a[3] {
            Value::Func(f) => f.clone(),
            _ => {
                return Err(err(
                    "E-ARGS",
                    "score needs a grader function as its fourth argument",
                    span,
                ))
            }
        };
        if steps == 0 || steps > 64 {
            return Err(err("E-ARGS", "score requires 1..64 new tokens", span));
        }
        let data = it
            .harness
            .resolve_data(&a[1])
            .map_err(|(c, m)| err(c, m, span))?;
        let prompts = cell_strings(&data.value, span, "prompts")?;
        if prompts.is_empty() || prompts.len() > 1024 {
            return Err(err("E-ARGS", "score requires 1..1024 prompts", span));
        }
        let sequences: Vec<_> = prompts
            .iter()
            .map(|p| model.encode(p))
            .collect::<Result<_, _>>()
            .map_err(fail)?;
        if sequences.iter().any(|s| s.len() + steps > model.context()) {
            return Err(err(
                "E-ARGS",
                "scored prompts must fit the context limit with room for new tokens",
                span,
            ));
        }
        let loc = it.loc(span);
        let cost = (sequences.len() as u64).div_ceil(512);
        it.harness
            .observe_score(cost, &loc, &data)
            .map_err(|m| err("E-BUDGET", m, span))?;
        let mut texts = Vec::new();
        let mut scores = Vec::new();
        for mut ids in sequences {
            let mut generated = Vec::new();
            for _ in 0..steps {
                let (_, l) = model.forward(&ids).map_err(fail)?;
                let next = best(&l) as u32;
                generated.push(next);
                ids.push(next);
                if next == model.eos() {
                    break;
                }
            }
            let text = model.decode(&generated).map_err(fail)?;
            let value = it
                .call_value(&grader, vec![Value::str(text.clone())], 1, span)?
                .into_iter()
                .next();
            let x = match value {
                Some(v) => num(it, &v, span, "the grader result")?,
                None => {
                    return Err(err(
                        "E-ARGS",
                        "the grader must return one number per response",
                        span,
                    ))
                }
            };
            scores.push(x);
            texts.push(Value::str(text));
        }
        let n = scores.len() as f64;
        let mean = scores.iter().sum::<f64>() / n;
        let mut result = IndexMap::new();
        result.insert("scores".into(), array(vec![scores.len(), 1], scores));
        result.insert("mean".into(), array(vec![1, 1], vec![mean]));
        result.insert("texts".into(), Value::cell(texts));
        return Ok(vec![Value::strukt(result)]);
    }
    let data = it
        .harness
        .resolve_data(&a[1])
        .map_err(|(c, m)| err(c, m, span))?;
    let prompts = cell_strings(&data.value, span, "prompts")?;
    if prompts.is_empty() || prompts.len() > 1024 {
        return Err(err(
            "E-ARGS",
            "native inference requires 1..1024 prompts",
            span,
        ));
    }
    let sequences: Vec<_> = prompts
        .iter()
        .map(|p| model.encode(p))
        .collect::<Result<_, _>>()
        .map_err(fail)?;
    let (layers_wanted, attn_wanted) = if name == "capture" && a.len() > 2 {
        let options = opts(a, 2, span)?;
        for key in options.keys() {
            if !["layers", "attn"].contains(&key.as_str()) {
                return Err(err(
                    "E-ARGS",
                    format!("unknown capture option '{key}'"),
                    span,
                ));
            }
        }
        let indices = |key: &str| -> R<Vec<usize>> {
            match options.get(key) {
                None => Ok(vec![]),
                Some(Value::Cell(c)) => c
                    .iter()
                    .map(|v| usize_arg(it, v, span, key))
                    .collect::<R<Vec<_>>>(),
                Some(Value::Arr(t)) => t
                    .data
                    .iter()
                    .map(|v| {
                        usize_arg(
                            it,
                            &Value::Arr(crate::tensor::Tensor::scalar(*v)),
                            span,
                            key,
                        )
                    })
                    .collect::<R<Vec<_>>>(),
                Some(_) => Err(err(
                    "E-ARGS",
                    format!("{key} needs an array or cell of layer indices"),
                    span,
                )),
            }
        };
        (indices("layers")?, indices("attn")?)
    } else {
        (vec![], vec![])
    };
    let steps = if name == "gen" {
        let steps = usize_arg(it, &a[2], span, "maximum new tokens")?;
        let options = opts(a, 3, span)?;
        for key in options.keys() {
            if key != "temperature" {
                return Err(err(
                    "E-ARGS",
                    format!("native gen does not support '{key}'"),
                    span,
                ));
            }
        }
        let temp = options
            .get("temperature")
            .map(|v| num(it, v, span, "temperature"))
            .transpose()?
            .unwrap_or(0.0);
        if temp != 0.0 {
            return Err(err(
                "E-ARGS",
                "native gen currently requires temperature 0",
                span,
            ));
        }
        if steps == 0 || steps > 64 || sequences.iter().any(|s| s.len() + steps > model.context()) {
            return Err(err(
                "E-ARGS",
                "native gen requires 1..64 new tokens within the context limit",
                span,
            ));
        }
        steps
    } else {
        0
    };
    let loc = it.loc(span);
    let cost = if name == "gen" {
        (sequences.len() as u64).div_ceil(512)
    } else {
        1
    };
    match it.harness.observe(name, cost, &loc, &data) {
        Ok(()) => {}
        Err(crate::harness::ObserveError::HeldOut) => {
            return Err(err(
                "E-HELDOUT",
                "held-out data may only be observed by score; declare a separate fitting set",
                span,
            ))
        }
        Err(crate::harness::ObserveError::Budget(m)) => return Err(err("E-BUDGET", m, span)),
    }
    let mut texts = Vec::new();
    let mut tokens = Vec::new();
    let mut hidden = Vec::new();
    let mut logits = Vec::new();
    let mut layer_states: Vec<Vec<f64>> = Vec::new();
    let mut attn_states: Vec<Vec<f64>> = Vec::new();
    for mut ids in sequences {
        if name == "gen" {
            let mut generated = Vec::new();
            for _ in 0..steps {
                let (_, scores) = model.forward(&ids).map_err(fail)?;
                let next = best(&scores) as u32;
                generated.push(next);
                ids.push(next);
                if next == model.eos() {
                    break;
                }
            }
            texts.push(Value::str(model.decode(&generated).map_err(fail)?));
            tokens.push(array(
                vec![1, generated.len()],
                generated.into_iter().map(f64::from).collect(),
            ));
        } else {
            let (h, l, layer_states_prompt, attn_states_prompt) = model
                .forward_trace(&ids, &layers_wanted, &attn_wanted)
                .map_err(fail)?;
            if name == "capture" {
                if !layers_wanted.is_empty() {
                    layer_states.push(
                        layer_states_prompt
                            .into_iter()
                            .flat_map(|(_, v)| v)
                            .collect(),
                    );
                }
                if !attn_wanted.is_empty() {
                    attn_states.push(
                        attn_states_prompt
                            .into_iter()
                            .flat_map(|(_, v)| v)
                            .collect(),
                    );
                }
            }
            texts.push(Value::str(model.decode(&[best(&l) as u32]).map_err(fail)?));
            tokens.push(array(
                vec![1, ids.len()],
                ids.into_iter().map(f64::from).collect(),
            ));
            hidden.extend(h);
            logits.extend(l);
        }
    }
    if name == "run" {
        return Ok(vec![array(vec![prompts.len(), model.vocab()], logits)]);
    }
    let mut result = IndexMap::new();
    result.insert("texts".into(), Value::cell(texts));
    if name == "capture" {
        result.insert("prompt_tokens".into(), Value::cell(tokens));
        result.insert(
            "hidden".into(),
            array(vec![prompts.len(), model.width()], hidden),
        );
        result.insert(
            "logits".into(),
            array(vec![prompts.len(), model.vocab()], logits),
        );
        if !layers_wanted.is_empty() {
            result.insert(
                "layers".into(),
                array(
                    vec![prompts.len() * layers_wanted.len(), model.width()],
                    layer_states.into_iter().flatten().collect(),
                ),
            );
            let ids: Vec<f64> = layers_wanted.iter().map(|l| *l as f64).collect();
            result.insert("layer_ids".into(), array(vec![1, ids.len()], ids));
        }
        if !attn_wanted.is_empty() {
            let attn_len = attn_states.first().map_or(0, |v| v.len());
            result.insert(
                "attn".into(),
                array(
                    vec![prompts.len() * attn_wanted.len(), attn_len],
                    attn_states.into_iter().flatten().collect(),
                ),
            );
            let ids: Vec<f64> = attn_wanted.iter().map(|l| *l as f64).collect();
            result.insert("attn_layer_ids".into(), array(vec![1, ids.len()], ids));
        }
    } else {
        result.insert("tokens".into(), Value::cell(tokens));
    }
    Ok(vec![Value::strukt(result)])
}
