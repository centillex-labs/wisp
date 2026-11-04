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
        _ => 2,
    };
    if a.len() < count || (!matches!(name, "gen" | "hf_model") && a.len() != count) {
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
    it.harness
        .observe(name, cost, &loc, &data)
        .map_err(|m| err("E-BUDGET", m, span))?;
    let mut texts = Vec::new();
    let mut tokens = Vec::new();
    let mut hidden = Vec::new();
    let mut logits = Vec::new();
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
            let (h, l) = model.forward(&ids).map_err(fail)?;
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
    } else {
        result.insert("tokens".into(), Value::cell(tokens));
    }
    Ok(vec![Value::strukt(result)])
}
