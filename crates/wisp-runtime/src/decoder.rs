//! Native CPU Qwen3 inference. Immutable decoder weights are shared by readout edits.
use candle_core::{DType, Device, Tensor};
use candle_nn::{ops, rotary_emb, VarBuilder};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokenizers::Tokenizer;

use crate::net::{self, Dtype, NetData};

type Result<T> = std::result::Result<T, String>;
type TraceList = Vec<(usize, Vec<f64>)>;
type TensorTrace = Vec<(usize, Tensor)>;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeadBinding {
    pub model: String,
    pub tokens: Vec<u32>,
    pub base_sha256: String,
}

#[derive(Clone)]
pub struct Model {
    base: Arc<Decoder>,
    head: Option<NetDataHead>,
    /// Canonical names of writer tensors replaced by `with_writers`, in order.
    edits: Vec<String>,
}
#[derive(Clone)]
struct NetDataHead {
    tokens: Vec<u32>,
    weights: Vec<f32>,
}

#[derive(Clone, Deserialize)]
struct Config {
    model_type: String,
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    max_position_embeddings: usize,
    rope_theta: f64,
    rope_scaling: Option<Yarn>,
    rms_norm_eps: f64,
    hidden_act: String,
    tie_word_embeddings: bool,
    attention_bias: bool,
    use_sliding_window: bool,
    eos_token_id: u32,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Yarn {
    rope_type: String,
    factor: f64,
    original_max_position_embeddings: usize,
    #[serde(default = "beta_fast")]
    beta_fast: f64,
    #[serde(default = "beta_slow")]
    beta_slow: f64,
    attention_factor: Option<f64>,
}
fn beta_fast() -> f64 {
    32.0
}
fn beta_slow() -> f64 {
    1.0
}

#[derive(Clone)]
struct Layer {
    input_norm: Tensor,
    post_norm: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
}
#[derive(Clone)]
struct Decoder {
    config: Config,
    embedding: Tensor,
    output: Tensor,
    layers: Vec<Layer>,
    norm: Tensor,
    tokenizer: Tokenizer,
    identity: String,
    dtype: Dtype,
    source: PathBuf,
    pool: std::sync::Arc<rayon::ThreadPool>,
}

fn file_hash(path: &Path) -> Result<String> {
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    let mut hash = Sha256::new();
    let mut buf = [0; 65536];
    loop {
        let n = file.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

impl Config {
    fn validate(&self) -> Result<()> {
        if self.model_type != "qwen3"
            || self.hidden_act != "silu"
            || self.attention_bias
            || self.use_sliding_window
        {
            return Err(
                "native backend supports Qwen3 with SiLU, full attention, and no attention bias"
                    .into(),
            );
        }
        if self.hidden_size == 0
            || self.hidden_size > 32768
            || self.intermediate_size == 0
            || self.intermediate_size > 131072
            || self.num_hidden_layers == 0
            || self.num_hidden_layers > 256
            || self.vocab_size == 0
            || self.vocab_size > 1_000_000
            || self.head_dim == 0
            || !self.head_dim.is_multiple_of(2)
            || self.head_dim > 512
            || self.num_attention_heads == 0
            || self.num_attention_heads > 256
            || self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
            || self.max_position_embeddings == 0
            || self.eos_token_id as usize >= self.vocab_size
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 1.0
            || !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
        {
            return Err(
                "invalid or unsupported Qwen3 dimensions or numerical configuration".into(),
            );
        }
        if let Some(y) = &self.rope_scaling {
            if y.rope_type != "yarn"
                || !y.factor.is_finite()
                || y.factor < 1.0
                || y.original_max_position_embeddings == 0
                || !y.beta_fast.is_finite()
                || !y.beta_slow.is_finite()
                || y.beta_slow <= 0.0
                || y.beta_fast <= y.beta_slow
                || y.attention_factor
                    .is_some_and(|a| !a.is_finite() || a <= 0.0)
            {
                return Err("unsupported or invalid YaRN configuration".into());
            }
        }
        Ok(())
    }
    fn context(&self) -> usize {
        self.max_position_embeddings.min(2048)
    }
    fn frequencies(&self) -> (Vec<f32>, f32) {
        let mut freq: Vec<_> = (0..self.head_dim)
            .step_by(2)
            .map(|i| 1.0 / (self.rope_theta as f32).powf(i as f32 / self.head_dim as f32))
            .collect();
        let Some(y) = &self.rope_scaling else {
            return (freq, 1.0);
        };
        let correction = |rot: f64| {
            self.head_dim as f64
                * (y.original_max_position_embeddings as f64 / (rot * 2.0 * std::f64::consts::PI))
                    .ln()
                / (2.0 * self.rope_theta.ln())
        };
        let low = correction(y.beta_fast).floor().max(0.0);
        let mut high = correction(y.beta_slow)
            .ceil()
            .min((self.head_dim - 1) as f64);
        if low == high {
            high += 0.001;
        }
        for (i, f) in freq.iter_mut().enumerate() {
            let ramp = ((i as f32 - low as f32) / (high as f32 - low as f32)).clamp(0.0, 1.0);
            *f = (*f / y.factor as f32) * ramp + *f * (1.0 - ramp);
        }
        (
            freq,
            y.attention_factor.unwrap_or(1.0 + 0.1 * y.factor.ln()) as f32,
        )
    }
}

impl Model {
    pub(crate) fn validate_config(path: &Path) -> Result<()> {
        let config: Config =
            serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        config.validate()
    }
    pub fn load(weights: &Path, config: &Path, tokenizer: &Path) -> Result<Self> {
        let cfg: Config =
            serde_json::from_slice(&std::fs::read(config).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        cfg.validate()?;
        let identity = format!(
            "wisp-qwen3-nonthinking-v1:{}:{}:{}",
            file_hash(weights)?,
            file_hash(config)?,
            file_hash(tokenizer)?
        );
        let weights_reader = crate::decoder_weights::Weights::open(weights)?;
        let dtype = weights_reader.dtype(if cfg.tie_word_embeddings {
            "model.embed_tokens.weight"
        } else {
            "lm_head.weight"
        })?;
        let tokenizer = Tokenizer::from_file(tokenizer).map_err(|e| e.to_string())?;
        for token in ["<|im_start|>", "<|im_end|>", "<think>", "</think>"] {
            if tokenizer.token_to_id(token).is_none() {
                return Err(format!("Qwen3 tokenizer is missing {token}"));
            }
        }
        let pool = std::sync::Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .build()
                .map_err(|e| e.to_string())?,
        );
        let vb = VarBuilder::from_backend(Box::new(weights_reader), DType::F32, Device::Cpu);
        let loaded = (|| -> candle_core::Result<_> {
            let d = cfg.hidden_size;
            let q = cfg.num_attention_heads * cfg.head_dim;
            let kv = cfg.num_key_value_heads * cfg.head_dim;
            let embedding = vb.get((cfg.vocab_size, d), "model.embed_tokens.weight")?;
            let output = if cfg.tie_word_embeddings {
                embedding.clone()
            } else {
                vb.get((cfg.vocab_size, d), "lm_head.weight")?
            };
            let norm = vb.get(d, "model.norm.weight")?;
            let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
            for i in 0..cfg.num_hidden_layers {
                let l = vb.pp(format!("model.layers.{i}"));
                layers.push(Layer {
                    input_norm: l.get(d, "input_layernorm.weight")?,
                    post_norm: l.get(d, "post_attention_layernorm.weight")?,
                    q: l.get((q, d), "self_attn.q_proj.weight")?,
                    k: l.get((kv, d), "self_attn.k_proj.weight")?,
                    v: l.get((kv, d), "self_attn.v_proj.weight")?,
                    o: l.get((d, q), "self_attn.o_proj.weight")?,
                    q_norm: l.get(cfg.head_dim, "self_attn.q_norm.weight")?,
                    k_norm: l.get(cfg.head_dim, "self_attn.k_norm.weight")?,
                    gate: l.get((cfg.intermediate_size, d), "mlp.gate_proj.weight")?,
                    up: l.get((cfg.intermediate_size, d), "mlp.up_proj.weight")?,
                    down: l.get((d, cfg.intermediate_size), "mlp.down_proj.weight")?,
                });
            }
            Ok((embedding, output, norm, layers))
        })()
        .map_err(|e| e.to_string())?;
        drop(vb);
        Ok(Self {
            base: Arc::new(Decoder {
                config: cfg,
                embedding: loaded.0,
                output: loaded.1,
                norm: loaded.2,
                layers: loaded.3,
                tokenizer,
                identity,
                dtype,
                source: weights.to_path_buf(),
                pool,
            }),
            head: None,
            edits: Vec::new(),
        })
    }
    pub fn parameter_count(&self) -> usize {
        let layer_weights: usize = self
            .base
            .layers
            .iter()
            .map(|l| {
                [
                    &l.input_norm,
                    &l.post_norm,
                    &l.q,
                    &l.k,
                    &l.v,
                    &l.o,
                    &l.q_norm,
                    &l.k_norm,
                    &l.gate,
                    &l.up,
                    &l.down,
                ]
                .iter()
                .map(|t| t.elem_count())
                .sum::<usize>()
            })
            .sum();
        layer_weights
            + self.base.embedding.elem_count()
            + self.base.norm.elem_count()
            + if self.base.config.tie_word_embeddings {
                0
            } else {
                self.base.output.elem_count()
            }
            + self.head.as_ref().map_or(0, |h| h.weights.len())
    }
    pub fn summary(&self) -> String {
        let edits = self.edits.len();
        format!(
            "Qwen3 CPU, {} layers, {} weights, {} edited output rows{}",
            self.base.layers.len(),
            self.parameter_count(),
            self.head.as_ref().map_or(0, |h| h.tokens.len()),
            if edits > 0 {
                format!(", {edits} edited writer tensors")
            } else {
                String::new()
            }
        )
    }
    pub fn width(&self) -> usize {
        self.base.config.hidden_size
    }
    pub fn vocab(&self) -> usize {
        self.base.config.vocab_size
    }
    pub fn context(&self) -> usize {
        self.base.config.context()
    }
    pub fn eos(&self) -> u32 {
        self.base.config.eos_token_id
    }
    pub fn identity(&self) -> &str {
        &self.base.identity
    }
    pub fn weights_path(&self) -> &Path {
        &self.base.source
    }
    pub fn edited_parts(&self) -> &[String] {
        &self.edits
    }
    pub fn encode(&self, prompt: &str) -> Result<Vec<u32>> {
        let text = format!(
            "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        let ids = self
            .base
            .tokenizer
            .encode(text, false)
            .map_err(|e| e.to_string())?
            .get_ids()
            .to_vec();
        if ids.is_empty()
            || ids.len() > self.context()
            || ids.iter().any(|&t| t as usize >= self.vocab())
        {
            return Err(format!(
                "native Qwen3 prompt must fit {} tokens and the model vocabulary",
                self.context()
            ));
        }
        Ok(ids)
    }
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.base
            .tokenizer
            .decode(ids, true)
            .map_err(|e| e.to_string())
    }
    pub fn token_id(&self, text: &str) -> Result<u32> {
        let enc = self
            .base
            .tokenizer
            .encode(text, false)
            .map_err(|e| e.to_string())?;
        match enc.get_ids() {
            [id] if (*id as usize) < self.vocab() => Ok(*id),
            _ => Err(format!("{text:?} must encode as exactly one token")),
        }
    }
    pub fn forward(&self, ids: &[u32]) -> Result<(Vec<f64>, Vec<f64>)> {
        let (h, logits, _, _) = self.forward_trace(ids, &[], &[])?;
        Ok((h, logits))
    }

    /// Final-position hidden state, logits, and per-layer traces. Layer entries
    /// carry the residual stream after that block, before the final norm;
    /// attention entries carry the last query position's probabilities flattened
    /// as heads by sequence position. `layers` and `attn` are layer indices in
    /// any order; each is reported at most once, in the requested order.
    pub fn forward_trace(
        &self,
        ids: &[u32],
        layers: &[usize],
        attn: &[usize],
    ) -> Result<(Vec<f64>, Vec<f64>, TraceList, TraceList)> {
        let want = |list: &[usize]| -> Result<Vec<usize>> {
            let mut sorted = list.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            if sorted.last().is_some_and(|l| *l >= self.base.layers.len()) {
                return Err(format!(
                    "layer index out of range; this model has {} layers",
                    self.base.layers.len()
                ));
            }
            Ok(sorted)
        };
        let want_layers = want(layers)?;
        let want_attn = want(attn)?;
        if ids.is_empty()
            || ids.len() > self.context()
            || ids.iter().any(|&t| t as usize >= self.vocab())
        {
            return Err("invalid native decoder input".into());
        }
        self.base
            .pool
            .install(|| {
                let (h, logits, layer_states, attn_probs) =
                    self.base.forward_ex(ids, &want_layers, &want_attn)?;
                let mut logits = logits.flatten_all()?.to_vec1::<f32>()?;
                if let Some(head) = &self.head {
                    let weights = Tensor::from_vec(
                        head.weights.clone(),
                        (head.tokens.len(), self.width()),
                        &Device::Cpu,
                    )?;
                    let edited = h.matmul(&weights.t()?)?.flatten_all()?.to_vec1::<f32>()?;
                    for (&token, value) in head.tokens.iter().zip(edited) {
                        logits[token as usize] = value;
                    }
                }
                let hidden = h.flatten_all()?.to_vec1::<f32>()?;
                if hidden.iter().chain(&logits).any(|x| !x.is_finite()) {
                    candle_core::bail!("native inference produced non-finite values");
                }
                let collect =
                    |list: &[(usize, Tensor)]| -> candle_core::Result<Vec<(usize, Vec<f64>)>> {
                        list.iter()
                            .map(|(i, t)| {
                                Ok((
                                    *i,
                                    t.flatten_all()?
                                        .to_vec1::<f32>()?
                                        .into_iter()
                                        .map(f64::from)
                                        .collect(),
                                ))
                            })
                            .collect()
                    };
                let layer_states = collect(&layer_states)?;
                let attn_probs = collect(&attn_probs)?;
                if layer_states
                    .iter()
                    .chain(&attn_probs)
                    .flat_map(|(_, v)| v.iter())
                    .any(|x| !x.is_finite())
                {
                    candle_core::bail!("native inference produced non-finite values");
                }
                Ok((
                    hidden.into_iter().map(f64::from).collect(),
                    logits.into_iter().map(f64::from).collect(),
                    layer_states,
                    attn_probs,
                ))
            })
            .map_err(|e: candle_core::Error| e.to_string())
    }
    pub fn token_head(&self, tokens: &[u32]) -> Result<NetData> {
        if tokens.is_empty()
            || tokens.iter().any(|&t| t as usize >= self.vocab())
            || tokens
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len()
                != tokens.len()
        {
            return Err("token_head requires distinct tokens within the vocabulary".into());
        }
        let ids = Tensor::new(tokens, &Device::Cpu).map_err(|e| e.to_string())?;
        let rows = self
            .base
            .output
            .index_select(&ids, 0)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| e.to_string())?;
        let mut net = net::mlp(&[self.width(), tokens.len()], "linear");
        net.name = "Qwen3 output rows".into();
        let w = net.parts.get_mut("W1").unwrap();
        w.dtype = self.base.dtype;
        w.data = rows.into_iter().map(f64::from).collect();
        net.head_binding = Some(HeadBinding {
            model: self.base.identity.clone(),
            tokens: tokens.to_vec(),
            base_sha256: crate::io::part_sha(w, &w.data),
        });
        if let Some(head) = &self.head {
            let w = net.parts.get_mut("W1").unwrap();
            for (r, id) in tokens.iter().enumerate() {
                if let Some(i) = head.tokens.iter().position(|t| t == id) {
                    for j in 0..self.width() {
                        w.data[r * self.width() + j] = head.weights[i * self.width() + j] as f64;
                    }
                }
            }
        }
        net.snapshot_base();
        Ok(net)
    }
    pub fn with_head(&self, head: &NetData) -> Result<Self> {
        let binding = head
            .head_binding
            .as_ref()
            .ok_or("head has no native-model binding")?;
        if binding.model != self.base.identity {
            return Err("head belongs to a different checkpoint, config, or tokenizer".into());
        }
        let expected = self.token_head(&binding.tokens)?;
        if binding.base_sha256 != expected.head_binding.as_ref().unwrap().base_sha256 {
            return Err("head source-row hash mismatch".into());
        }
        let w = head.parts.get("W1").ok_or("head is missing W1")?;
        let b = head.parts.get("b1").ok_or("head is missing b1")?;
        if head.parts.len() != 2
            || w.shape != [binding.tokens.len(), self.width()]
            || w.data.len() != binding.tokens.len() * self.width()
            || b.shape != [binding.tokens.len()]
            || b.data.len() != binding.tokens.len()
            || b.data.iter().any(|&v| v != 0.0)
        {
            return Err(
                "head must contain only the bound output rows and unchanged zero biases".into(),
            );
        }
        let weights: Vec<f32> = w.data.iter().map(|&v| w.dtype.round(v) as f32).collect();
        if weights.iter().any(|v| !v.is_finite()) {
            return Err("head weights must remain finite in their stored dtype".into());
        }
        let mut result = self.clone();
        // Replacing a readout is explicit; do not silently stack or mutate shared models.
        result.head = Some(NetDataHead {
            tokens: binding.tokens.clone(),
            weights,
        });
        Ok(result)
    }

    /// Replace writer tensors (the matrices whose columns write into the
    /// residual stream, and the other layer projections) with edited versions.
    /// `parts` must come from `read_weights` on this same model so the binding
    /// is provenance-checked; tensors are rounded to the checkpoint's stored
    /// dtype before use.
    pub fn with_writers(&self, parts: &NetData) -> Result<Self> {
        let bound = parts
            .meta
            .get("model_identity")
            .and_then(|v| v.as_str())
            .ok_or("writer parts carry no model binding; read them with read_weights")?;
        if bound != self.base.identity {
            return Err(
                "writer parts belong to a different checkpoint, config, or tokenizer".into(),
            );
        }
        let c = &self.base.config;
        let d = c.hidden_size;
        let q = c.num_attention_heads * c.head_dim;
        let kv = c.num_key_value_heads * c.head_dim;
        let ff = c.intermediate_size;
        let mut base = (*self.base.clone()).clone();
        let mut applied: Vec<String> = Vec::new();
        for (name, part) in &parts.parts {
            let rest = name
                .strip_prefix("L[")
                .and_then(|s| s.split_once(']'))
                .ok_or_else(|| format!("writer part `{name}` is not a layer tensor"))?;
            let layer: usize = rest
                .0
                .parse()
                .map_err(|_| format!("bad layer in `{name}`"))?;
            let field = match rest.1 {
                ".mlp.down" => (0, d, ff),
                ".mlp.gate" => (0, ff, d),
                ".mlp.up" => (0, ff, d),
                ".attn.q" => (0, q, d),
                ".attn.k" => (0, kv, d),
                ".attn.v" => (0, kv, d),
                ".attn.o" => (0, d, q),
                other => {
                    return Err(format!(
                        "writer part `{name}` is not a supported layer projection ({other})"
                    ))
                }
            };
            if layer >= self.base.layers.len() {
                return Err(format!("writer part `{name}` is outside the model"));
            }
            let expected = [field.1, field.2];
            if part.shape != expected {
                return Err(format!(
                    "writer part `{name}` must be {} by {}, got {} by {}",
                    expected[0],
                    expected[1],
                    part.shape.first().copied().unwrap_or(0),
                    part.shape.get(1).copied().unwrap_or(0)
                ));
            }
            if part.dtype != self.base.dtype {
                return Err(format!(
                    "writer part `{name}` must stay in the checkpoint's stored dtype"
                ));
            }
            if part.data.iter().any(|v| !v.is_finite()) {
                return Err(format!("writer part `{name}` contains non-finite values"));
            }
            let tensor = Tensor::from_vec(
                part.data
                    .iter()
                    .map(|&v| part.dtype.round(v) as f32)
                    .collect(),
                (expected[0], expected[1]),
                &Device::Cpu,
            )
            .map_err(|e| e.to_string())?;
            let target = &mut base.layers[layer];
            match rest.1 {
                ".mlp.down" => target.down = tensor,
                ".mlp.gate" => target.gate = tensor,
                ".mlp.up" => target.up = tensor,
                ".attn.q" => target.q = tensor,
                ".attn.k" => target.k = tensor,
                ".attn.v" => target.v = tensor,
                _ => target.o = tensor,
            }
            applied.push(name.clone());
        }
        if applied.is_empty() {
            return Err("writer parts contain no layer tensors".into());
        }
        Ok(Self {
            base: Arc::new(base),
            head: self.head.clone(),
            edits: applied,
        })
    }
}

impl Decoder {
    fn forward_ex(
        &self,
        ids: &[u32],
        want_layers: &[usize],
        want_attn: &[usize],
    ) -> candle_core::Result<(Tensor, Tensor, TensorTrace, TensorTrace)> {
        let c = &self.config;
        let s = ids.len();
        let ids = Tensor::new(ids, &Device::Cpu)?;
        let mut x = self.embedding.index_select(&ids, 0)?;
        let (freq, factor) = c.frequencies();
        let phases: Vec<f32> = (0..s)
            .flat_map(|p| freq.iter().map(move |f| p as f32 * f))
            .collect();
        let phases = Tensor::from_vec(phases, (s, c.head_dim / 2), &Device::Cpu)?;
        let cos = phases.cos()?.affine(factor as f64, 0.0)?;
        let sin = phases.sin()?.affine(factor as f64, 0.0)?;
        let mask: Vec<f32> = (0..s)
            .flat_map(|i| (0..s).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        let mask = Tensor::from_vec(mask, (1, 1, s, s), &Device::Cpu)?;
        let eps = c.rms_norm_eps as f32;
        let mut layer_states = Vec::new();
        let mut attn_probs = Vec::new();
        for (li, l) in self.layers.iter().enumerate() {
            let z = ops::rms_norm(&x, &l.input_norm, eps)?;
            let q = z
                .matmul(&l.q.t()?)?
                .reshape((s, c.num_attention_heads, c.head_dim))?
                .transpose(0, 1)?
                .unsqueeze(0)?
                .contiguous()?;
            let k = z
                .matmul(&l.k.t()?)?
                .reshape((s, c.num_key_value_heads, c.head_dim))?
                .transpose(0, 1)?
                .unsqueeze(0)?
                .contiguous()?;
            let v = z
                .matmul(&l.v.t()?)?
                .reshape((s, c.num_key_value_heads, c.head_dim))?
                .transpose(0, 1)?
                .unsqueeze(0)?
                .contiguous()?;
            let q = rotary_emb::rope(&ops::rms_norm(&q, &l.q_norm, eps)?, &cos, &sin)?;
            let k = rotary_emb::rope(&ops::rms_norm(&k, &l.k_norm, eps)?, &cos, &sin)?;
            let repeat = |t: Tensor| {
                t.unsqueeze(2)?
                    .expand((
                        1,
                        c.num_key_value_heads,
                        c.num_attention_heads / c.num_key_value_heads,
                        s,
                        c.head_dim,
                    ))?
                    .reshape((1, c.num_attention_heads, s, c.head_dim))?
                    .contiguous()
            };
            let k = repeat(k)?;
            let v = repeat(v)?;
            let scores = q
                .matmul(&k.t()?)?
                .affine(1.0 / (c.head_dim as f64).sqrt(), 0.0)?
                .broadcast_add(&mask)?;
            let probs = ops::softmax_last_dim(&scores)?;
            if want_attn.binary_search(&li).is_ok() {
                // Last query position's distribution over inputs, heads by position.
                let last = probs.narrow(2, s - 1, 1)?.contiguous()?;
                attn_probs.push((li, last.reshape((1, c.num_attention_heads * s))?));
            }
            let context = probs
                .matmul(&v)?
                .transpose(1, 2)?
                .reshape((s, c.num_attention_heads * c.head_dim))?;
            x = (x + context.matmul(&l.o.t()?)?)?;
            let z = ops::rms_norm(&x, &l.post_norm, eps)?;
            let gated = (ops::silu(&z.matmul(&l.gate.t()?)?)? * z.matmul(&l.up.t()?)?)?;
            x = (x + gated.matmul(&l.down.t()?)?)?;
            if want_layers.binary_search(&li).is_ok() {
                let last = x.narrow(0, s - 1, 1)?.contiguous()?;
                layer_states.push((li, last));
            }
        }
        let h = ops::rms_norm(&x.narrow(0, s - 1, 1)?.contiguous()?, &self.norm, eps)?;
        let logits = h.matmul(&self.output.t()?)?;
        Ok((h, logits, layer_states, attn_probs))
    }
}
