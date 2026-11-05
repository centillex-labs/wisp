//! The harness: budget, permissions, protected parts, data roles and the run log.

use crate::value::Value;
use indexmap::IndexMap;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::HashSet;

pub struct ObservationData {
    pub value: Value,
    pub name: Option<String>,
    pub key: String,
    pub heldout: bool,
}

/// Content identity prevents renaming an already-observed set to confirm it.
fn data_key(value: &Value) -> Result<String, String> {
    fn hash(h: &mut Sha256, v: &Value) -> Result<(), String> {
        match v {
            Value::Arr(t) => {
                h.update(b"array");
                h.update((t.shape.len() as u64).to_le_bytes());
                for n in &t.shape {
                    h.update((*n as u64).to_le_bytes());
                }
                for x in &t.data {
                    h.update(x.to_bits().to_le_bytes());
                }
            }
            Value::Str(s) => {
                h.update(b"string");
                h.update((s.len() as u64).to_le_bytes());
                h.update(s.as_bytes());
            }
            Value::Cell(c) => {
                h.update(b"cell");
                h.update((c.len() as u64).to_le_bytes());
                for x in c.iter() {
                    hash(h, x)?;
                }
            }
            _ => {
                return Err(
                    "datasets contain only arrays, strings, or cells of these values".into(),
                )
            }
        }
        Ok(())
    }
    let mut h = Sha256::new();
    hash(&mut h, value)?;
    Ok(format!("{:x}", h.finalize()))
}
use std::fs::OpenOptions;
use std::io::Write;

/// Why an observation was refused. Held-out data has its own diagnostic code so
/// fit-purpose rejections are not mistaken for budget exhaustion.
pub enum ObserveError {
    Budget(String),
    HeldOut,
}

impl From<String> for ObserveError {
    fn from(m: String) -> Self {
        ObserveError::Budget(m)
    }
}

#[derive(Default)]
pub struct Harness {
    /// Passes allowed in total; None means unlimited.
    pub passes: Option<u64>,
    pub used: u64,
    pub allow: Vec<String>,
    pub protect: Vec<String>,
    pub test_sets: Vec<String>,
    pub observed: Vec<String>,
    pub practice_sets: Vec<String>,
    datasets: IndexMap<String, Dataset>,
    activation_sets: HashSet<String>,
    first_observation: IndexMap<String, String>,
    /// Restricts paired metric callbacks to the supplied data.
    pub required_data: Option<String>,
    pub observation_count: u64,
    pub log_path: Option<String>,
    pub log: Vec<serde_json::Value>,
    /// When set, charges are covered by an enclosing call (e.g. `confirm`).
    pub covered_by: Option<String>,
}

struct Dataset {
    value: Value,
    key: String,
    heldout: bool,
}

impl Harness {
    pub fn register_dataset(
        &mut self,
        name: &str,
        value: Value,
        heldout: bool,
    ) -> Result<(), String> {
        if self.datasets.contains_key(name) {
            return Err(format!("dataset `{name}` is already registered"));
        }
        if !matches!(value, Value::Arr(_) | Value::Cell(_)) {
            return Err("a dataset must be an array of rows or a cell of sequences".into());
        }
        let key = data_key(&value)?;
        self.datasets.insert(
            name.into(),
            Dataset {
                value,
                key,
                heldout,
            },
        );
        Ok(())
    }

    /// Trusted host registration only. Cached model behavior must not become Lit.
    pub fn register_activations(&mut self, name: &str, mut value: Value) -> Result<(), String> {
        let Value::Arr(ref mut tensor) = value else {
            return Err("activation caches must be numeric arrays".into());
        };
        tensor.tag = crate::tensor::Tag::Act;
        self.register_dataset(name, value, false)?;
        self.activation_sets.insert(name.into());
        Ok(())
    }

    pub fn is_activation_set(&self, name: &str) -> bool {
        self.activation_sets.contains(name)
    }

    pub fn resolve_data(&self, input: &Value) -> Result<ObservationData, (&'static str, String)> {
        let (value, name, key, heldout) = if let Value::Str(name) = input {
            if self.test_sets.iter().any(|n| n == name.as_ref()) {
                return Err((
                    "E-FORBIDDEN",
                    format!("`{name}` is a task.test set and cannot be observed"),
                ));
            }
            let (v, key, heldout) = self
                .datasets
                .get(name.as_ref())
                .map(|d| (d.value.clone(), d.key.clone(), d.heldout))
                .ok_or_else(|| {
                    (
                        "E-UNDEFINED",
                        format!(
                        "dataset `{name}` is not registered; declare it with dataset(name, inputs)"
                    ),
                    )
                })?;
            (v, Some(name.to_string()), key, heldout)
        } else {
            (
                input.clone(),
                None,
                data_key(input).map_err(|m| ("E-TYPE", m))?,
                false,
            )
        };
        for test in &self.test_sets {
            if self.datasets.get(test).is_some_and(|d| d.key == key) {
                return Err((
                    "E-FORBIDDEN",
                    "data matches a protected task.test set".into(),
                ));
            }
        }
        if self
            .required_data
            .as_ref()
            .is_some_and(|required| *required != key)
        {
            return Err(("E-FORBIDDEN", "a paired metric must observe the data supplied to pair/confirm, not a different set".into()));
        }
        Ok(ObservationData {
            value,
            name,
            key,
            heldout,
        })
    }

    pub fn require_fresh(&self, data: &ObservationData) -> Result<(), (&'static str, String)> {
        if let Some(first) = self.first_observation.get(&data.key) {
            return Err((
                "E-REUSED-SET",
                format!("confirmation data was already observed by {first}"),
            ));
        }
        Ok(())
    }

    /// Charge before recording exposure; a budget refusal does not consume a set.
    /// Fit-purpose calls (`capture`, `run`, `gen`, `tf`, `attn`, `pair`) go here so
    /// held-out sets are rejected; `score` uses `observe_score`.
    pub fn observe(
        &mut self,
        call: &str,
        cost: u64,
        loc: &str,
        data: &ObservationData,
    ) -> Result<(), ObserveError> {
        if data.heldout {
            return Err(ObserveError::HeldOut);
        }
        let desc = data.name.as_deref().unwrap_or("inline data");
        self.charge(call, cost, loc, desc)?;
        self.first_observation
            .entry(data.key.clone())
            .or_insert_with(|| format!("{call} at {loc}"));
        if let Some(name) = &data.name {
            if !self.observed.contains(name) {
                self.observed.push(name.clone());
            }
        }
        self.observation_count += 1;
        Ok(())
    }

    /// Charge a scoring pass. Held-out data is allowed, and scoring the same set
    /// repeatedly is what a keep-if-better loop needs, so freshness is not
    /// consumed. Used only by `score`.
    pub fn observe_score(
        &mut self,
        cost: u64,
        loc: &str,
        data: &ObservationData,
    ) -> Result<(), String> {
        let desc = data
            .name
            .as_deref()
            .map(|n| format!("{n} (held-out: {})", data.heldout))
            .unwrap_or_else(|| "inline data".to_string());
        self.charge("score", cost, loc, &desc)?;
        if let Some(name) = &data.name {
            if !self.observed.contains(name) {
                self.observed.push(name.clone());
            }
        }
        self.observation_count += 1;
        Ok(())
    }

    pub fn left(&self) -> Option<u64> {
        self.passes.map(|p| p.saturating_sub(self.used))
    }

    /// Charge `cost` passes for `call`. Fails without charging if the budget is too small.
    pub fn charge(&mut self, call: &str, cost: u64, loc: &str, desc: &str) -> Result<(), String> {
        if let Some(by) = self.covered_by.clone() {
            self.write(json!({"call": call, "cost": 0, "covered_by": by, "at": loc, "args": desc}));
            return Ok(());
        }
        if let Some(left) = self.left() {
            if cost > left {
                let msg = format!(
                    "`{call}` costs {cost} pass(es) but {left} of {} are left",
                    self.passes.unwrap()
                );
                self.write(json!({"call": call, "cost": cost, "refused": true, "left": left, "at": loc, "args": desc}));
                return Err(msg);
            }
        }
        self.used += cost;
        let left = self.left();
        self.write(json!({"call": call, "cost": cost, "used": self.used, "left": left, "at": loc, "args": desc}));
        Ok(())
    }

    pub fn allows(&self, what: &str) -> bool {
        self.allow.iter().any(|a| a == what)
    }

    fn write(&mut self, line: serde_json::Value) {
        if let Some(p) = &self.log_path {
            if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(p) {
                let _ = writeln!(f, "{line}");
            }
        }
        self.log.push(line);
    }
}
