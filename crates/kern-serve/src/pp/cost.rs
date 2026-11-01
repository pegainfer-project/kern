//! What an item costs each stage, as the head's policy believes it: per
//! stage s, an item packing sequences i with n_i new rows over p_i rows of
//! context costs
//!
//! ```text
//! t_s = a_s + b_s Σ n_i + c_s Σ p_i + d_s Σ n_i (p_i + n_i / 2) + e_s · sequences
//! ```
//!
//! read from a fit of the pipeline's own item timings (`--pp-cost`, the
//! format `{"fit": {"<stage>": {"a_ms", "b_us_per_row", "c_ms_per_1k_ctx",
//! "d_ns_per_row_ctx", "e_ms_per_seq"?}}}`). The model is the policy's
//! knowledge of the pipeline and nothing else: the runtime never reads it.
//!
//! The sums run left to right with Python's compensated float sum, so a
//! policy growing an item one part at a time and one costing it whole get
//! the same bits, and so does an offline replay of the same decisions.

use anyhow::{bail, Context, Result};
use serde_json::Value;

/// Stages a cost model may have.
pub const MAX_STAGES: usize = 16;

/// One time per stage, µs; entries past the model's stages are zero.
pub type Times = [f64; MAX_STAGES];

/// Per stage: µs fixed, per row, per context token, per row × context
/// token, per packed sequence.
#[derive(Clone, Debug, PartialEq)]
pub struct Cost {
    stages: Vec<[f64; 5]>,
}

impl Cost {
    /// A costfit's stages, in numeric order of their keys.
    pub fn parse(text: &str) -> Result<Cost> {
        let fit: Value = serde_json::from_str(text).context("a costfit is JSON")?;
        let f = fit.get("fit").and_then(Value::as_object).context("a costfit has a `fit` object")?;
        let mut stages: Vec<(u64, &Value)> = f
            .iter()
            .filter(|(k, _)| !k.is_empty() && k.bytes().all(|b| b.is_ascii_digit()))
            .map(|(k, v)| (k.parse().expect("digits parse"), v))
            .collect();
        stages.sort_by_key(|&(k, _)| k);
        if !(1..=MAX_STAGES).contains(&stages.len()) {
            bail!("a costfit has {} stages, expected 1..={MAX_STAGES}", stages.len());
        }
        let stages = stages
            .into_iter()
            .enumerate()
            .map(|(i, (k, v))| {
                if k != i as u64 {
                    bail!("costfit stage `{k}` where stage `{i}` was expected");
                }
                let num = |name: &str| {
                    v.get(name).and_then(Value::as_f64).with_context(|| format!("costfit stage `{k}` has no `{name}`"))
                };
                let e = match v.get("e_ms_per_seq") {
                    None => 0.0,
                    Some(x) => {
                        x.as_f64().with_context(|| format!("costfit stage `{k}`: `e_ms_per_seq` is a number"))?
                    }
                };
                Ok([
                    num("a_ms")? * 1e3,
                    num("b_us_per_row")?,
                    num("c_ms_per_1k_ctx")?,
                    num("d_ns_per_row_ctx")? * 1e-3,
                    e * 1e3,
                ])
            })
            .collect::<Result<_>>()?;
        Ok(Cost { stages })
    }

    /// A model that thinks every item free: what a policy that never reads
    /// one is handed.
    pub fn free(stages: usize) -> Cost {
        Cost { stages: vec![[0.0; 5]; stages.clamp(1, MAX_STAGES)] }
    }

    pub fn stages(&self) -> usize {
        self.stages.len()
    }

    /// Each stage's time for `parts`, (context, rows) per packed sequence.
    pub fn times(&self, parts: &[(i64, i64)]) -> Times {
        self.of(&Item::of(parts))
    }

    /// The slowest stage's time for `parts`.
    pub fn slowest(&self, parts: &[(i64, i64)]) -> f64 {
        self.slowest_of(&Item::of(parts))
    }

    pub fn of(&self, it: &Item) -> Times {
        let mut out = [0.0; MAX_STAGES];
        let (rows, ctx, att) = (it.rows as f64, it.ctx as f64, it.att.value());
        for (o, &[a, b, c, d, e]) in out.iter_mut().zip(&self.stages) {
            let t = a + b * rows + c * ctx + d * att;
            *o = if e == 0.0 { t } else { t + e * it.parts as f64 };
        }
        out
    }

    pub fn slowest_of(&self, it: &Item) -> f64 {
        let t = self.of(it);
        t[1..self.stages()].iter().fold(t[0], |m, &x| max(m, x))
    }
}

/// An item as the model sees it, built one part at a time.
#[derive(Clone, Copy, Debug, Default)]
pub struct Item {
    parts: i64,
    rows: i64,
    ctx: i64,
    /// Σ n (p + n / 2).
    att: Sum,
}

impl Item {
    /// The item with one more part, `n` rows over `p` context.
    pub fn with(mut self, p: i64, n: i64) -> Item {
        self.parts += 1;
        self.rows += n;
        self.ctx += p;
        self.att = self.att.plus(n as f64 * (p as f64 + n as f64 / 2.0));
        self
    }

    fn of(parts: &[(i64, i64)]) -> Item {
        parts.iter().fold(Item::default(), |it, &(p, n)| it.with(p, n))
    }
}

/// Python's `max(a, b)`: the first wins a tie.
pub(super) fn max(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// Python 3.12's `sum` of floats as a running state: Neumaier-compensated,
/// the compensation added once at the end.
#[derive(Clone, Copy, Debug, Default)]
struct Sum {
    s: f64,
    c: f64,
}

impl Sum {
    fn plus(self, x: f64) -> Sum {
        let t = self.s + x;
        let c = self.c + if self.s.abs() >= x.abs() { (self.s - t) + x } else { (x - t) + self.s };
        Sum { s: t, c }
    }

    fn value(self) -> f64 {
        if self.c != 0.0 && self.c.is_finite() {
            self.s + self.c
        } else {
            self.s
        }
    }
}
