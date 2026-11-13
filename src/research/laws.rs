//! Multi-family mathematical law approximation & comparison for inhomogeneous
//! Markov transition probabilities `p_{i->j}(t)`.
//!
//! Unlike single-family polynomial regression, this module evaluates multiple
//! competing law families (`Constant`, `Linear`, `Quadratic`, `Cubic`,
//! `ExponentialRelax`, `Logarithmic`, `PowerRelax`) on normalised block time
//! `u = (t - t0) / max(1, span) in [0, 1]`, computes `RMSE`, `MAE`, `R^2`,
//! small-sample corrected `AICc`, and **out-of-block validation RMSE**.

use crate::error::{DemMcmError, Result};
use crate::gui::metrics_eval::vec_f64_json;
use crate::io::json::JsonValue;
use crate::linalg::{polyfit, polyval};

/// Candidate mathematical law families for `p_{i->j}(t)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LawFamily {
    Constant,
    Linear,
    Quadratic,
    Cubic,
    ExponentialRelax,
    Logarithmic,
    PowerRelax,
}

impl LawFamily {
    pub fn all() -> Vec<Self> {
        vec![
            Self::Constant,
            Self::Linear,
            Self::Quadratic,
            Self::Cubic,
            Self::ExponentialRelax,
            Self::Logarithmic,
            Self::PowerRelax,
        ]
    }

    pub fn id(self) -> &'static str {
        match self {
            Self::Constant => "constant",
            Self::Linear => "linear",
            Self::Quadratic => "quadratic",
            Self::Cubic => "cubic",
            Self::ExponentialRelax => "exponential",
            Self::Logarithmic => "logarithmic",
            Self::PowerRelax => "power",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::Constant => "Constante",
            Self::Linear => "Linéaire",
            Self::Quadratic => "Quadratique",
            Self::Cubic => "Cubique",
            Self::ExponentialRelax => "Exponentielle",
            Self::Logarithmic => "Logarithmique",
            Self::PowerRelax => "Puissance",
        }
    }

    pub fn from_id(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "constant" | "const" | "c" => Some(Self::Constant),
            "linear" | "lin" | "poly1" => Some(Self::Linear),
            "quadratic" | "quad" | "poly2" => Some(Self::Quadratic),
            "cubic" | "poly3" => Some(Self::Cubic),
            "exponential" | "exp" | "exponential_relax" => Some(Self::ExponentialRelax),
            "logarithmic" | "log" => Some(Self::Logarithmic),
            "power" | "power_relax" | "pow" => Some(Self::PowerRelax),
            _ => None,
        }
    }

    /// Number of free model parameters `k`.
    pub fn n_params(self) -> usize {
        match self {
            Self::Constant => 1,
            Self::Linear | Self::Logarithmic => 2,
            Self::Quadratic | Self::ExponentialRelax | Self::PowerRelax => 3,
            Self::Cubic => 4,
        }
    }
}

/// Criterion used to rank competing law families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LawSelectionCriterion {
    Aicc,
    Rmse,
    OutOfBlockRmse,
    R2,
}

impl LawSelectionCriterion {
    pub fn from_str_lossy(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "rmse" => Self::Rmse,
            "oob" | "out_of_block" | "out_of_block_rmse" => Self::OutOfBlockRmse,
            "r2" => Self::R2,
            _ => Self::Aicc,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Aicc => "aicc",
            Self::Rmse => "rmse",
            Self::OutOfBlockRmse => "out_of_block_rmse",
            Self::R2 => "r2",
        }
    }
}

/// Fitted law candidate with diagnostics and dense curve samples.
#[derive(Debug, Clone)]
pub struct LawCandidateFit {
    pub family: LawFamily,
    pub n_params: usize,
    pub coefficients: Vec<f64>,
    pub t0: f64,
    pub t_span: f64,
    pub formula_label: String,
    pub rmse: f64,
    pub mae: f64,
    pub r2: f64,
    pub aicc: f64,
    pub out_of_block_rmse: Option<f64>,
    pub predicted: Vec<f64>,
    pub dense_times: Vec<f64>,
    pub dense_predicted: Vec<f64>,
}

impl LawCandidateFit {
    /// Evaluate the fitted law on an arbitrary vector of times `t`.
    pub fn evaluate_at(&self, times: &[f64]) -> Vec<f64> {
        let span = self.t_span.max(1e-9);
        times
            .iter()
            .map(|&t| {
                let u = ((t - self.t0) / span).max(0.0);
                let val = match self.family {
                    LawFamily::Constant
                    | LawFamily::Linear
                    | LawFamily::Quadratic
                    | LawFamily::Cubic => polyval(&self.coefficients, u),
                    LawFamily::ExponentialRelax => {
                        // coefficients = [p_inf, delta_p, alpha]
                        let p_inf = self.coefficients.first().copied().unwrap_or(0.0);
                        let dp = self.coefficients.get(1).copied().unwrap_or(0.0);
                        let alpha = self.coefficients.get(2).copied().unwrap_or(1.0);
                        p_inf + dp * (-alpha * u).exp()
                    }
                    LawFamily::Logarithmic => {
                        // coefficients = [a, b] -> a + b * ln(1 + u)
                        let a = self.coefficients.first().copied().unwrap_or(0.0);
                        let b = self.coefficients.get(1).copied().unwrap_or(0.0);
                        a + b * (1.0 + u).ln()
                    }
                    LawFamily::PowerRelax => {
                        // coefficients = [a, b, gamma] -> a + b * (1 + u)^(-gamma)
                        let a = self.coefficients.first().copied().unwrap_or(0.0);
                        let b = self.coefficients.get(1).copied().unwrap_or(0.0);
                        let gamma = self.coefficients.get(2).copied().unwrap_or(1.0);
                        a + b * (1.0 + u).powf(-gamma)
                    }
                };
                if val.is_finite() {
                    val.clamp(0.0, 1.0)
                } else {
                    0.0
                }
            })
            .collect()
    }

    pub fn to_json(&self) -> JsonValue {
        let mut o = JsonValue::object();
        o.insert("family", JsonValue::String(self.family.id().to_string()));
        o.insert(
            "family_name",
            JsonValue::String(self.family.display_name().to_string()),
        );
        o.insert("n_params", JsonValue::Number(self.n_params as f64));
        o.insert("coefficients", vec_f64_json(&self.coefficients));
        o.insert(
            "formula_label",
            JsonValue::String(self.formula_label.clone()),
        );
        o.insert("rmse", JsonValue::Number(self.rmse));
        o.insert("mae", JsonValue::Number(self.mae));
        o.insert("r2", JsonValue::Number(self.r2));
        o.insert("aicc", JsonValue::Number(self.aicc));
        o.insert(
            "out_of_block_rmse",
            self.out_of_block_rmse
                .map(JsonValue::Number)
                .unwrap_or(JsonValue::Null),
        );
        o.insert("predicted", vec_f64_json(&self.predicted));
        o.insert("dense_times", vec_f64_json(&self.dense_times));
        o.insert("dense_predicted", vec_f64_json(&self.dense_predicted));
        o
    }
}

/// Complete multi-law comparison for one transition `p_{src -> dst}(t)`.
#[derive(Debug, Clone)]
pub struct TransitionLawComparison {
    pub src: usize,
    pub dst: usize,
    pub times: Vec<f64>,
    pub values: Vec<f64>,
    pub criterion: LawSelectionCriterion,
    pub best_family: LawFamily,
    pub candidates: Vec<LawCandidateFit>,
}

impl TransitionLawComparison {
    pub fn best_candidate(&self) -> Option<&LawCandidateFit> {
        self.candidates.iter().find(|c| c.family == self.best_family)
    }

    pub fn to_json(&self) -> JsonValue {
        let mut o = JsonValue::object();
        o.insert("src", JsonValue::Number(self.src as f64));
        o.insert("dst", JsonValue::Number(self.dst as f64));
        o.insert("times", vec_f64_json(&self.times));
        o.insert("values", vec_f64_json(&self.values));
        o.insert(
            "criterion",
            JsonValue::String(self.criterion.as_str().to_string()),
        );
        o.insert(
            "best_family",
            JsonValue::String(self.best_family.id().to_string()),
        );
        o.insert(
            "best_family_name",
            JsonValue::String(self.best_family.display_name().to_string()),
        );
        o.insert(
            "candidates",
            JsonValue::Array(self.candidates.iter().map(|c| c.to_json()).collect()),
        );
        o
    }
}

fn ols_2param(phi: &[f64], y: &[f64]) -> (f64, f64) {
    let n = phi.len().min(y.len());
    if n == 0 {
        return (0.0, 0.0);
    }
    let mean_p = phi[..n].iter().sum::<f64>() / n as f64;
    let mean_y = y[..n].iter().sum::<f64>() / n as f64;
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..n {
        let dp = phi[i] - mean_p;
        num += dp * (y[i] - mean_y);
        den += dp * dp;
    }
    if den.abs() < 1e-18 {
        (mean_y, 0.0)
    } else {
        let b = num / den;
        let a = mean_y - b * mean_p;
        (a, b)
    }
}

fn compute_metrics(y: &[f64], y_hat: &[f64], k: usize) -> (f64, f64, f64, f64) {
    let n = y.len().min(y_hat.len()).max(1);
    let mut sq_sum = 0.0;
    let mut abs_sum = 0.0;
    for i in 0..n {
        let err = y[i] - y_hat[i];
        sq_sum += err * err;
        abs_sum += err.abs();
    }
    let mse = sq_sum / n as f64;
    let rmse = mse.sqrt();
    let mae = abs_sum / n as f64;

    let y_mean = y.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let r2 = if ss_tot > 1e-12 {
        1.0 - sq_sum / ss_tot
    } else if sq_sum <= 1e-12 {
        1.0
    } else {
        0.0
    };

    // Small-sample corrected AICc. When n <= k + 1, add a strong parsimony penalty
    // so over-parameterized curves never spuriously win on 2-3 blocks.
    let nf = n as f64;
    let kf = k as f64;
    let aic_base = nf * (mse.max(1e-18)).ln() + 2.0 * kf;
    let correction = if n > k + 1 {
        (2.0 * kf * (kf + 1.0)) / (nf - kf - 1.0)
    } else {
        40.0 * (k + 2 - n) as f64
    };
    let aicc = aic_base + correction;
    (rmse, mae, r2, aicc)
}

fn fit_raw_family(
    t: &[f64],
    y: &[f64],
    t0: f64,
    t_span: f64,
    family: LawFamily,
    dense_times: &[f64],
) -> Result<LawCandidateFit> {
    let n = t.len();
    let span = t_span.max(1e-9);
    let u: Vec<f64> = t.iter().map(|&ti| ((ti - t0) / span).max(0.0)).collect();

    let (coefficients, formula_label) = match family {
        LawFamily::Constant => {
            let mean = y.iter().sum::<f64>() / n.max(1) as f64;
            (vec![mean], format!("p = {mean:.4}"))
        }
        LawFamily::Linear => {
            let deg = 1usize.min(n.saturating_sub(1));
            let c = polyfit(&u, y, deg)?;
            let c_full = if deg == 0 { vec![0.0, c[0]] } else { c };
            let lbl = format!("p = {:.4} {:+.4}·u", c_full[1], c_full[0]);
            (c_full, lbl)
        }
        LawFamily::Quadratic => {
            let deg = 2usize.min(n.saturating_sub(1));
            let mut c = polyfit(&u, y, deg)?;
            while c.len() < 3 {
                c.insert(0, 0.0);
            }
            let lbl = format!("p = {:.4} {:+.4}·u {:+.4}·u²", c[2], c[1], c[0]);
            (c, lbl)
        }
        LawFamily::Cubic => {
            let deg = 3usize.min(n.saturating_sub(1));
            let mut c = polyfit(&u, y, deg)?;
            while c.len() < 4 {
                c.insert(0, 0.0);
            }
            let lbl = format!(
                "p = {:.4} {:+.4}·u {:+.4}·u² {:+.4}·u³",
                c[3], c[2], c[1], c[0]
            );
            (c, lbl)
        }
        LawFamily::ExponentialRelax => {
            // Coarse log-grid + local refinement over alpha in [0.08, 12.0]
            let mut best_alpha = 1.5;
            let mut best_a = y.first().copied().unwrap_or(0.0);
            let mut best_b = 0.0;
            let mut best_sse = f64::INFINITY;

            let eval_alpha = |alpha: f64| -> (f64, f64, f64) {
                let phi: Vec<f64> = u.iter().map(|&ui| (-alpha * ui).exp()).collect();
                let (a, b) = ols_2param(&phi, y);
                let sse: f64 = phi
                    .iter()
                    .zip(y.iter())
                    .map(|(&p, &yi)| {
                        let r = (a + b * p) - yi;
                        r * r
                    })
                    .sum();
                (a, b, sse)
            };

            for step in 0..32 {
                let alpha = 0.08 * (12.0_f64 / 0.08).powf(step as f64 / 31.0);
                let (a, b, sse) = eval_alpha(alpha);
                if sse < best_sse {
                    best_sse = sse;
                    best_alpha = alpha;
                    best_a = a;
                    best_b = b;
                }
            }
            // Local refinement around best_alpha
            let mut lo = (best_alpha * 0.65).max(0.02);
            let mut hi = (best_alpha * 1.45).min(18.0);
            for _ in 0..10 {
                let m1 = lo + (hi - lo) / 3.0;
                let m2 = hi - (hi - lo) / 3.0;
                let (a1, b1, s1) = eval_alpha(m1);
                let (a2, b2, s2) = eval_alpha(m2);
                if s1 <= s2 {
                    hi = m2;
                    if s1 < best_sse {
                        best_sse = s1;
                        best_alpha = m1;
                        best_a = a1;
                        best_b = b1;
                    }
                } else {
                    lo = m1;
                    if s2 < best_sse {
                        best_sse = s2;
                        best_alpha = m2;
                        best_a = a2;
                        best_b = b2;
                    }
                }
            }
            let lbl = format!(
                "p = {best_a:.4} {best_b:+.4}·exp(-{best_alpha:.2}·u)"
            );
            (vec![best_a, best_b, best_alpha], lbl)
        }
        LawFamily::Logarithmic => {
            let phi: Vec<f64> = u.iter().map(|&ui| (1.0 + ui).ln()).collect();
            let (a, b) = ols_2param(&phi, y);
            let lbl = format!("p = {a:.4} {b:+.4}·ln(1+u)");
            (vec![a, b], lbl)
        }
        LawFamily::PowerRelax => {
            let mut best_gamma = 1.0;
            let mut best_a = y.first().copied().unwrap_or(0.0);
            let mut best_b = 0.0;
            let mut best_sse = f64::INFINITY;

            for g in [0.2, 0.35, 0.5, 0.75, 1.0, 1.4, 2.0, 2.8, 4.0] {
                let phi: Vec<f64> = u.iter().map(|&ui| (1.0 + ui).powf(-g)).collect();
                let (a, b) = ols_2param(&phi, y);
                let sse: f64 = phi
                    .iter()
                    .zip(y.iter())
                    .map(|(&p, &yi)| {
                        let r = (a + b * p) - yi;
                        r * r
                    })
                    .sum();
                if sse < best_sse {
                    best_sse = sse;
                    best_gamma = g;
                    best_a = a;
                    best_b = b;
                }
            }
            let lbl = format!("p = {best_a:.4} {best_b:+.4}·(1+u)^(-{best_gamma:.2})");
            (vec![best_a, best_b, best_gamma], lbl)
        }
    };

    let mut candidate = LawCandidateFit {
        family,
        n_params: family.n_params(),
        coefficients,
        t0,
        t_span: span,
        formula_label,
        rmse: 0.0,
        mae: 0.0,
        r2: 1.0,
        aicc: 0.0,
        out_of_block_rmse: None,
        predicted: Vec::new(),
        dense_times: dense_times.to_vec(),
        dense_predicted: Vec::new(),
    };
    candidate.predicted = candidate.evaluate_at(t);
    candidate.dense_predicted = candidate.evaluate_at(dense_times);
    let (rmse, mae, r2, aicc) = compute_metrics(y, &candidate.predicted, candidate.n_params);
    candidate.rmse = rmse;
    candidate.mae = mae;
    candidate.r2 = r2;
    candidate.aicc = aicc;
    Ok(candidate)
}

/// Fit one specific law family on `(times, values)` and evaluate out-of-block RMSE
/// when enough blocks are present (`n >= 3`).
pub fn fit_law_family(
    times: &[f64],
    values: &[f64],
    family: LawFamily,
    dense_times: &[f64],
) -> Result<LawCandidateFit> {
    if times.is_empty() || times.len() != values.len() {
        return Err(DemMcmError::ShapeMismatch(
            "times and values must be non-empty and equal length".to_string(),
        ));
    }
    let t0 = times.iter().copied().fold(f64::INFINITY, f64::min);
    let t1 = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = (t1 - t0).max(1.0);

    let mut fit = fit_raw_family(times, values, t0, span, family, dense_times)?;

    // Out-of-block temporal holdout validation: fit on initial blocks, predict held-out tail blocks
    let n = times.len();
    let n_train = if n >= 4 {
        (n - (n / 4).max(1)).max(2)
    } else if n == 3 && family.n_params() <= 2 {
        2
    } else {
        0
    };
    if n_train >= 2 && n_train < n && n_train >= family.n_params().min(2) {
        if let Ok(train_fit) = fit_raw_family(
            &times[..n_train],
            &values[..n_train],
            t0,
            span,
            family,
            dense_times,
        ) {
            let preds = train_fit.evaluate_at(&times[n_train..]);
            let mse_oob = preds
                .iter()
                .zip(values[n_train..].iter())
                .map(|(&p, &v)| (p - v).powi(2))
                .sum::<f64>()
                / preds.len().max(1) as f64;
            fit.out_of_block_rmse = Some(mse_oob.sqrt());
        }
    }

    Ok(fit)
}

/// Compare multiple mathematical law families on a transition curve `p_{src -> dst}(t)`.
pub fn compare_transition_laws(
    src: usize,
    dst: usize,
    times: &[f64],
    values: &[f64],
    families: &[LawFamily],
    criterion: LawSelectionCriterion,
) -> Result<TransitionLawComparison> {
    if times.is_empty() || times.len() != values.len() {
        return Err(DemMcmError::ShapeMismatch(
            "times and values must be non-empty and equal length".to_string(),
        ));
    }
    let active_families = if families.is_empty() {
        LawFamily::all()
    } else {
        families.to_vec()
    };

    let t0 = times.iter().copied().fold(f64::INFINITY, f64::min);
    let t1 = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let n_dense = if (t1 - t0).abs() > 1e-12 { 36 } else { 1 };
    let dense_times: Vec<f64> = if n_dense == 1 {
        vec![t0]
    } else {
        (0..n_dense)
            .map(|k| t0 + (t1 - t0) * (k as f64 / (n_dense - 1) as f64))
            .collect()
    };

    let mut candidates = Vec::with_capacity(active_families.len());
    for fam in active_families {
        // Skip Cubic when fewer than 4 points are available unless explicitly requested alone
        if fam == LawFamily::Cubic && times.len() < 4 && candidates.len() >= 2 {
            continue;
        }
        if let Ok(fit) = fit_law_family(times, values, fam, &dense_times) {
            candidates.push(fit);
        }
    }
    if candidates.is_empty() {
        candidates.push(fit_law_family(
            times,
            values,
            LawFamily::Constant,
            &dense_times,
        )?);
    }

    let best_family = select_best_family(&candidates, criterion);

    Ok(TransitionLawComparison {
        src,
        dst,
        times: times.to_vec(),
        values: values.to_vec(),
        criterion,
        best_family,
        candidates,
    })
}

fn select_best_family(
    candidates: &[LawCandidateFit],
    criterion: LawSelectionCriterion,
) -> LawFamily {
    let mut best = &candidates[0];
    for c in candidates.iter().skip(1) {
        let better = match criterion {
            LawSelectionCriterion::Aicc => c.aicc < best.aicc,
            LawSelectionCriterion::Rmse => c.rmse < best.rmse,
            LawSelectionCriterion::OutOfBlockRmse => {
                let co = c.out_of_block_rmse.unwrap_or(c.rmse + 1.0);
                let bo = best.out_of_block_rmse.unwrap_or(best.rmse + 1.0);
                co < bo
            }
            LawSelectionCriterion::R2 => c.r2 > best.r2,
        };
        if better {
            best = c;
        }
    }
    best.family
}
