//! Temporal & block-by-block evolution analysis of inhomogeneous transition matrix
//! components `p_{i->j}(t)` across `P_blocks`, matching
//! `postprocessing/postprocess_inhomogeneous.py::fig_matrix_components_evolution`.

use crate::gui::metrics_eval::vec_f64_json;
use crate::io::json::JsonValue;
use crate::linalg::Matrix;
use crate::metrics::fit_probability_evolution;
use crate::research::laws::{
    compare_transition_laws, LawCandidateFit, LawFamily, LawSelectionCriterion,
};

/// Fitted polynomial + multi-family evolution laws for one significant transition `p_{i->j}(t)`
/// across inhomogeneous NLT blocks.
#[derive(Debug, Clone)]
pub struct TransitionLawItem {
    pub src: usize,
    pub dst: usize,
    pub is_diagonal: bool,
    pub block_times: Vec<f64>,
    pub values: Vec<f64>,
    pub predicted: Vec<f64>,
    pub dense_times: Vec<f64>,
    pub dense_predicted: Vec<f64>,
    pub degree: usize,
    pub rmse: f64,
    pub r2: f64,
    pub max_prob: f64,
    pub mean_prob: f64,
    pub range_prob: f64,
    pub law_label: String,
    pub best_law_family: String,
    pub law_candidates: Vec<LawCandidateFit>,
}

impl TransitionLawItem {
    pub fn to_json(&self) -> JsonValue {
        let mut o = JsonValue::object();
        o.insert("src", JsonValue::Number(self.src as f64));
        o.insert("dst", JsonValue::Number(self.dst as f64));
        o.insert("is_diagonal", JsonValue::Bool(self.is_diagonal));
        o.insert("block_times", vec_f64_json(&self.block_times));
        o.insert("values", vec_f64_json(&self.values));
        o.insert("predicted", vec_f64_json(&self.predicted));
        o.insert("dense_times", vec_f64_json(&self.dense_times));
        o.insert("dense_predicted", vec_f64_json(&self.dense_predicted));
        o.insert("degree", JsonValue::Number(self.degree as f64));
        o.insert("rmse", JsonValue::Number(self.rmse));
        o.insert("r2", JsonValue::Number(self.r2));
        o.insert("max_prob", JsonValue::Number(self.max_prob));
        o.insert("mean_prob", JsonValue::Number(self.mean_prob));
        o.insert("range_prob", JsonValue::Number(self.range_prob));
        o.insert("law_label", JsonValue::String(self.law_label.clone()));
        o.insert(
            "best_law_family",
            JsonValue::String(self.best_law_family.clone()),
        );
        o.insert(
            "law_candidates",
            JsonValue::Array(self.law_candidates.iter().map(|c| c.to_json()).collect()),
        );
        o
    }
}

/// Complete inhomogeneous transition component dataset for one species (`small` or `large`).
#[derive(Debug, Clone)]
pub struct MatrixEvolutionData {
    pub species: String,
    pub n_states: usize,
    pub threshold: f64,
    pub block_times: Vec<f64>,
    pub block_labels: Vec<String>,
    pub global_vmax: f64,
    pub components: Vec<TransitionLawItem>,
    pub delta_norms_l1: Vec<f64>,
    pub delta_max_abs: Vec<f64>,
}

impl MatrixEvolutionData {
    pub fn to_json(&self) -> JsonValue {
        let mut o = JsonValue::object();
        o.insert("species", JsonValue::String(self.species.clone()));
        o.insert("n_states", JsonValue::Number(self.n_states as f64));
        o.insert("threshold", JsonValue::Number(self.threshold));
        o.insert("block_times", vec_f64_json(&self.block_times));
        o.insert(
            "block_labels",
            JsonValue::Array(
                self.block_labels
                    .iter()
                    .map(|s| JsonValue::String(s.clone()))
                    .collect(),
            ),
        );
        o.insert("global_vmax", JsonValue::Number(self.global_vmax));
        o.insert(
            "components",
            JsonValue::Array(self.components.iter().map(|c| c.to_json()).collect()),
        );
        o.insert("delta_norms_l1", vec_f64_json(&self.delta_norms_l1));
        o.insert("delta_max_abs", vec_f64_json(&self.delta_max_abs));
        o
    }
}

/// Extract all significant `p_{i->j}(t)` trajectories across `blocks` where
/// `max_b P^{(b)}_{i,j} > threshold` (matching `postprocess_inhomogeneous.py`),
/// fit polynomial laws up to `max_fit_degree`, and compute shared `global_vmax`.
pub fn compute_matrix_evolution(
    blocks: &[Matrix],
    block_times: &[f64],
    block_labels: &[String],
    species: &str,
    threshold: f64,
    max_fit_degree: usize,
    max_components: usize,
) -> MatrixEvolutionData {
    if blocks.is_empty() || block_times.is_empty() {
        return MatrixEvolutionData {
            species: species.to_string(),
            n_states: 0,
            threshold,
            block_times: block_times.to_vec(),
            block_labels: block_labels.to_vec(),
            global_vmax: 1.0,
            components: Vec::new(),
            delta_norms_l1: Vec::new(),
            delta_max_abs: Vec::new(),
        };
    }

    let n_states = blocks[0].rows;
    let n_blocks = blocks.len().min(block_times.len());
    let x_vals = &block_times[..n_blocks];

    // Dense time grid (28 points) for smooth curve rendering of polynomial laws
    let x_min = x_vals.iter().copied().fold(f64::INFINITY, f64::min);
    let x_max = x_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let n_dense = if (x_max - x_min).abs() > 1e-12 { 28 } else { 1 };
    let dense_times: Vec<f64> = if n_dense == 1 {
        vec![x_min]
    } else {
        (0..n_dense)
            .map(|k| x_min + (x_max - x_min) * (k as f64 / (n_dense - 1) as f64))
            .collect()
    };

    // Collect every (i, j) where at least one block exceeds `threshold`
    let mut candidates: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n_states {
        for j in 0..n_states {
            let mut mx = 0.0f64;
            for b in &blocks[..n_blocks] {
                let v = b.get(i, j);
                if v.is_finite() && v > mx {
                    mx = v;
                }
            }
            if mx > threshold {
                candidates.push((i, j, mx));
            }
        }
    }
    candidates.sort_by(|a, b| {
        b.2.total_cmp(&a.2)
            .then_with(|| a.0.cmp(&b.0))
            .then_with(|| a.1.cmp(&b.1))
    });
    candidates.truncate(max_components);

    let mut global_vmax = 0.0f64;
    let mut components = Vec::with_capacity(candidates.len());

    for (comp_idx, (src, dst, max_prob)) in candidates.into_iter().enumerate() {
        let values: Vec<f64> = blocks[..n_blocks]
            .iter()
            .map(|b| {
                let v = b.get(src, dst);
                if v.is_finite() {
                    v
                } else {
                    0.0
                }
            })
            .collect();
        let min_prob = values.iter().copied().fold(f64::INFINITY, f64::min);
        let mean_prob = values.iter().sum::<f64>() / values.len().max(1) as f64;
        let range_prob = (max_prob - min_prob).max(0.0);

        global_vmax = global_vmax.max(max_prob);

        let (predicted, dense_predicted, degree, rmse, r2, law_label) = if n_blocks >= 2 {
            match fit_probability_evolution(x_vals, &values, max_fit_degree) {
                Ok(fit) => {
                    let pred = fit.predict(x_vals);
                    let dense_pred = fit.predict(&dense_times);
                    for &pv in &pred {
                        if pv.is_finite() {
                            global_vmax = global_vmax.max(pv);
                        }
                    }
                    (
                        pred,
                        dense_pred,
                        fit.degree,
                        fit.rmse,
                        fit.r2,
                        fit.law_label,
                    )
                }
                Err(_) => (
                    values.clone(),
                    vec![mean_prob; dense_times.len()],
                    0,
                    0.0,
                    1.0,
                    format!("p = {mean_prob:.4}"),
                ),
            }
        } else {
            let v0 = values[0];
            (
                values.clone(),
                vec![v0; dense_times.len()],
                0,
                0.0,
                1.0,
                format!("p = {v0:.4}"),
            )
        };

        let (best_law_family, law_candidates) = if n_blocks >= 2 && comp_idx < 40 {
            match compare_transition_laws(
                src,
                dst,
                x_vals,
                &values,
                &LawFamily::all(),
                LawSelectionCriterion::Aicc,
            ) {
                Ok(cmp) => (cmp.best_family.id().to_string(), cmp.candidates),
                Err(_) => ("constant".to_string(), Vec::new()),
            }
        } else {
            (format!("poly{degree}"), Vec::new())
        };

        components.push(TransitionLawItem {
            src,
            dst,
            is_diagonal: src == dst,
            block_times: x_vals.to_vec(),
            values,
            predicted,
            dense_times: dense_times.clone(),
            dense_predicted,
            degree,
            rmse,
            r2,
            max_prob,
            mean_prob,
            range_prob,
            law_label,
            best_law_family,
            law_candidates,
        });
    }

    // Consecutive inter-block differences: Delta_k = P_{k+1} - P_k
    let mut delta_norms_l1 = Vec::new();
    let mut delta_max_abs = Vec::new();
    for k in 0..n_blocks.saturating_sub(1) {
        let a = &blocks[k];
        let b = &blocks[k + 1];
        let norm_a: f64 = a
            .data
            .iter()
            .filter(|x| x.is_finite())
            .map(|x| x.abs())
            .sum();
        let mut norm_diff = 0.0f64;
        let mut max_d = 0.0f64;
        for (&va, &vb) in a.data.iter().zip(b.data.iter()) {
            let da = if va.is_finite() { va } else { 0.0 };
            let db = if vb.is_finite() { vb } else { 0.0 };
            let diff = (db - da).abs();
            norm_diff += diff;
            max_d = max_d.max(diff);
        }
        delta_norms_l1.push(if norm_a > 1e-12 {
            norm_diff / norm_a
        } else {
            0.0
        });
        delta_max_abs.push(max_d);
    }

    MatrixEvolutionData {
        species: species.to_string(),
        n_states,
        threshold,
        block_times: x_vals.to_vec(),
        block_labels: block_labels.iter().take(n_blocks).cloned().collect(),
        global_vmax: (global_vmax * 1.15).max(0.05),
        components,
        delta_norms_l1,
        delta_max_abs,
    }
}
