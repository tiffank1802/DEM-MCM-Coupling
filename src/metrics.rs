//! Physics of the DEM/MCM mixing — metrics, stochasticity standardisation,
//! propagation, probability law fitting, and physical validation.
//!
//! Mirrors `postprocessing/metrics.py` as the single source of truth for
//! row-stochastic Markov propagation (`phi_next = phi @ P`) and segregation
//! indices.

use crate::config::ExperimentData;
use crate::error::{DemMcmError, Result};
use crate::linalg::{polyfit, polyval, real_eigvals_abs, stationary_distribution_left, Matrix};

// ============================================================================
// CONVENTION DETECTION & STANDARDISATION
// ============================================================================

/// Stochasticity convention of a transition matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Convention {
    Row,
    Column,
    None,
}

impl Convention {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Row => "row",
            Self::Column => "column",
            Self::None => "none",
        }
    }
}

fn is_close(a: f64, b: f64, rtol: f64, atol: f64) -> bool {
    (a - b).abs() <= atol + rtol * b.abs()
}

/// Detect whether `P` is row-stochastic, column-stochastic, or unnormalised.
///
/// Mirrors `postprocessing.metrics.detect_convention`. Note that any row
/// containing `NaN` propagates `NaN` in column sums just as NumPy `sum(axis=0)`
/// does, ensuring unvisited `NaN` rows in row-stochastic matrices stay
/// classified as [`Convention::Row`].
pub fn detect_convention(p: &Matrix, rtol: f64) -> Convention {
    let n = p.rows;
    if n == 0 || p.cols != n {
        return Convention::None;
    }

    let mut row_sums = vec![0.0f64; n];
    let mut col_sums = vec![0.0f64; n];
    for (r, row_sum) in row_sums.iter_mut().enumerate() {
        let mut rs = 0.0f64;
        for (c, col_sum) in col_sums.iter_mut().enumerate() {
            let v = p.get(r, c);
            rs += v;
            *col_sum += v;
        }
        *row_sum = rs;
    }

    let active_rows: Vec<f64> = row_sums.iter().copied().filter(|&s| s > 0.0).collect();
    let active_cols: Vec<f64> = col_sums.iter().copied().filter(|&s| s > 0.0).collect();

    let rows_ok =
        !active_rows.is_empty() && active_rows.iter().all(|&s| is_close(s, 1.0, rtol, 1e-8));
    let cols_ok =
        !active_cols.is_empty() && active_cols.iter().all(|&s| is_close(s, 1.0, rtol, 1e-8));

    if rows_ok && !cols_ok {
        Convention::Row
    } else if cols_ok && !rows_ok {
        Convention::Column
    } else if rows_ok && cols_ok {
        Convention::Row
    } else {
        Convention::None
    }
}

/// Return `(P_row, was_transposed)` in row-stochastic convention.
///
/// Legacy column-stochastic matrices are transposed; non-stochastic matrices
/// are renormalised row-wise (zero / `NaN` rows remain unchanged).
pub fn standardize_transition_matrix(p: &Matrix) -> (Matrix, bool) {
    match detect_convention(p, 1e-3) {
        Convention::Column => (p.transpose(), true),
        Convention::None => {
            let n = p.rows;
            let mut out = p.clone();
            for r in 0..n {
                let s: f64 = out.row(r).iter().sum();
                if s > 0.0 {
                    for c in 0..n {
                        out.data[r * n + c] /= s;
                    }
                }
            }
            (out, false)
        }
        Convention::Row => (p.clone(), false),
    }
}

/// Standardise, deactivate unvisited states (`row_sum < threshold`), and
/// renormalise active rows to sum strictly to 1.0.
pub fn clean_transition_matrix(p: &Matrix, threshold: f64) -> (Matrix, Vec<bool>) {
    let (mut p_clean, _) = standardize_transition_matrix(p);
    let n = p_clean.rows;
    let mut activated = vec![false; n];
    for (r, act) in activated.iter_mut().enumerate() {
        let s: f64 = p_clean.row(r).iter().sum();
        if s >= threshold {
            *act = true;
            for c in 0..n {
                p_clean.data[r * n + c] /= s;
            }
        } else {
            *act = false;
            for c in 0..n {
                p_clean.data[r * n + c] = 0.0;
            }
        }
    }
    (p_clean, activated)
}

/// Replace any `NaN` rows (unvisited states) with `0.0` before matrix-vector multiplication.
fn sanitize_unvisited_rows(p: &Matrix) -> Matrix {
    let mut out = p.clone();
    for r in 0..out.rows {
        let has_nan = out.row(r).iter().any(|v| !v.is_finite());
        if has_nan {
            for c in 0..out.cols {
                if !out.get(r, c).is_finite() {
                    out.set(r, c, 0.0);
                }
            }
        }
    }
    out
}

// ============================================================================
// PROPAGATION
// ============================================================================

/// Homogeneous Markov propagation `S_next = S @ P`.
///
/// Mirrors `postprocessing.metrics.propagate_markov`.
pub fn propagate_markov(
    s0: &[f64],
    p: &Matrix,
    times: &[i64],
    start_idx: i64,
    tau: usize,
    activated: &[bool],
) -> Result<(Matrix, Vec<i64>)> {
    let (p_std_raw, _) = standardize_transition_matrix(p);
    let p_std = sanitize_unvisited_rows(&p_std_raw);
    let n_states = p_std.rows;
    if s0.len() != n_states || activated.len() != n_states {
        return Err(DemMcmError::ShapeMismatch(
            "S0, activated, and P dimensions must match".to_string(),
        ));
    }
    if tau == 0 {
        return Err(DemMcmError::InvalidArgument("tau must be >= 1".to_string()));
    }

    let row_start = times.partition_point(|&t| t < start_idx);
    let times_full = &times[row_start..];
    let mut times_markov = Vec::new();
    for idx in (0..times_full.len()).step_by(tau) {
        times_markov.push(times_full[idx]);
    }

    let mut s = s0.to_vec();
    for (i, &act) in activated.iter().enumerate() {
        if !act {
            s[i] = 0.0;
        }
    }
    let n_steps = times_markov.len();
    let mut traj = Matrix::zeros(n_steps, n_states);
    if n_steps > 0 {
        traj.set_row(0, &s);
    }
    for t in 1..n_steps {
        s = p_std.vec_mul(&s)?;
        traj.set_row(t, &s);
    }
    Ok((traj, times_markov))
}

/// Inhomogeneous Markov propagation using time-varying `P_blocks[k]`.
///
/// Mirrors `postprocessing.metrics.propagate_markov_inhomogeneous`.
#[allow(clippy::too_many_arguments)]
pub fn propagate_markov_inhomogeneous(
    s0: &[f64],
    p_blocks: &[Matrix],
    times: &[i64],
    start_idx: i64,
    tau: usize,
    activated: &[bool],
    step: Option<usize>,
    nlt: Option<usize>,
) -> Result<(Matrix, Vec<i64>)> {
    if p_blocks.is_empty() {
        return Err(DemMcmError::InvalidArgument(
            "P_blocks must not be empty".to_string(),
        ));
    }
    let n_states = p_blocks[0].rows;
    if s0.len() != n_states || activated.len() != n_states {
        return Err(DemMcmError::ShapeMismatch(
            "S0, activated, and P_blocks dimensions must match".to_string(),
        ));
    }
    if tau == 0 {
        return Err(DemMcmError::InvalidArgument("tau must be >= 1".to_string()));
    }

    let p_std: Vec<Matrix> = p_blocks
        .iter()
        .map(|b| sanitize_unvisited_rows(&standardize_transition_matrix(b).0))
        .collect();

    let row_start = times.partition_point(|&t| t < start_idx);
    let times_full = &times[row_start..];
    let mut times_markov = Vec::new();
    for idx in (0..times_full.len()).step_by(tau) {
        times_markov.push(times_full[idx]);
    }
    let n_out = times_markov.len();
    let n_steps = n_out.saturating_sub(1);
    let n_blocks = p_std.len();

    let mut s = s0.to_vec();
    for (i, &act) in activated.iter().enumerate() {
        if !act {
            s[i] = 0.0;
        }
    }
    let mut traj = Matrix::zeros(n_out, n_states);
    if n_out > 0 {
        traj.set_row(0, &s);
    }

    if let (Some(step_val), Some(_nlt_val)) = (step, nlt) {
        let block_duration = (step_val + tau) as i64;
        for t in 1..n_out {
            let time_curr = times_markov[t - 1];
            let block_idx = ((time_curr - start_idx) / block_duration)
                .max(0)
                .min((n_blocks - 1) as i64) as usize;
            s = p_std[block_idx].vec_mul(&s)?;
            traj.set_row(t, &s);
        }
    } else {
        let block_size = n_steps.checked_div(n_blocks).unwrap_or(1).max(1);
        for t in 1..n_out {
            let block_idx = ((t - 1) / block_size).min(n_blocks - 1);
            s = p_std[block_idx].vec_mul(&s)?;
            traj.set_row(t, &s);
        }
    }

    Ok((traj, times_markov))
}

// ============================================================================
// SEGREGATION METRICS
// ============================================================================

/// Global RSD of particle distribution per timestep over `activated` cells.
pub fn rsd_from_s(s: &Matrix, activated: &[bool]) -> Vec<f64> {
    let active_cols: Vec<usize> = activated
        .iter()
        .enumerate()
        .filter_map(|(i, &a)| if a { Some(i) } else { None })
        .collect();
    if active_cols.is_empty() {
        return vec![0.0; s.rows];
    }
    let n_a = active_cols.len() as f64;
    let mut out = Vec::with_capacity(s.rows);
    for r in 0..s.rows {
        let mean = active_cols.iter().map(|&c| s.get(r, c)).sum::<f64>() / n_a;
        if mean > 0.0 {
            let var = active_cols
                .iter()
                .map(|&c| {
                    let d = s.get(r, c) - mean;
                    d * d
                })
                .sum::<f64>()
                / n_a;
            out.push(var.sqrt() / mean);
        } else {
            out.push(0.0);
        }
    }
    out
}

/// Local concentration `C(t, i) = S_small / (S_small + S_large)` in `[0, 1]`.
pub fn concentration_from_s(s_small: &Matrix, s_large: &Matrix) -> Matrix {
    let mut c = Matrix::zeros(s_small.rows, s_small.cols);
    for r in 0..s_small.rows {
        for col in 0..s_small.cols {
            let sm = s_small.get(r, col);
            let lg = s_large.get(r, col);
            let tot = sm + lg;
            if tot > 0.0 {
                c.set(r, col, sm / tot);
            }
        }
    }
    c
}

/// Concentration RSD across mutually active cells (`act_s & act_l`).
pub fn rsd_concentration(
    s_small: &Matrix,
    s_large: &Matrix,
    act_s: &[bool],
    act_l: &[bool],
) -> Vec<f64> {
    let active_cols: Vec<usize> = act_s
        .iter()
        .zip(act_l.iter())
        .enumerate()
        .filter_map(|(i, (&a, &b))| if a && b { Some(i) } else { None })
        .collect();
    if active_cols.is_empty() {
        return vec![0.0; s_small.rows];
    }
    let n_a = active_cols.len() as f64;
    let mut out = Vec::with_capacity(s_small.rows);
    for r in 0..s_small.rows {
        let mut c_vals = Vec::with_capacity(active_cols.len());
        for &col in &active_cols {
            let sm = s_small.get(r, col);
            let lg = s_large.get(r, col);
            let tot = sm + lg;
            c_vals.push(if tot > 0.0 { sm / tot } else { 0.0 });
        }
        let mean = c_vals.iter().sum::<f64>() / n_a;
        if mean > 0.0 {
            let var = c_vals
                .iter()
                .map(|&v| {
                    let d = v - mean;
                    d * d
                })
                .sum::<f64>()
                / n_a;
            out.push(var.sqrt() / mean);
        } else {
            out.push(0.0);
        }
    }
    out
}

/// Shannon entropy `H(t) = -sum p_i log(p_i)` over activated cells.
pub fn entropy_from_s(s: &Matrix, activated: &[bool]) -> Vec<f64> {
    let active_cols: Vec<usize> = activated
        .iter()
        .enumerate()
        .filter_map(|(i, &a)| if a { Some(i) } else { None })
        .collect();
    let mut out = Vec::with_capacity(s.rows);
    for r in 0..s.rows {
        let total: f64 = active_cols.iter().map(|&c| s.get(r, c)).sum();
        if total <= 0.0 {
            out.push(0.0);
            continue;
        }
        let mut h = 0.0f64;
        for &c in &active_cols {
            let p = s.get(r, c) / total;
            if p > 0.0 {
                h -= p * p.ln();
            }
        }
        out.push(h);
    }
    out
}

/// Binary mixing entropy over mutually active cells.
pub fn entropy_concentration(
    s_small: &Matrix,
    s_large: &Matrix,
    act_s: &[bool],
    act_l: &[bool],
    normalized: bool,
) -> Vec<f64> {
    let active_cols: Vec<usize> = act_s
        .iter()
        .zip(act_l.iter())
        .enumerate()
        .filter_map(|(i, (&a, &b))| if a && b { Some(i) } else { None })
        .collect();
    let n_active = active_cols.len();
    let mut out = Vec::with_capacity(s_small.rows);
    for r in 0..s_small.rows {
        let mut h = 0.0f64;
        for &col in &active_cols {
            let sm = s_small.get(r, col);
            let lg = s_large.get(r, col);
            let tot = sm + lg;
            if tot > 0.0 {
                let cv = sm / tot;
                if cv > 0.0 && cv < 1.0 {
                    h -= cv * cv.ln() + (1.0 - cv) * (1.0 - cv).ln();
                }
            }
        }
        if normalized && n_active > 0 {
            h /= (n_active as f64) * 2.0f64.ln();
        }
        out.push(h);
    }
    out
}

/// Danckwerts intensity of segregation `I(t) = Var(C) / (C_bar * (1 - C_bar))`.
pub fn intensity_of_segregation(
    s_small: &Matrix,
    s_large: &Matrix,
    act_s: &[bool],
    act_l: &[bool],
) -> Vec<f64> {
    let active_cols: Vec<usize> = act_s
        .iter()
        .zip(act_l.iter())
        .enumerate()
        .filter_map(|(i, (&a, &b))| if a && b { Some(i) } else { None })
        .collect();
    if active_cols.is_empty() {
        return vec![0.0; s_small.rows];
    }
    let n_a = active_cols.len() as f64;
    let mut out = Vec::with_capacity(s_small.rows);
    for r in 0..s_small.rows {
        let mut c_vals = Vec::with_capacity(active_cols.len());
        for &col in &active_cols {
            let sm = s_small.get(r, col);
            let lg = s_large.get(r, col);
            let tot = sm + lg;
            c_vals.push(if tot > 0.0 { sm / tot } else { 0.0 });
        }
        let mean = c_vals.iter().sum::<f64>() / n_a;
        let denom = mean * (1.0 - mean);
        if denom > 0.0 {
            let var = c_vals
                .iter()
                .map(|&v| {
                    let d = v - mean;
                    d * d
                })
                .sum::<f64>()
                / n_a;
            out.push(var / denom);
        } else {
            out.push(0.0);
        }
    }
    out
}

/// Mixing times (`t50`, `t90`, ...) for an RSD decay curve.
pub fn mixing_times(rsd: &[f64], times: &[f64], fractions: &[f64]) -> Vec<(f64, Option<f64>)> {
    let rsd_0 = if !rsd.is_empty() && rsd[0] > 0.0 {
        rsd[0]
    } else {
        1.0
    };
    fractions
        .iter()
        .map(|&f| {
            let hit = rsd
                .iter()
                .enumerate()
                .find(|(_, &v)| v < f * rsd_0)
                .map(|(i, _)| times[i]);
            (f, hit)
        })
        .collect()
}

/// Stationary distribution `pi` solving `pi P = pi`.
pub fn stationary_distribution(p: &Matrix) -> Result<Vec<f64>> {
    let (p_std_raw, _) = standardize_transition_matrix(p);
    let p_std = sanitize_unvisited_rows(&p_std_raw);
    stationary_distribution_left(&p_std)
}

/// Row-normalise a state matrix `S` by total particles per timestep.
pub fn normalize_by_particle_count(s: &Matrix) -> Matrix {
    let mut out = Matrix::zeros(s.rows, s.cols);
    for r in 0..s.rows {
        let tot: f64 = s.row(r).iter().sum();
        if tot > 0.0 {
            for c in 0..s.cols {
                out.set(r, c, s.get(r, c) / tot);
            }
        }
    }
    out
}

// ============================================================================
// PROBABILITY EVOLUTION POLYNOMIAL FITTING
// ============================================================================

#[derive(Debug, Clone)]
pub struct ProbabilityLawFit {
    pub degree: usize,
    pub coefficients: Vec<f64>, // highest power first
    pub rmse: f64,
    pub r2: f64,
    pub law_label: String,
}

impl ProbabilityLawFit {
    pub fn predict(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .map(|&xi| polyval(&self.coefficients, xi))
            .collect()
    }
}

fn polynomial_law_label(coefficients: &[f64], degree: usize) -> String {
    let mut terms: Vec<String> = Vec::new();
    for (idx, &coef) in coefficients.iter().enumerate() {
        let k = degree - idx;
        if coef.abs() < 1e-12 {
            continue;
        }
        let sign = if coef > 0.0 && !terms.is_empty() {
            "+"
        } else {
            ""
        };
        if k == 0 {
            terms.push(format!("{sign}{coef:.4}"));
        } else if k == 1 {
            terms.push(format!("{sign}{coef:.4} t"));
        } else {
            terms.push(format!("{sign}{coef:.4} t^{k}"));
        }
    }
    if terms.is_empty() {
        "p = 0".to_string()
    } else {
        format!("p = {}", terms.join(" "))
    }
}

/// Fit the lowest-degree polynomial up to `max_degree` whose RMSE stays within
/// 5% of the minimum RMSE.
pub fn fit_probability_evolution(
    x: &[f64],
    y: &[f64],
    max_degree: usize,
) -> Result<ProbabilityLawFit> {
    if x.is_empty() || x.len() != y.len() {
        return Err(DemMcmError::ShapeMismatch(
            "x and y must be non-empty and share the same length".to_string(),
        ));
    }
    let max_d = max_degree.min(x.len() - 1);
    let mut fits: Vec<(usize, Vec<f64>, f64)> = Vec::new();
    let mut best_rmse = f64::INFINITY;

    for d in 0..=max_d {
        let coef = polyfit(x, y, d)?;
        let y_hat: Vec<f64> = x.iter().map(|&xi| polyval(&coef, xi)).collect();
        let mse = y
            .iter()
            .zip(y_hat.iter())
            .map(|(&yi, &yh)| (yi - yh).powi(2))
            .sum::<f64>()
            / (y.len() as f64);
        let rmse = mse.sqrt();
        best_rmse = best_rmse.min(rmse);
        fits.push((d, coef, rmse));
    }

    let (chosen_degree, coefficients, rmse) = fits
        .into_iter()
        .find(|(_, _, r)| *r <= best_rmse * 1.05 + 1e-12)
        .unwrap();

    let y_hat: Vec<f64> = x.iter().map(|&xi| polyval(&coefficients, xi)).collect();
    let y_mean = y.iter().sum::<f64>() / (y.len() as f64);
    let ss_res: f64 = y
        .iter()
        .zip(y_hat.iter())
        .map(|(&yi, &yh)| (yi - yh).powi(2))
        .sum();
    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let r2 = if ss_tot > 1e-12 {
        1.0 - ss_res / ss_tot
    } else if ss_res <= 1e-12 {
        1.0
    } else {
        0.0
    };

    Ok(ProbabilityLawFit {
        degree: chosen_degree,
        law_label: polynomial_law_label(&coefficients, chosen_degree),
        coefficients,
        rmse,
        r2,
    })
}

// ============================================================================
// PHYSICAL VALIDATION
// ============================================================================

#[derive(Debug, Clone)]
pub struct ValidationCheck {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub folder: String,
    pub checks: Vec<ValidationCheck>,
    pub passed: bool,
}

impl ValidationReport {
    pub fn new(folder: impl Into<String>) -> Self {
        Self {
            folder: folder.into(),
            checks: Vec::new(),
            passed: false,
        }
    }

    pub fn add(&mut self, name: impl Into<String>, passed: bool, detail: impl Into<String>) {
        self.checks.push(ValidationCheck {
            name: name.into(),
            passed,
            detail: detail.into(),
        });
    }
}

impl std::fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Validation of '{}'", self.folder)?;
        for c in &self.checks {
            let mark = if c.passed { "PASS" } else { "FAIL" };
            writeln!(f, "  [{mark}] {} — {}", c.name, c.detail)?;
        }
        write!(
            f,
            "  -> {}",
            if self.passed {
                "ALL CHECKS PASSED"
            } else {
                "SOME CHECKS FAILED"
            }
        )
    }
}

/// Run physical coherence validation checks on a loaded [`ExperimentData`].
pub fn validate_experiment(folder_name: &str, exp: &ExperimentData, rtol: f64) -> ValidationReport {
    let mut report = ValidationReport::new(folder_name);
    if exp.species.is_empty() {
        report.add("species present", false, "no per-species data found");
        report.passed = false;
        return report;
    }
    let mut sp_keys: Vec<String> = exp.species.keys().cloned().collect();
    sp_keys.sort();
    report.add(
        "species present",
        true,
        format!("{} species: {}", sp_keys.len(), sp_keys.join(", ")),
    );

    for sp in &sp_keys {
        let data = &exp.species[sp];
        let p_raw = data
            .p
            .as_ref()
            .or_else(|| data.p_blocks.as_ref().and_then(|b| b.first()));
        let Some(p_raw) = p_raw else {
            report.add(
                format!("{sp}: matrix present"),
                false,
                "no P or P_blocks[0]",
            );
            continue;
        };

        // 1. Non-negativity
        let negatives = p_raw.data.iter().filter(|&&v| v < 0.0).count();
        report.add(
            format!("{sp}: non-negative entries"),
            negatives == 0,
            format!("{negatives} negative entries / {}", p_raw.data.len()),
        );

        // 2. Stochasticity after standardisation
        let (p_std_raw, transposed) = standardize_transition_matrix(p_raw);
        if transposed {
            report.add(
                format!("{sp}: convention"),
                true,
                "legacy column-stochastic matrix auto-transposed",
            );
        }
        let n_states = p_std_raw.rows;
        let row_sums: Vec<f64> = (0..n_states)
            .map(|r| p_std_raw.row(r).iter().sum::<f64>())
            .collect();
        let active: Vec<bool> = row_sums.iter().map(|&s| s > 0.0).collect();
        let max_dev = row_sums
            .iter()
            .zip(active.iter())
            .filter_map(|(&s, &a)| if a { Some((s - 1.0).abs()) } else { None })
            .fold(0.0f64, f64::max);
        let n_active = active.iter().filter(|&&a| a).count();
        report.add(
            format!("{sp}: row-stochastic after standardisation"),
            n_active > 0 && max_dev <= rtol,
            format!("max |row sum - 1| = {max_dev:.2e} over {n_active} visited rows"),
        );

        // 3. Diagonal dominance
        let p_std = sanitize_unvisited_rows(&p_std_raw);
        let diag_mean = if n_states > 0 {
            (0..n_states).map(|i| p_std.get(i, i)).sum::<f64>() / (n_states as f64)
        } else {
            0.0
        };
        report.add(
            format!("{sp}: diagonal dominance"),
            diag_mean >= 0.1,
            format!("mean(P(i -> i)) = {diag_mean:.3}"),
        );

        // 4. Mass conservation over 50 steps
        let s0: Vec<f64> = active.iter().map(|&a| if a { 1.0 } else { 0.0 }).collect();
        let s0_sum: f64 = s0.iter().sum();
        let mut s = s0;
        let mut conserved = true;
        for _ in 0..50 {
            match p_std.vec_mul(&s) {
                Ok(next) => {
                    s = next;
                    let sum: f64 = s.iter().sum();
                    if !is_close(sum, s0_sum, 1e-9, 1e-12) {
                        conserved = false;
                        break;
                    }
                }
                Err(_) => {
                    conserved = false;
                    break;
                }
            }
        }
        let final_sum: f64 = s.iter().sum();
        report.add(
            format!("{sp}: mass conservation over 50 steps"),
            conserved,
            format!("total = {s0_sum:.1} -> {final_sum:.3}"),
        );

        // 5. Stationarity
        let eigs = real_eigvals_abs(&p_std.transpose()).unwrap_or_default();
        let dominant = eigs.iter().map(|&v| v.abs()).fold(0.0f64, f64::max);
        let pi_ok = match stationary_distribution(&p_std) {
            Ok(pi) => is_close(dominant, 1.0, 1e-6, 1e-6) && pi.iter().all(|&x| x >= 0.0),
            Err(_) => false,
        };
        report.add(
            format!("{sp}: stationarity"),
            pi_ok,
            format!("|lambda|max = {dominant:.6}"),
        );
    }

    // 6. Cross-species concentration RSD bounds
    if sp_keys.len() >= 2 {
        let sa = &exp.species[&sp_keys[0]].s_matrix;
        let sb = &exp.species[&sp_keys[1]].s_matrix;
        if sa.rows > 0 && sa.cols > 0 && sa.shape() == sb.shape() {
            let act = vec![true; sa.cols];
            let rsd = rsd_concentration(sa, sb, &act, &act);
            let rsd_min = rsd.iter().copied().fold(f64::INFINITY, f64::min);
            let rsd_max = rsd.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let rsd_first = *rsd.first().unwrap_or(&0.0);
            let rsd_last = *rsd.last().unwrap_or(&0.0);
            report.add(
                "concentration RSD in [0, 1]",
                rsd.iter().all(|&v| (-1e-12..=1.0 + 1e-6).contains(&v)),
                format!("min = {rsd_min:.4}, max = {rsd_max:.4}, final = {rsd_last:.4}"),
            );
            let decay_ok = rsd_last <= rsd_first * 1.05 + 1e-9;
            report.add(
                "concentration RSD decays (mixing)",
                decay_ok,
                format!("initial = {rsd_first:.4} -> final = {rsd_last:.4}"),
            );
        }
    }

    report.passed = report.checks.iter().all(|c| c.passed);
    report
}
