//! Chain evaluation metrics & discrepancy heatmaps matching `postprocessing/postprocess.py`
//! and `postprocessing/postprocess_inhomogeneous.py`.

use std::collections::HashMap;

use crate::error::Result;
use crate::io::json::JsonValue;
use crate::linalg::{real_eigvals_abs, Matrix};
use crate::metrics::{
    entropy_concentration, intensity_of_segregation, mixing_times, normalize_by_particle_count,
    rsd_concentration, ValidationReport,
};

/// Complete post-processing & error metrics matching `postprocessing/postprocess.py`
/// and `postprocessing/postprocess_inhomogeneous.py`.
#[derive(Debug, Clone)]
pub struct ChainPredictionMetrics {
    pub times_eval: Vec<i64>,
    // ── Criterion: Teneur en l'espèce cible (small ou large) ──
    pub rsd_dem: Vec<f64>,
    pub rsd_markov: Vec<f64>,
    pub entropy_dem: Vec<f64>,
    pub entropy_markov: Vec<f64>,
    pub danckwerts_dem: Vec<f64>,
    pub danckwerts_markov: Vec<f64>,
    pub rmse_teneur: f64,
    pub mae_teneur: f64,
    pub max_err_teneur: f64,
    pub rmse_rsd: f64,
    pub mae_rsd: f64,
    pub max_abs_err_rsd: f64,
    pub t50_dem: Option<f64>,
    pub t90_dem: Option<f64>,
    pub t50_markov: Option<f64>,
    pub t90_markov: Option<f64>,
    // ── Nombre de particules (fig_compare_n_particles / fig_states_totale) ──
    pub n_particles_small_dem: Vec<f64>,
    pub n_particles_small_mk: Vec<f64>,
    pub n_particles_large_dem: Vec<f64>,
    pub n_particles_large_mk: Vec<f64>,
    pub n_particles_total_dem: Vec<f64>,
    pub n_particles_total_mk: Vec<f64>,
    pub mass_drift_pct: f64,
    // ── Erreur L1 normalisée par le nombre de particules (calculate_abs_error_over_time) ──
    pub l1_error_small: Vec<f64>,
    pub l1_error_large: Vec<f64>,
    pub l1_error_total: Vec<f64>,
    pub l1_mean_target: f64,
    pub l1_max_target: f64,
    pub l1_mean_total: f64,
    // ── Écart RMS & heatmap par cellule (calculate_discrepancy_per_cell) ──
    pub rms_discrepancy_target: Vec<f64>,
    pub rms_discrepancy_total: Vec<f64>,
    pub rms_discrepancy_mean: f64,
    pub rms_discrepancy_max: f64,
    pub cell_rmse_per_cell: Vec<f64>,
    // ── Diagnostics matriciels & validation physique ──
    pub diag_mean_small: f64,
    pub diag_mean_large: f64,
    pub lambda_max_small: f64,
    pub lambda_max_large: f64,
    pub mass_initial_small: f64,
    pub mass_final_small: f64,
    pub mass_initial_large: f64,
    pub mass_final_large: f64,
    pub validation_passed: bool,
    pub validation_checks: Vec<(String, bool, String)>,
}

/// Compact row for multi-method metric & error benchmarking (`POST /api/benchmark`).
#[derive(Debug, Clone)]
pub struct MethodBenchmarkRow {
    pub method: String,
    pub n_cells: usize,
    pub n_visited: usize,
    pub n_particles_dem_mean: f64,
    pub n_particles_hom_final: f64,
    pub n_particles_inhom_final: f64,
    pub hom_rmse_teneur: f64,
    pub inhom_rmse_teneur: f64,
    pub hom_mae_teneur: f64,
    pub inhom_mae_teneur: f64,
    pub hom_max_teneur: f64,
    pub inhom_max_teneur: f64,
    pub hom_rmse_rsd: f64,
    pub inhom_rmse_rsd: f64,
    pub hom_l1_mean: f64,
    pub inhom_l1_mean: f64,
    pub hom_rms_disc: f64,
    pub inhom_rms_disc: f64,
    pub lambda_max_hom: f64,
    pub diag_mean_hom: f64,
    pub t50_dem: Option<f64>,
    pub t50_hom: Option<f64>,
    pub t50_inhom: Option<f64>,
}

impl MethodBenchmarkRow {
    pub fn to_json(&self) -> JsonValue {
        let opt_num = |o: Option<f64>| o.map(JsonValue::Number).unwrap_or(JsonValue::Null);
        let mut o = JsonValue::object();
        o.insert("method", JsonValue::String(self.method.clone()));
        o.insert("n_cells", JsonValue::Number(self.n_cells as f64));
        o.insert("n_visited", JsonValue::Number(self.n_visited as f64));
        o.insert(
            "n_particles_dem_mean",
            JsonValue::Number(self.n_particles_dem_mean),
        );
        o.insert(
            "n_particles_hom_final",
            JsonValue::Number(self.n_particles_hom_final),
        );
        o.insert(
            "n_particles_inhom_final",
            JsonValue::Number(self.n_particles_inhom_final),
        );
        o.insert("hom_rmse_teneur", JsonValue::Number(self.hom_rmse_teneur));
        o.insert(
            "inhom_rmse_teneur",
            JsonValue::Number(self.inhom_rmse_teneur),
        );
        o.insert("hom_mae_teneur", JsonValue::Number(self.hom_mae_teneur));
        o.insert("inhom_mae_teneur", JsonValue::Number(self.inhom_mae_teneur));
        o.insert("hom_max_teneur", JsonValue::Number(self.hom_max_teneur));
        o.insert("inhom_max_teneur", JsonValue::Number(self.inhom_max_teneur));
        o.insert("hom_rmse_rsd", JsonValue::Number(self.hom_rmse_rsd));
        o.insert("inhom_rmse_rsd", JsonValue::Number(self.inhom_rmse_rsd));
        o.insert("hom_l1_mean", JsonValue::Number(self.hom_l1_mean));
        o.insert("inhom_l1_mean", JsonValue::Number(self.inhom_l1_mean));
        o.insert("hom_rms_disc", JsonValue::Number(self.hom_rms_disc));
        o.insert("inhom_rms_disc", JsonValue::Number(self.inhom_rms_disc));
        o.insert("lambda_max_hom", JsonValue::Number(self.lambda_max_hom));
        o.insert("diag_mean_hom", JsonValue::Number(self.diag_mean_hom));
        o.insert("t50_dem", opt_num(self.t50_dem));
        o.insert("t50_hom", opt_num(self.t50_hom));
        o.insert("t50_inhom", opt_num(self.t50_inhom));
        o
    }
}

pub(crate) struct ChainEvalInput<'a> {
    pub s_dem_small: &'a Matrix,
    pub s_dem_large: &'a Matrix,
    pub s_mk_small: &'a Matrix,
    pub s_mk_large: &'a Matrix,
    pub times_dem: &'a [i64],
    pub times_eval: &'a [i64],
    pub p_ref_small: &'a Matrix,
    pub p_ref_large: &'a Matrix,
    pub activated_small: &'a [bool],
    pub activated_large: &'a [bool],
    pub target_is_large: bool,
    pub validation: &'a ValidationReport,
}

fn build_dem_submatrix(s_dem: &Matrix, t_map: &HashMap<i64, usize>, times_eval: &[i64]) -> Matrix {
    let mut sub = Matrix::zeros(times_eval.len(), s_dem.cols);
    for (ei, &t) in times_eval.iter().enumerate() {
        let r = *t_map.get(&t).unwrap_or(&0);
        sub.row_mut(ei).copy_from_slice(s_dem.row(r));
    }
    sub
}

fn sum_matrices(a: &Matrix, b: &Matrix) -> Matrix {
    let mut out = Matrix::zeros(a.rows, a.cols);
    for i in 0..a.data.len() {
        out.data[i] = a.data[i] + b.data[i];
    }
    out
}

fn active_l1_and_rms_discrepancy(
    s_dem: &Matrix,
    s_mk: &Matrix,
    act_cols: &[usize],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n_t = s_dem.rows.min(s_mk.rows);
    let n_a = act_cols.len();
    if n_a == 0 || n_t == 0 {
        return (vec![0.0; n_t], vec![0.0; n_t], vec![0.0; s_dem.cols]);
    }
    let p_dem = normalize_by_particle_count(s_dem);
    let p_mk = normalize_by_particle_count(s_mk);

    let mut l1_series = Vec::with_capacity(n_t);
    let mut rms_series = Vec::with_capacity(n_t);
    let mut cell_sq_sum = vec![0.0f64; s_dem.cols];

    for r in 0..n_t {
        let mut l1 = 0.0;
        let mut sq = 0.0;
        for &c in act_cols {
            let d = (p_mk.get(r, c) - p_dem.get(r, c)).abs();
            l1 += d;
            sq += d * d;
            cell_sq_sum[c] += d * d;
        }
        l1_series.push(l1);
        rms_series.push((sq / n_a as f64).sqrt());
    }
    let cell_rmse: Vec<f64> = cell_sq_sum
        .into_iter()
        .map(|s| (s / n_t as f64).sqrt())
        .collect();
    (l1_series, rms_series, cell_rmse)
}

pub(crate) fn evaluate_chain_metrics(input: ChainEvalInput<'_>) -> Result<ChainPredictionMetrics> {
    let ChainEvalInput {
        s_dem_small,
        s_dem_large,
        s_mk_small,
        s_mk_large,
        times_dem,
        times_eval,
        p_ref_small,
        p_ref_large,
        activated_small,
        activated_large,
        target_is_large,
        validation,
    } = input;

    let t_map: HashMap<i64, usize> = times_dem.iter().enumerate().map(|(i, &t)| (t, i)).collect();
    let n_eval = times_eval.len();
    let nc = s_dem_small.cols;
    let dem_s_sub = build_dem_submatrix(s_dem_small, &t_map, times_eval);
    let dem_l_sub = build_dem_submatrix(s_dem_large, &t_map, times_eval);

    let (dem_t, dem_o, mk_t, mk_o, act_t, act_o) = if target_is_large {
        (
            &dem_l_sub,
            &dem_s_sub,
            s_mk_large,
            s_mk_small,
            activated_large,
            activated_small,
        )
    } else {
        (
            &dem_s_sub,
            &dem_l_sub,
            s_mk_small,
            s_mk_large,
            activated_small,
            activated_large,
        )
    };

    let rsd_dem = rsd_concentration(dem_t, dem_o, act_t, act_o);
    let rsd_markov = rsd_concentration(mk_t, mk_o, act_t, act_o);
    let entropy_dem = entropy_concentration(dem_t, dem_o, act_t, act_o, true);
    let entropy_markov = entropy_concentration(mk_t, mk_o, act_t, act_o, true);
    let danckwerts_dem = intensity_of_segregation(dem_t, dem_o, act_t, act_o);
    let danckwerts_markov = intensity_of_segregation(mk_t, mk_o, act_t, act_o);

    let times_f64: Vec<f64> = times_eval.iter().map(|&t| t as f64).collect();
    let mt_dem = mixing_times(&rsd_dem, &times_f64, &[0.5, 0.1]);
    let mt_mk = mixing_times(&rsd_markov, &times_f64, &[0.5, 0.1]);
    let t50_dem = mt_dem[0].1;
    let t90_dem = mt_dem[1].1;
    let t50_markov = mt_mk[0].1;
    let t90_markov = mt_mk[1].1;

    let n_cmp = rsd_dem.len().min(rsd_markov.len());
    let (mut sq_sum, mut abs_sum, mut max_abs) = (0.0, 0.0, 0.0f64);
    for i in 0..n_cmp {
        let e = (rsd_markov[i] - rsd_dem[i]).abs();
        sq_sum += e * e;
        abs_sum += e;
        max_abs = max_abs.max(e);
    }
    let denom = n_cmp.max(1) as f64;
    let rmse_rsd = (sq_sum / denom).sqrt();
    let mae_rsd = abs_sum / denom;

    let act_both: Vec<usize> = (0..nc)
        .filter(|&c| {
            *activated_small.get(c).unwrap_or(&false) && *activated_large.get(c).unwrap_or(&false)
        })
        .collect();
    let (mut t_sq, mut t_abs, mut t_max, mut t_cnt) = (0.0, 0.0, 0.0f64, 0usize);
    for r in 0..n_eval.min(mk_t.rows) {
        for &c in &act_both {
            let td_tot = dem_t.get(r, c) + dem_o.get(r, c);
            let tm_tot = mk_t.get(r, c) + mk_o.get(r, c);
            if td_tot > 1e-9 && tm_tot > 1e-9 {
                let td = dem_t.get(r, c) / td_tot;
                let tm = mk_t.get(r, c) / tm_tot;
                let err = (tm - td).abs();
                t_sq += err * err;
                t_abs += err;
                t_max = t_max.max(err);
                t_cnt += 1;
            }
        }
    }
    let rmse_teneur = if t_cnt > 0 {
        (t_sq / t_cnt as f64).sqrt()
    } else {
        0.0
    };
    let mae_teneur = if t_cnt > 0 { t_abs / t_cnt as f64 } else { 0.0 };
    let max_err_teneur = t_max;

    let mut n_particles_small_dem = Vec::with_capacity(n_eval);
    let mut n_particles_small_mk = Vec::with_capacity(n_eval);
    let mut n_particles_large_dem = Vec::with_capacity(n_eval);
    let mut n_particles_large_mk = Vec::with_capacity(n_eval);
    let mut n_particles_total_dem = Vec::with_capacity(n_eval);
    let mut n_particles_total_mk = Vec::with_capacity(n_eval);

    let act_s_cols: Vec<usize> = activated_small
        .iter()
        .enumerate()
        .filter_map(|(i, &a)| a.then_some(i))
        .collect();
    let act_l_cols: Vec<usize> = activated_large
        .iter()
        .enumerate()
        .filter_map(|(i, &a)| a.then_some(i))
        .collect();
    let act_any_cols: Vec<usize> = (0..nc)
        .filter(|&c| {
            *activated_small.get(c).unwrap_or(&false) || *activated_large.get(c).unwrap_or(&false)
        })
        .collect();

    for r in 0..n_eval {
        let ds: f64 = act_s_cols.iter().map(|&c| dem_s_sub.get(r, c)).sum();
        let ms: f64 = act_s_cols.iter().map(|&c| s_mk_small.get(r, c)).sum();
        let dl: f64 = act_l_cols.iter().map(|&c| dem_l_sub.get(r, c)).sum();
        let ml: f64 = act_l_cols.iter().map(|&c| s_mk_large.get(r, c)).sum();
        n_particles_small_dem.push(ds);
        n_particles_small_mk.push(ms);
        n_particles_large_dem.push(dl);
        n_particles_large_mk.push(ml);
        n_particles_total_dem.push(ds + dl);
        n_particles_total_mk.push(ms + ml);
    }

    let n0_tot = n_particles_total_mk.first().copied().unwrap_or(0.0);
    let nf_tot = n_particles_total_mk.last().copied().unwrap_or(0.0);
    let mass_drift_pct = if n0_tot > 1e-9 {
        ((nf_tot - n0_tot).abs() / n0_tot) * 100.0
    } else {
        0.0
    };

    let (l1_error_small, rms_small, cell_rmse_small) =
        active_l1_and_rms_discrepancy(&dem_s_sub, s_mk_small, &act_s_cols);
    let (l1_error_large, rms_large, cell_rmse_large) =
        active_l1_and_rms_discrepancy(&dem_l_sub, s_mk_large, &act_l_cols);
    let dem_tot_sub = sum_matrices(&dem_s_sub, &dem_l_sub);
    let mk_tot = sum_matrices(s_mk_small, s_mk_large);
    let (l1_error_total, rms_discrepancy_total, _) =
        active_l1_and_rms_discrepancy(&dem_tot_sub, &mk_tot, &act_any_cols);

    let (l1_target, rms_discrepancy_target, cell_rmse_per_cell) = if target_is_large {
        (&l1_error_large, rms_large, cell_rmse_large)
    } else {
        (&l1_error_small, rms_small, cell_rmse_small)
    };
    let l1_mean_target = if l1_target.is_empty() {
        0.0
    } else {
        l1_target.iter().sum::<f64>() / l1_target.len() as f64
    };
    let l1_max_target = l1_target.iter().copied().fold(0.0f64, f64::max);
    let l1_mean_total = if l1_error_total.is_empty() {
        0.0
    } else {
        l1_error_total.iter().sum::<f64>() / l1_error_total.len() as f64
    };

    let rms_discrepancy_mean = if rms_discrepancy_target.is_empty() {
        0.0
    } else {
        rms_discrepancy_target.iter().sum::<f64>() / rms_discrepancy_target.len() as f64
    };
    let rms_discrepancy_max = rms_discrepancy_target
        .iter()
        .copied()
        .fold(0.0f64, f64::max);

    let diag_mean = |p: &Matrix| -> f64 {
        let mut s = 0.0;
        let mut c = 0usize;
        for i in 0..p.rows.min(p.cols) {
            let v = p.get(i, i);
            if v.is_finite() {
                s += v;
                c += 1;
            }
        }
        if c > 0 {
            s / c as f64
        } else {
            0.0
        }
    };
    let lmax = |p: &Matrix| -> f64 {
        let clean = p.nan_to_num(0.0);
        real_eigvals_abs(&clean.transpose())
            .map(|e| e.into_iter().fold(0.0f64, f64::max))
            .unwrap_or(0.0)
    };

    let m_i_s: f64 = if n_eval > 0 {
        s_mk_small.row(0).iter().sum()
    } else {
        0.0
    };
    let m_f_s: f64 = if n_eval > 0 {
        s_mk_small.row(n_eval - 1).iter().sum()
    } else {
        0.0
    };
    let m_i_l: f64 = if n_eval > 0 {
        s_mk_large.row(0).iter().sum()
    } else {
        0.0
    };
    let m_f_l: f64 = if n_eval > 0 {
        s_mk_large.row(n_eval - 1).iter().sum()
    } else {
        0.0
    };

    let checks = validation
        .checks
        .iter()
        .map(|c| (c.name.clone(), c.passed, c.detail.clone()))
        .collect();

    Ok(ChainPredictionMetrics {
        times_eval: times_eval.to_vec(),
        rsd_dem,
        rsd_markov,
        entropy_dem,
        entropy_markov,
        danckwerts_dem,
        danckwerts_markov,
        rmse_teneur,
        mae_teneur,
        max_err_teneur,
        rmse_rsd,
        mae_rsd,
        max_abs_err_rsd: max_abs,
        t50_dem,
        t90_dem,
        t50_markov,
        t90_markov,
        n_particles_small_dem,
        n_particles_small_mk,
        n_particles_large_dem,
        n_particles_large_mk,
        n_particles_total_dem,
        n_particles_total_mk,
        mass_drift_pct,
        l1_error_small,
        l1_error_large,
        l1_error_total,
        l1_mean_target,
        l1_max_target,
        l1_mean_total,
        rms_discrepancy_target,
        rms_discrepancy_total,
        rms_discrepancy_mean,
        rms_discrepancy_max,
        cell_rmse_per_cell,
        diag_mean_small: diag_mean(p_ref_small),
        diag_mean_large: diag_mean(p_ref_large),
        lambda_max_small: lmax(p_ref_small),
        lambda_max_large: lmax(p_ref_large),
        mass_initial_small: m_i_s,
        mass_final_small: m_f_s,
        mass_initial_large: m_i_l,
        mass_final_large: m_f_l,
        validation_passed: validation.passed,
        validation_checks: checks,
    })
}

pub(crate) fn cell_teneur_row(s_target: &[f64], s_other: &[f64]) -> Vec<f64> {
    s_target
        .iter()
        .zip(s_other.iter())
        .map(|(&a, &b)| {
            let tot = a + b;
            if tot > 1e-12 {
                (a / tot).clamp(0.0, 1.0)
            } else {
                0.5
            }
        })
        .collect()
}

pub(crate) fn build_discrepancy_heatmap(
    s_dem: &Matrix,
    s_mk: &Matrix,
    times_dem: &[i64],
    times_eval: &[i64],
    active_cols: &[usize],
) -> (Vec<i64>, Vec<Vec<f64>>) {
    let t_map: HashMap<i64, usize> = times_dem.iter().enumerate().map(|(i, &t)| (t, i)).collect();
    let dem_sub = build_dem_submatrix(s_dem, &t_map, times_eval);
    let p_dem = normalize_by_particle_count(&dem_sub);
    let p_mk = normalize_by_particle_count(s_mk);
    let n_eval = times_eval.len().min(p_mk.rows);
    let stride = (n_eval / 48).max(1);
    let mut h_times = Vec::new();
    for r in (0..n_eval).step_by(stride) {
        h_times.push(times_eval[r]);
    }
    let mut mat = Vec::with_capacity(active_cols.len());
    for &c in active_cols {
        let mut row = Vec::with_capacity(h_times.len());
        for r in (0..n_eval).step_by(stride) {
            row.push((p_mk.get(r, c) - p_dem.get(r, c)).abs());
        }
        mat.push(row);
    }
    (h_times, mat)
}

pub(crate) fn matrix_to_json(m: &Matrix) -> JsonValue {
    JsonValue::Array(
        (0..m.rows)
            .map(|r| {
                JsonValue::Array(
                    m.row(r)
                        .iter()
                        .map(|&v| {
                            if v.is_finite() {
                                JsonValue::Number(v)
                            } else {
                                JsonValue::Null
                            }
                        })
                        .collect(),
                )
            })
            .collect(),
    )
}

pub(crate) fn vec_f64_json(v: &[f64]) -> JsonValue {
    JsonValue::Array(
        v.iter()
            .map(|&x| {
                if x.is_finite() {
                    JsonValue::Number(x)
                } else {
                    JsonValue::Null
                }
            })
            .collect(),
    )
}

impl ChainPredictionMetrics {
    pub fn to_json(&self) -> JsonValue {
        let opt_num = |o: Option<f64>| o.map(JsonValue::Number).unwrap_or(JsonValue::Null);
        let mut o = JsonValue::object();
        o.insert(
            "times_eval",
            JsonValue::Array(
                self.times_eval
                    .iter()
                    .map(|&t| JsonValue::Number(t as f64))
                    .collect(),
            ),
        );
        o.insert("rsd_dem", vec_f64_json(&self.rsd_dem));
        o.insert("rsd_markov", vec_f64_json(&self.rsd_markov));
        o.insert("entropy_dem", vec_f64_json(&self.entropy_dem));
        o.insert("entropy_markov", vec_f64_json(&self.entropy_markov));
        o.insert("danckwerts_dem", vec_f64_json(&self.danckwerts_dem));
        o.insert("danckwerts_markov", vec_f64_json(&self.danckwerts_markov));
        o.insert("rmse_teneur", JsonValue::Number(self.rmse_teneur));
        o.insert("mae_teneur", JsonValue::Number(self.mae_teneur));
        o.insert("max_err_teneur", JsonValue::Number(self.max_err_teneur));
        o.insert("rmse_rsd", JsonValue::Number(self.rmse_rsd));
        o.insert("mae_rsd", JsonValue::Number(self.mae_rsd));
        o.insert("max_abs_err_rsd", JsonValue::Number(self.max_abs_err_rsd));
        o.insert("t50_dem", opt_num(self.t50_dem));
        o.insert("t90_dem", opt_num(self.t90_dem));
        o.insert("t50_markov", opt_num(self.t50_markov));
        o.insert("t90_markov", opt_num(self.t90_markov));
        o.insert(
            "n_particles_small_dem",
            vec_f64_json(&self.n_particles_small_dem),
        );
        o.insert(
            "n_particles_small_mk",
            vec_f64_json(&self.n_particles_small_mk),
        );
        o.insert(
            "n_particles_large_dem",
            vec_f64_json(&self.n_particles_large_dem),
        );
        o.insert(
            "n_particles_large_mk",
            vec_f64_json(&self.n_particles_large_mk),
        );
        o.insert(
            "n_particles_total_dem",
            vec_f64_json(&self.n_particles_total_dem),
        );
        o.insert(
            "n_particles_total_mk",
            vec_f64_json(&self.n_particles_total_mk),
        );
        o.insert("mass_drift_pct", JsonValue::Number(self.mass_drift_pct));
        o.insert("l1_error_small", vec_f64_json(&self.l1_error_small));
        o.insert("l1_error_large", vec_f64_json(&self.l1_error_large));
        o.insert("l1_error_total", vec_f64_json(&self.l1_error_total));
        o.insert("l1_mean_target", JsonValue::Number(self.l1_mean_target));
        o.insert("l1_max_target", JsonValue::Number(self.l1_max_target));
        o.insert("l1_mean_total", JsonValue::Number(self.l1_mean_total));
        o.insert(
            "rms_discrepancy_target",
            vec_f64_json(&self.rms_discrepancy_target),
        );
        o.insert(
            "rms_discrepancy_total",
            vec_f64_json(&self.rms_discrepancy_total),
        );
        o.insert(
            "rms_discrepancy_mean",
            JsonValue::Number(self.rms_discrepancy_mean),
        );
        o.insert(
            "rms_discrepancy_max",
            JsonValue::Number(self.rms_discrepancy_max),
        );
        o.insert("cell_rmse_per_cell", vec_f64_json(&self.cell_rmse_per_cell));
        o.insert("diag_mean_small", JsonValue::Number(self.diag_mean_small));
        o.insert("diag_mean_large", JsonValue::Number(self.diag_mean_large));
        o.insert("lambda_max_small", JsonValue::Number(self.lambda_max_small));
        o.insert("lambda_max_large", JsonValue::Number(self.lambda_max_large));
        o.insert(
            "mass_initial_small",
            JsonValue::Number(self.mass_initial_small),
        );
        o.insert("mass_final_small", JsonValue::Number(self.mass_final_small));
        o.insert(
            "mass_initial_large",
            JsonValue::Number(self.mass_initial_large),
        );
        o.insert("mass_final_large", JsonValue::Number(self.mass_final_large));
        o.insert("validation_passed", JsonValue::Bool(self.validation_passed));
        o.insert(
            "validation_checks",
            JsonValue::Array(
                self.validation_checks
                    .iter()
                    .map(|(n, p, d)| {
                        let mut c = JsonValue::object();
                        c.insert("name", JsonValue::String(n.clone()));
                        c.insert("passed", JsonValue::Bool(*p));
                        c.insert("detail", JsonValue::String(d.clone()));
                        c
                    })
                    .collect(),
            ),
        );
        o
    }
}
