//! Pure-Rust interactive simulation runner for 3-D mesh extraction,
//! full-duration DEM prediction, and `postprocess.py` / `postprocess_inhomogeneous.py`
//! metric & error comparison (teneur, particle counts, L1, RMS discrepancy,
//! RSD, entropy, Danckwerts, mixing times, inter-block diffs & p_ij(t) laws).

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::config::{ExperimentData, SpeciesExperimentData};
use crate::data::{DataSource, DemFrame, LocalDataSource};
use crate::error::{DemMcmError, Result};
use crate::gui::mesh3d::{extract_mesh_3d, Mesh3DData};
use crate::gui::state::SimulationRequest;
use crate::io::json::JsonValue;
use crate::linalg::{real_eigvals_abs, Matrix};
use crate::metrics::{
    clean_transition_matrix, entropy_concentration, fit_probability_evolution,
    intensity_of_segregation, mixing_times, normalize_by_particle_count, propagate_markov,
    propagate_markov_inhomogeneous, rsd_concentration, stationary_distribution,
    validate_experiment, ValidationReport,
};
use crate::partitioners::{create_partitioner, PartitionerDiagnostics};
use crate::run_sweep::{
    detect_species, run_experiment, run_inhomogeneous_experiment, run_no_species_experiment,
    ExperimentConfig,
};

/// Cached dataset holder using `Arc` so repeated mesh switches & benchmarks are zero-copy.
#[derive(Debug, Default)]
pub struct GuiDatasetCache {
    synthetic: Option<Arc<BTreeMap<usize, DemFrame>>>,
    real_chunks: Mutex<Option<Arc<BTreeMap<usize, DemFrame>>>>,
}

impl GuiDatasetCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate a realistic 3-D rotating-drum DEM dataset across **360 timesteps** (`0..359`)
    /// with **1 030 particles/timestep** matching real DEM proportions:
    /// `350` small particles (`d = 0.004`) and `680` large particles (`d = 0.008`).
    pub fn get_synthetic(&mut self) -> Arc<BTreeMap<usize, DemFrame>> {
        if self.synthetic.is_none() {
            self.synthetic = Some(Arc::new(generate_synthetic_drum_frames(360, 1030)));
        }
        Arc::clone(self.synthetic.as_ref().unwrap())
    }

    /// Load the **entire DEM simulation duration** from `data/chunks` (`0..5999`)
    /// and keep it in an `Arc` cache.
    pub fn get_or_load_real(&self, data_root: &Path) -> Result<Arc<BTreeMap<usize, DemFrame>>> {
        let mut guard = self
            .real_chunks
            .lock()
            .map_err(|_| DemMcmError::DataSource("GUI dataset mutex poisoned".to_string()))?;
        if let Some(ref cached) = *guard {
            return Ok(Arc::clone(cached));
        }
        let src = LocalDataSource::new(data_root);
        let frames = src.read_timesteps(None)?;
        if frames.is_empty() {
            return Err(DemMcmError::MissingData(format!(
                "No DEM frames loaded from {}",
                data_root.display()
            )));
        }
        let arc = Arc::new(frames);
        *guard = Some(Arc::clone(&arc));
        Ok(arc)
    }

    pub fn has_real_cached(&self) -> bool {
        self.real_chunks
            .lock()
            .map(|g| g.is_some())
            .unwrap_or(false)
    }
}

/// Generate a realistic 3-D rotating drum DEM trajectory (`0..n_frames`).
/// Ratios match the real drum DEM (~34% small `d = 0.004`, ~66% large `d = 0.008`).
pub fn generate_synthetic_drum_frames(
    n_frames: usize,
    n_particles: usize,
) -> BTreeMap<usize, DemFrame> {
    let mut frames = BTreeMap::new();
    let r_drum = 0.095;
    let z_half = 0.025;
    let omega = 4.0;
    let n_small = ((n_particles as f64) * 0.34).round() as usize;

    for ts in 0..n_frames {
        let mix_progress = 1.0 - (-0.011 * ts as f64).exp();
        let mut coords = Vec::with_capacity(n_particles);
        let mut vels = Vec::with_capacity(n_particles);
        let mut diams = Vec::with_capacity(n_particles);

        for p in 0..n_particles {
            let is_small = p < n_small;
            let li = if is_small { p } else { p - n_small };
            let u1 = ((li * 73 + 19) % 997) as f64 / 997.0;
            let u2 = ((li * 131 + 37) % 991) as f64 / 991.0;
            let u3 = ((li * 211 + 53) % 983) as f64 / 983.0;

            let phi_seed = (li as f64 * 0.61803398875) % 1.0;
            let base_angle = if is_small {
                std::f64::consts::PI * (0.68 + 0.64 * u1)
            } else {
                std::f64::consts::PI * (-0.32 + 0.64 * u1)
            };
            let freq = 0.048 + 0.042 * u2;
            let shear = mix_progress * (0.4 + 3.2 * phi_seed) * std::f64::consts::PI;
            let theta =
                base_angle + freq * ts as f64 + shear * if li % 2 == 0 { 1.0 } else { -0.75 };
            // Slight radial segregation: small particles migrate slightly inward
            let r_bias = if is_small { 0.88 } else { 1.04 };
            let r = (r_drum
                * r_bias
                * (0.18
                    + 0.78 * (0.35 + 0.65 * u2).sqrt() * (1.0 - 0.10 * mix_progress * (0.5 - u3))))
                .clamp(0.012, r_drum * 0.96);

            let x = r * theta.cos();
            let y = r * theta.sin() - 0.015;
            let z_mix = (u3 - 0.5) * 2.0 * z_half + 0.004 * (0.07 * ts as f64 + li as f64).sin();
            let z = z_mix.clamp(-z_half, z_half);

            let vx = -omega * y + 0.04 * theta.sin();
            let vy = omega * x - 0.04 * theta.cos();
            let vz = 0.015 * (0.10 * ts as f64 + li as f64).cos();

            coords.push([x, y, z]);
            vels.push([vx, vy, vz]);
            diams.push(if is_small { 0.004 } else { 0.008 });
        }

        frames.insert(ts, DemFrame::with_kinematics(ts, coords, vels, diams));
    }
    frames
}

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

/// One inhomogeneous temporal block summary + transition matrix.
#[derive(Debug, Clone)]
pub struct BlockSummary {
    pub block_index: usize,
    pub t_start: i64,
    pub t_end: i64,
    pub n_visited_small: usize,
    pub diag_mean_small: f64,
    pub lambda_max_small: f64,
    pub p_small: Matrix,
    pub p_large: Matrix,
}

/// Fitted polynomial evolution law for a significant transition `p_{i->j}(t)` across NLT blocks
/// (`postprocessing/postprocess_inhomogeneous.py::fig_matrix_components_evolution`).
#[derive(Debug, Clone)]
pub struct TransitionLawItem {
    pub src: usize,
    pub dst: usize,
    pub block_times: Vec<f64>,
    pub values: Vec<f64>,
    pub predicted: Vec<f64>,
    pub degree: usize,
    pub rmse: f64,
    pub r2: f64,
    pub law_label: String,
}

/// Downsampled particle & cell snapshot for 3-D playback along the full DEM duration.
#[derive(Debug, Clone)]
pub struct SnapshotFrame3D {
    pub timestep: usize,
    pub coords: Vec<[f64; 3]>,
    pub velocities: Vec<[f64; 3]>,
    pub diameters: Vec<f64>,
    pub is_small: Vec<bool>,
    pub cell_ids: Vec<i64>,
    pub cell_pop: Vec<usize>,
    /// DEM local content (teneur) of `target_species` per cell `[0, 1]`.
    pub cell_conc_dem: Vec<f64>,
    /// True full-timestep particle counts in the DEM (un-subsampled).
    pub n_small_total: usize,
    pub n_large_total: usize,
    pub n_total: usize,
}

/// Full comparison bundle returned by `run_gui_comparison`.
#[derive(Debug, Clone)]
pub struct ComparisonBundle {
    pub dataset_name: String,
    pub method: String,
    pub label: String,
    pub target_species: String,
    pub is_single_species: bool,
    pub dem_t_min: i64,
    pub dem_t_max: i64,
    pub dem_total_timesteps: usize,
    pub model_t_start: i64,
    pub model_t_end: i64,
    pub pred_t_start: i64,
    pub pred_t_end: i64,
    pub mesh: Mesh3DData,
    pub diagnostics: PartitionerDiagnostics,
    pub hom_metrics: ChainPredictionMetrics,
    pub inhom_metrics: ChainPredictionMetrics,
    pub p_hom_small: Matrix,
    pub p_hom_large: Matrix,
    pub blocks: Vec<BlockSummary>,
    pub block_rel_diffs: Vec<f64>,
    pub p_laws: Vec<TransitionLawItem>,
    pub frames: Vec<SnapshotFrame3D>,
    /// Per-eval-step, per-cell teneur of `target_species` `[n_eval][n_cells]`.
    pub hom_conc_traj: Vec<Vec<f64>>,
    pub inhom_conc_traj: Vec<Vec<f64>>,
    pub dem_conc_traj: Vec<Vec<f64>>,
    /// Top populated cells `[top_cells]` + per-cell population trajectories for DEM vs Hom vs Inhom.
    pub top_cells: Vec<usize>,
    pub top_cell_pop_dem: Vec<Vec<f64>>,
    pub top_cell_pop_hom: Vec<Vec<f64>>,
    pub top_cell_pop_inhom: Vec<Vec<f64>>,
    /// Per-cell discrepancy heatmap `[n_active][n_heat_times]` for Hom and Inhom.
    pub active_cell_ids: Vec<usize>,
    pub heatmap_times: Vec<i64>,
    pub discrepancy_heatmap_hom: Vec<Vec<f64>>,
    pub discrepancy_heatmap_inhom: Vec<Vec<f64>>,
    /// Strongest Markov transition edges `(src_cell, dst_cell, prob)` for 3-D flow arrows.
    pub top_edges_small: Vec<(usize, usize, f64)>,
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

struct ChainEvalInput<'a> {
    s_dem_small: &'a Matrix,
    s_dem_large: &'a Matrix,
    s_mk_small: &'a Matrix,
    s_mk_large: &'a Matrix,
    times_dem: &'a [i64],
    times_eval: &'a [i64],
    p_ref_small: &'a Matrix,
    p_ref_large: &'a Matrix,
    activated_small: &'a [bool],
    activated_large: &'a [bool],
    target_is_large: bool,
    validation: &'a ValidationReport,
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

fn evaluate_chain_metrics(input: ChainEvalInput<'_>) -> Result<ChainPredictionMetrics> {
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

    // Order matrices according to target_species so RSD / Entropy / Danckwerts / Teneur reflect it!
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

    // RSD errors
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

    // Direct Teneur error across mutually active cells & timesteps
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

    // Particle counts across full eval trajectory (fig_compare_n_particles / fig_states_totale)
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

    // Normalized L1 error + RMS discrepancy per cell (postprocess.py)
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

fn cell_teneur_row(s_target: &[f64], s_other: &[f64]) -> Vec<f64> {
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

fn build_discrepancy_heatmap(
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

/// Run partitioner fitting + 3-D mesh extraction + homogeneous & inhomogeneous
/// model construction + **full-duration DEM prediction** (`ts_min .. ts_max`).
pub fn run_gui_comparison(
    req: &SimulationRequest,
    frames: &BTreeMap<usize, DemFrame>,
) -> Result<ComparisonBundle> {
    if frames.is_empty() {
        return Err(DemMcmError::MissingData(
            "Empty DEM frame collection".to_string(),
        ));
    }

    let sorted_ts: Vec<usize> = frames.keys().copied().collect();
    let ts_min = sorted_ts[0];
    let ts_max = *sorted_ts.last().unwrap();

    // Model construction window (`start_index .. start_index + NLT*(step+tau)`)
    let start_index = if frames.contains_key(&req.start_index) {
        req.start_index
    } else {
        ts_min
    };
    let perm_start = if req.permanent_start >= ts_min && req.permanent_start <= ts_max {
        req.permanent_start
    } else {
        ts_min
    };

    // Subsample permanent phase cleanly to <= 24,000 points for fast interactive fitting
    // while keeping full spatial extent (`cartesian` uses all frames or dense sample).
    let perm_ts: Vec<usize> = sorted_ts
        .iter()
        .copied()
        .filter(|&t| t >= perm_start)
        .collect();
    let fit_ts = if perm_ts.is_empty() {
        &sorted_ts
    } else {
        &perm_ts
    };
    let approx_pts = frames[&fit_ts[0]].n_particles().max(1);
    let target_frames = (24_000 / approx_pts).max(1);
    let fit_stride = (fit_ts.len() / target_frames).max(1);

    let mut perm_coords = Vec::new();
    let mut perm_vels = Vec::new();
    for (idx, &ts) in fit_ts.iter().enumerate() {
        if idx % fit_stride != 0 && ts != ts_max {
            continue;
        }
        let f = &frames[&ts];
        perm_coords.extend_from_slice(&f.coordinates);
        perm_vels.extend_from_slice(&f.velocities);
    }

    let mut partitioner = create_partitioner(&req.method, &req.kwargs)?;
    if perm_vels.len() == perm_coords.len() {
        partitioner.set_dem_velocities(Some(&perm_vels));
    }
    partitioner.fit(&perm_coords)?;

    // Build model configuration (construction phase)
    let nlt = req.nlt.max(1);
    let step = req.step.max(1);
    let dt = req.dt.max(1);
    let tau = req.tau.max(1);
    let model_t_start = start_index as i64;
    let model_t_end = (start_index + nlt * (step + tau)).min(ts_max) as i64;

    let cfg = ExperimentConfig {
        method: req.method.clone(),
        method_kwargs: req.kwargs.clone(),
        nlt,
        tau,
        step,
        dt,
        start_index,
        particle_diameter: None,
        inhomogeneous: false,
    };

    let hom_res = if req.no_species {
        run_no_species_experiment(&cfg, partitioner.as_mut(), frames)?
    } else {
        run_experiment(&cfg, partitioner.as_mut(), frames)?
    };

    let mut cfg_inh = cfg.clone();
    cfg_inh.inhomogeneous = true;
    let inh_res = if req.no_species {
        run_no_species_experiment(&cfg_inh, partitioner.as_mut(), frames)?
    } else {
        run_inhomogeneous_experiment(&cfg_inh, partitioner.as_mut(), frames)?
    };

    let mut species_keys: Vec<String> = hom_res.species.keys().cloned().collect();
    species_keys.sort();
    let is_single = species_keys.len() < 2;
    let (sp_a_key, sp_b_key) = if is_single {
        (species_keys[0].as_str(), species_keys[0].as_str())
    } else {
        let small_k = species_keys
            .iter()
            .find(|s| s.as_str() == "small")
            .unwrap_or(&species_keys[0]);
        let large_k = species_keys
            .iter()
            .find(|s| s.as_str() == "large")
            .unwrap_or(&species_keys[1]);
        (small_k.as_str(), large_k.as_str())
    };
    let target_is_large = !is_single && req.target_species == "large";

    let hs = &hom_res.species[sp_a_key];
    let hl = &hom_res.species[sp_b_key];
    let p_hom_s = hs.p.as_ref().unwrap();
    let p_hom_l = hl.p.as_ref().unwrap();
    let (ps_clean, act_s) = clean_transition_matrix(p_hom_s, 0.5);
    let (pl_clean, act_l) = clean_transition_matrix(p_hom_l, 0.5);

    let times_i64 = &hs.times;
    // Full-duration DEM prediction (`pred_start = ts_min`, matching postprocess.prepare_species)
    let pred_start = if req.predict_full_duration {
        ts_min
    } else {
        start_index
    };
    let i_pred_row = times_i64
        .iter()
        .position(|&t| t >= pred_start as i64)
        .unwrap_or(0);
    let pred_t_start = times_i64[i_pred_row];
    let pred_t_end = *times_i64.last().unwrap_or(&pred_t_start);

    // 1. Propagate Homogeneous chain across full DEM duration
    let (s_mk_hom_s, times_eval) = propagate_markov(
        hs.s_matrix.row(i_pred_row),
        &ps_clean,
        times_i64,
        pred_t_start,
        tau,
        &act_s,
    )?;
    let (s_mk_hom_l, _) = propagate_markov(
        hl.s_matrix.row(i_pred_row),
        &pl_clean,
        times_i64,
        pred_t_start,
        tau,
        &act_l,
    )?;

    let mut hom_exp_map = HashMap::new();
    hom_exp_map.insert(
        sp_a_key.to_string(),
        SpeciesExperimentData {
            s_matrix: hs.s_matrix.clone(),
            times: times_i64.clone(),
            p: Some(p_hom_s.clone()),
            p_blocks: None,
        },
    );
    if !is_single {
        hom_exp_map.insert(
            sp_b_key.to_string(),
            SpeciesExperimentData {
                s_matrix: hl.s_matrix.clone(),
                times: times_i64.clone(),
                p: Some(p_hom_l.clone()),
                p_blocks: None,
            },
        );
    }
    let hom_folder = cfg.output_folder(None)?;
    let hom_exp = ExperimentData {
        config: cfg.to_json(),
        stats: hom_res.stats.clone(),
        inhomogeneous: false,
        inhomogeneous_metadata: None,
        states_matrix: Some(hom_res.states_matrix.clone()),
        species: hom_exp_map,
    };
    let val_hom = validate_experiment(&hom_folder, &hom_exp, 1e-3);
    let hom_metrics = evaluate_chain_metrics(ChainEvalInput {
        s_dem_small: &hs.s_matrix,
        s_dem_large: &hl.s_matrix,
        s_mk_small: &s_mk_hom_s,
        s_mk_large: &s_mk_hom_l,
        times_dem: times_i64,
        times_eval: &times_eval,
        p_ref_small: p_hom_s,
        p_ref_large: p_hom_l,
        activated_small: &act_s,
        activated_large: &act_l,
        target_is_large,
        validation: &val_hom,
    })?;

    // 2. Propagate Inhomogeneous chain across full DEM duration.
    // Fill any active row missing transitions in a short sub-block with the
    // homogeneous row (or self-loop) so particle count is strictly conserved (0% drift).
    let is_res = inh_res
        .species
        .get(sp_a_key)
        .or_else(|| inh_res.species.values().next())
        .ok_or_else(|| DemMcmError::MissingData("No inhomogeneous species output".to_string()))?;
    let il_res = inh_res.species.get(sp_b_key).unwrap_or(is_res);
    let fallback_blocks_s = vec![p_hom_s.clone(); nlt];
    let fallback_blocks_l = vec![p_hom_l.clone(); nlt];
    let raw_blocks_s = is_res.p_blocks.as_ref().unwrap_or(&fallback_blocks_s);
    let raw_blocks_l = il_res.p_blocks.as_ref().unwrap_or(&fallback_blocks_l);

    let stabilize_blocks = |raw: &[Matrix], p_fallback: &Matrix, act: &[bool]| -> Vec<Matrix> {
        raw.iter()
            .map(|b| {
                let (mut cb, b_act) = clean_transition_matrix(b, 0.5);
                for (r, &is_act) in act.iter().enumerate() {
                    if is_act && !b_act[r] {
                        let fb_sum: f64 = p_fallback.row(r).iter().sum();
                        if fb_sum > 0.5 {
                            cb.set_row(r, p_fallback.row(r));
                        } else {
                            cb.set(r, r, 1.0);
                        }
                    }
                }
                cb
            })
            .collect()
    };

    let clean_blocks_s = stabilize_blocks(raw_blocks_s, &ps_clean, &act_s);
    let clean_blocks_l = stabilize_blocks(raw_blocks_l, &pl_clean, &act_l);
    let (s_mk_inh_s, inh_times_eval) = propagate_markov_inhomogeneous(
        is_res.s_matrix.row(i_pred_row),
        &clean_blocks_s,
        times_i64,
        pred_t_start,
        tau,
        &act_s,
        Some(step),
        Some(nlt),
    )?;
    let (s_mk_inh_l, _) = propagate_markov_inhomogeneous(
        il_res.s_matrix.row(i_pred_row),
        &clean_blocks_l,
        times_i64,
        pred_t_start,
        tau,
        &act_l,
        Some(step),
        Some(nlt),
    )?;

    let mut inh_exp_map = HashMap::new();
    inh_exp_map.insert(
        sp_a_key.to_string(),
        SpeciesExperimentData {
            s_matrix: is_res.s_matrix.clone(),
            times: times_i64.clone(),
            p: None,
            p_blocks: Some(raw_blocks_s.clone()),
        },
    );
    if !is_single {
        inh_exp_map.insert(
            sp_b_key.to_string(),
            SpeciesExperimentData {
                s_matrix: il_res.s_matrix.clone(),
                times: times_i64.clone(),
                p: None,
                p_blocks: Some(raw_blocks_l.clone()),
            },
        );
    }
    let mut inh_meta = JsonValue::object();
    inh_meta.insert("n_blocks", JsonValue::Number(raw_blocks_s.len() as f64));
    inh_meta.insert("step", JsonValue::Number(step as f64));
    inh_meta.insert("nlt", JsonValue::Number(nlt as f64));
    let inh_folder = format!("inhomogeneous_{hom_folder}");
    let inh_exp = ExperimentData {
        inhomogeneous: true,
        config: cfg_inh.to_json(),
        stats: inh_res.stats.clone(),
        inhomogeneous_metadata: Some(inh_meta),
        states_matrix: Some(inh_res.states_matrix.clone()),
        species: inh_exp_map,
    };
    let val_inh = validate_experiment(&inh_folder, &inh_exp, 1e-3);
    let block0_s = raw_blocks_s.first().unwrap_or(p_hom_s);
    let block0_l = raw_blocks_l.first().unwrap_or(p_hom_l);
    let inhom_metrics = evaluate_chain_metrics(ChainEvalInput {
        s_dem_small: &is_res.s_matrix,
        s_dem_large: &il_res.s_matrix,
        s_mk_small: &s_mk_inh_s,
        s_mk_large: &s_mk_inh_l,
        times_dem: times_i64,
        times_eval: &inh_times_eval,
        p_ref_small: block0_s,
        p_ref_large: block0_l,
        activated_small: &act_s,
        activated_large: &act_l,
        target_is_large,
        validation: &val_inh,
    })?;

    // Per-block summaries + consecutive block relative diffs (fig_matrix_differences)
    let mut blocks = Vec::with_capacity(raw_blocks_s.len());
    let mut block_times_f64 = Vec::with_capacity(raw_blocks_s.len());
    for (b, bs) in raw_blocks_s.iter().enumerate() {
        let bl = &raw_blocks_l[b];
        let t0 = start_index as i64 + (b * (step + tau)) as i64;
        let t1 = t0 + (step + tau) as i64;
        block_times_f64.push(t0 as f64);
        let mut diag_vals = Vec::new();
        let mut n_vis = 0usize;
        for i in 0..bs.rows {
            let v = bs.get(i, i);
            if bs.row(i).iter().any(|&x| x.is_finite() && x > 0.0) {
                n_vis += 1;
            }
            if v.is_finite() {
                diag_vals.push(v);
            }
        }
        let diag_mean = if diag_vals.is_empty() {
            0.0
        } else {
            diag_vals.iter().sum::<f64>() / diag_vals.len() as f64
        };
        let lam = real_eigvals_abs(&bs.nan_to_num(0.0).transpose())
            .map(|e| e.into_iter().fold(0.0f64, f64::max))
            .unwrap_or(0.0);
        blocks.push(BlockSummary {
            block_index: b,
            t_start: t0,
            t_end: t1,
            n_visited_small: n_vis,
            diag_mean_small: diag_mean,
            lambda_max_small: lam,
            p_small: bs.clone(),
            p_large: bl.clone(),
        });
    }

    let ref_blocks = if target_is_large {
        &clean_blocks_l
    } else {
        &clean_blocks_s
    };
    let mut block_rel_diffs = Vec::new();
    for k in 0..ref_blocks.len().saturating_sub(1) {
        let a = &ref_blocks[k];
        let b = &ref_blocks[k + 1];
        let norm_a: f64 = a.data.iter().map(|x| x.abs()).sum();
        let norm_diff: f64 = a
            .data
            .iter()
            .zip(b.data.iter())
            .map(|(&x, &y)| (y - x).abs())
            .sum();
        block_rel_diffs.push(if norm_a > 1e-12 {
            norm_diff / norm_a
        } else {
            0.0
        });
    }

    // Fit polynomial laws on top significant off-diagonal p_{i->j}(t) across NLT blocks
    let mut p_laws = Vec::new();
    if ref_blocks.len() >= 2 {
        let n_st = ref_blocks[0].rows;
        let mut candidates: Vec<(usize, usize, f64)> = Vec::new();
        for i in 0..n_st {
            for j in 0..n_st {
                if i == j {
                    continue;
                }
                let mx = ref_blocks
                    .iter()
                    .map(|m| m.get(i, j))
                    .fold(0.0f64, f64::max);
                if mx > 0.04 {
                    candidates.push((i, j, mx));
                }
            }
        }
        candidates.sort_by(|a, b| b.2.total_cmp(&a.2));
        candidates.truncate(8);
        for (i, j, _) in candidates {
            let vals: Vec<f64> = ref_blocks.iter().map(|m| m.get(i, j)).collect();
            if let Ok(fit) = fit_probability_evolution(&block_times_f64, &vals, 2) {
                let pred = fit.predict(&block_times_f64);
                p_laws.push(TransitionLawItem {
                    src: i,
                    dst: j,
                    block_times: block_times_f64.clone(),
                    values: vals,
                    predicted: pred,
                    degree: fit.degree,
                    rmse: fit.rmse,
                    r2: fit.r2,
                    law_label: fit.law_label,
                });
            }
        }
    }

    // 3-D mesh extraction using representative permanent-phase coordinates so every
    // spatial cluster is faithfully represented, plus per-frame diagnostics.
    let first_frame = frames
        .get(&start_index)
        .unwrap_or_else(|| frames.values().next().unwrap());
    let mesh_coords: Vec<[f64; 3]> = if perm_coords.len() <= 4000 {
        perm_coords.clone()
    } else {
        let step_p = (perm_coords.len() / 4000).max(1);
        perm_coords.iter().step_by(step_p).copied().collect()
    };
    let mesh_vels: Vec<[f64; 3]> = if perm_vels.len() == perm_coords.len() {
        if perm_vels.len() <= 4000 {
            perm_vels.clone()
        } else {
            let step_p = (perm_vels.len() / 4000).max(1);
            perm_vels.iter().step_by(step_p).copied().collect()
        }
    } else {
        Vec::new()
    };
    let mesh_vel_opt = if mesh_vels.len() == mesh_coords.len() {
        Some(mesh_vels.as_slice())
    } else {
        None
    };
    let mesh_states = partitioner.compute_states(&mesh_coords, mesh_vel_opt)?;
    let mesh = extract_mesh_3d(partitioner.as_ref(), &mesh_coords, &mesh_states)?;
    let diagnostics =
        partitioner.diagnostics(&first_frame.coordinates, Some(&first_frame.velocities))?;

    // Downsampled 3-D playback frames across the full DEM duration (up to 48 steps, all two particle sizes)
    let n_cells = partitioner.n_cells();
    let first_ref = frames.values().next().unwrap();
    let sp_groups = detect_species(first_ref);
    let sp_small_mask = sp_groups
        .iter()
        .find(|g| g.0 == "small")
        .or_else(|| sp_groups.first())
        .map(|g| g.1.clone())
        .unwrap_or_else(|| vec![true; first_ref.n_particles()]);
    let frame_stride = (sorted_ts.len() / 48).max(1);
    let mut snapshot_frames = Vec::new();
    for (row_idx, &ts) in sorted_ts.iter().enumerate() {
        if row_idx % frame_stride != 0 && ts != ts_min && ts != start_index && ts != ts_max {
            continue;
        }
        let f = &frames[&ts];
        let pt_stride = (f.n_particles() / 700).max(1);
        let state_row = &hom_res.states_matrix[row_idx];
        let mut sub_coords = Vec::new();
        let mut sub_vels = Vec::new();
        let mut sub_diams = Vec::new();
        let mut sub_small = Vec::new();
        let mut sub_cids = Vec::new();
        let mut pop = vec![0usize; n_cells];
        let mut pop_s = vec![0usize; n_cells];
        let mut pop_l = vec![0usize; n_cells];
        let mut n_small_total = 0usize;
        let mut n_large_total = 0usize;

        for (i, &cid) in state_row.iter().enumerate() {
            let is_s = *sp_small_mask.get(i).unwrap_or(&true);
            if is_s {
                n_small_total += 1;
            } else {
                n_large_total += 1;
            }
            if cid >= 0 && (cid as usize) < n_cells {
                pop[cid as usize] += 1;
                if is_s {
                    pop_s[cid as usize] += 1;
                } else {
                    pop_l[cid as usize] += 1;
                }
            }
            if i % pt_stride == 0 {
                sub_coords.push(f.coordinates[i]);
                sub_vels.push(*f.velocities.get(i).unwrap_or(&[0.0; 3]));
                sub_diams.push(
                    *f.diameters
                        .get(i)
                        .unwrap_or(if is_s { &0.004 } else { &0.008 }),
                );
                sub_small.push(is_s);
                sub_cids.push(cid);
            }
        }
        let cell_conc_dem: Vec<f64> = (0..n_cells)
            .map(|c| {
                if pop[c] > 0 {
                    let num = if target_is_large { pop_l[c] } else { pop_s[c] };
                    num as f64 / pop[c] as f64
                } else {
                    0.5
                }
            })
            .collect();

        snapshot_frames.push(SnapshotFrame3D {
            timestep: ts,
            coords: sub_coords,
            velocities: sub_vels,
            diameters: sub_diams,
            is_small: sub_small,
            cell_ids: sub_cids,
            cell_pop: pop,
            cell_conc_dem,
            n_small_total,
            n_large_total,
            n_total: f.n_particles(),
        });
    }

    // Cell teneur trajectories along times_eval (target_species / total)
    let t_row_map: HashMap<i64, usize> =
        times_i64.iter().enumerate().map(|(r, &t)| (t, r)).collect();
    let mut dem_conc_traj = Vec::with_capacity(times_eval.len());
    let mut hom_conc_traj = Vec::with_capacity(times_eval.len());
    let mut inhom_conc_traj = Vec::with_capacity(times_eval.len());
    for (ei, &t) in times_eval.iter().enumerate() {
        let dr = *t_row_map.get(&t).unwrap_or(&0);
        let ir = ei.min(s_mk_inh_s.rows.saturating_sub(1));
        if target_is_large {
            dem_conc_traj.push(cell_teneur_row(hl.s_matrix.row(dr), hs.s_matrix.row(dr)));
            hom_conc_traj.push(cell_teneur_row(s_mk_hom_l.row(ei), s_mk_hom_s.row(ei)));
            inhom_conc_traj.push(cell_teneur_row(s_mk_inh_l.row(ir), s_mk_inh_s.row(ir)));
        } else {
            dem_conc_traj.push(cell_teneur_row(hs.s_matrix.row(dr), hl.s_matrix.row(dr)));
            hom_conc_traj.push(cell_teneur_row(s_mk_hom_s.row(ei), s_mk_hom_l.row(ei)));
            inhom_conc_traj.push(cell_teneur_row(s_mk_inh_s.row(ir), s_mk_inh_l.row(ir)));
        }
    }

    // Top-9 populated active cells + their population trajectories (`fig_compare_teneur` / `fig_states_top_populated`)
    let mut active_cell_ids: Vec<usize> = (0..n_cells)
        .filter(|&c| *act_s.get(c).unwrap_or(&false) || *act_l.get(c).unwrap_or(&false))
        .collect();
    if active_cell_ids.is_empty() {
        active_cell_ids = (0..n_cells).collect();
    }
    let mut cell_mean_occ: Vec<(usize, f64)> = active_cell_ids
        .iter()
        .map(|&c| {
            let sum: f64 = times_eval
                .iter()
                .map(|&t| {
                    let dr = *t_row_map.get(&t).unwrap_or(&0);
                    hs.s_matrix.get(dr, c) + hl.s_matrix.get(dr, c)
                })
                .sum();
            (c, sum / times_eval.len().max(1) as f64)
        })
        .collect();
    cell_mean_occ.sort_by(|a, b| b.1.total_cmp(&a.1));
    let top_cells: Vec<usize> = cell_mean_occ.into_iter().take(9).map(|(c, _)| c).collect();

    let mut top_cell_pop_dem = Vec::with_capacity(times_eval.len());
    let mut top_cell_pop_hom = Vec::with_capacity(times_eval.len());
    let mut top_cell_pop_inhom = Vec::with_capacity(times_eval.len());
    for (ei, &t) in times_eval.iter().enumerate() {
        let dr = *t_row_map.get(&t).unwrap_or(&0);
        let ir = ei.min(s_mk_inh_s.rows.saturating_sub(1));
        top_cell_pop_dem.push(
            top_cells
                .iter()
                .map(|&c| hs.s_matrix.get(dr, c) + hl.s_matrix.get(dr, c))
                .collect(),
        );
        top_cell_pop_hom.push(
            top_cells
                .iter()
                .map(|&c| s_mk_hom_s.get(ei, c) + s_mk_hom_l.get(ei, c))
                .collect(),
        );
        top_cell_pop_inhom.push(
            top_cells
                .iter()
                .map(|&c| s_mk_inh_s.get(ir, c) + s_mk_inh_l.get(ir, c))
                .collect(),
        );
    }

    // Discrepancy heatmaps (Temps x Cellules) matching fig_discrepancy_analysis
    let (heatmap_times, discrepancy_heatmap_hom) = build_discrepancy_heatmap(
        if target_is_large {
            &hl.s_matrix
        } else {
            &hs.s_matrix
        },
        if target_is_large {
            &s_mk_hom_l
        } else {
            &s_mk_hom_s
        },
        times_i64,
        &times_eval,
        &active_cell_ids,
    );
    let (_, discrepancy_heatmap_inhom) = build_discrepancy_heatmap(
        if target_is_large {
            &il_res.s_matrix
        } else {
            &is_res.s_matrix
        },
        if target_is_large {
            &s_mk_inh_l
        } else {
            &s_mk_inh_s
        },
        times_i64,
        &inh_times_eval,
        &active_cell_ids,
    );

    // Collect top off-diagonal transition edges for 3-D flow arrows (max 120 strongest edges)
    let edge_mat = if target_is_large {
        &pl_clean
    } else {
        &ps_clean
    };
    let mut edges_s = Vec::new();
    for i in 0..edge_mat.rows {
        for j in 0..edge_mat.cols {
            if i == j {
                continue;
            }
            let p = edge_mat.get(i, j);
            if p >= 0.04 {
                edges_s.push((i, j, p));
            }
        }
    }
    edges_s.sort_by(|a, b| b.2.total_cmp(&a.2));
    edges_s.truncate(120);

    let _ = stationary_distribution(&ps_clean);

    Ok(ComparisonBundle {
        dataset_name: req.dataset.clone(),
        method: req.method.clone(),
        label: mesh.label.clone(),
        target_species: req.target_species.clone(),
        is_single_species: is_single,
        dem_t_min: ts_min as i64,
        dem_t_max: ts_max as i64,
        dem_total_timesteps: sorted_ts.len(),
        model_t_start,
        model_t_end,
        pred_t_start,
        pred_t_end,
        mesh,
        diagnostics,
        hom_metrics,
        inhom_metrics,
        p_hom_small: p_hom_s.clone(),
        p_hom_large: p_hom_l.clone(),
        blocks,
        block_rel_diffs,
        p_laws,
        frames: snapshot_frames,
        hom_conc_traj,
        inhom_conc_traj,
        dem_conc_traj,
        top_cells,
        top_cell_pop_dem,
        top_cell_pop_hom,
        top_cell_pop_inhom,
        active_cell_ids,
        heatmap_times,
        discrepancy_heatmap_hom,
        discrepancy_heatmap_inhom,
        top_edges_small: edges_s,
    })
}

/// Evaluate multiple partitioning methods side-by-side on the exact same DEM dataset
/// and temporal settings (`POST /api/benchmark`).
pub fn run_method_benchmark(
    base_req: &SimulationRequest,
    methods: &[String],
    frames: &BTreeMap<usize, DemFrame>,
) -> Vec<MethodBenchmarkRow> {
    let mut rows = Vec::with_capacity(methods.len());
    for m in methods {
        let mut r = base_req.clone();
        r.method = m.clone();
        r.kwargs = if m == &base_req.method {
            base_req.kwargs.clone()
        } else {
            SimulationRequest::default_kwargs_for(m)
        };
        if let Ok(b) = run_gui_comparison(&r, frames) {
            let n_dem_mean = if b.hom_metrics.n_particles_total_dem.is_empty() {
                0.0
            } else {
                b.hom_metrics.n_particles_total_dem.iter().sum::<f64>()
                    / b.hom_metrics.n_particles_total_dem.len() as f64
            };
            rows.push(MethodBenchmarkRow {
                method: m.clone(),
                n_cells: b.mesh.n_cells,
                n_visited: b.diagnostics.n_visited,
                n_particles_dem_mean: n_dem_mean,
                n_particles_hom_final: b
                    .hom_metrics
                    .n_particles_total_mk
                    .last()
                    .copied()
                    .unwrap_or(0.0),
                n_particles_inhom_final: b
                    .inhom_metrics
                    .n_particles_total_mk
                    .last()
                    .copied()
                    .unwrap_or(0.0),
                hom_rmse_teneur: b.hom_metrics.rmse_teneur,
                inhom_rmse_teneur: b.inhom_metrics.rmse_teneur,
                hom_mae_teneur: b.hom_metrics.mae_teneur,
                inhom_mae_teneur: b.inhom_metrics.mae_teneur,
                hom_max_teneur: b.hom_metrics.max_err_teneur,
                inhom_max_teneur: b.inhom_metrics.max_err_teneur,
                hom_rmse_rsd: b.hom_metrics.rmse_rsd,
                inhom_rmse_rsd: b.inhom_metrics.rmse_rsd,
                hom_l1_mean: b.hom_metrics.l1_mean_target,
                inhom_l1_mean: b.inhom_metrics.l1_mean_target,
                hom_rms_disc: b.hom_metrics.rms_discrepancy_mean,
                inhom_rms_disc: b.inhom_metrics.rms_discrepancy_mean,
                lambda_max_hom: b.hom_metrics.lambda_max_small,
                diag_mean_hom: b.hom_metrics.diag_mean_small,
                t50_dem: b.hom_metrics.t50_dem,
                t50_hom: b.hom_metrics.t50_markov,
                t50_inhom: b.inhom_metrics.t50_markov,
            });
        }
    }
    rows
}

fn matrix_to_json(m: &Matrix) -> JsonValue {
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

fn vec_f64_json(v: &[f64]) -> JsonValue {
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

impl ComparisonBundle {
    pub fn to_json(&self) -> JsonValue {
        let p3 = |v: &[f64; 3]| {
            JsonValue::Array(vec![
                JsonValue::Number(v[0]),
                JsonValue::Number(v[1]),
                JsonValue::Number(v[2]),
            ])
        };
        let mut root = JsonValue::object();
        root.insert("dataset_name", JsonValue::String(self.dataset_name.clone()));
        root.insert("method", JsonValue::String(self.method.clone()));
        root.insert("label", JsonValue::String(self.label.clone()));
        root.insert(
            "target_species",
            JsonValue::String(self.target_species.clone()),
        );
        root.insert("is_single_species", JsonValue::Bool(self.is_single_species));
        root.insert("dem_t_min", JsonValue::Number(self.dem_t_min as f64));
        root.insert("dem_t_max", JsonValue::Number(self.dem_t_max as f64));
        root.insert(
            "dem_total_timesteps",
            JsonValue::Number(self.dem_total_timesteps as f64),
        );
        root.insert(
            "model_t_start",
            JsonValue::Number(self.model_t_start as f64),
        );
        root.insert("model_t_end", JsonValue::Number(self.model_t_end as f64));
        root.insert("pred_t_start", JsonValue::Number(self.pred_t_start as f64));
        root.insert("pred_t_end", JsonValue::Number(self.pred_t_end as f64));
        root.insert("mesh", self.mesh.to_json());

        let mut diag = JsonValue::object();
        diag.insert(
            "pop_min",
            JsonValue::Number(self.diagnostics.pop_min as f64),
        );
        diag.insert(
            "pop_max",
            JsonValue::Number(self.diagnostics.pop_max as f64),
        );
        diag.insert("pop_mean", JsonValue::Number(self.diagnostics.pop_mean));
        diag.insert("pop_std", JsonValue::Number(self.diagnostics.pop_std));
        diag.insert(
            "n_empty",
            JsonValue::Number(self.diagnostics.n_empty as f64),
        );
        diag.insert(
            "n_visited",
            JsonValue::Number(self.diagnostics.n_visited as f64),
        );
        diag.insert(
            "fraction_visited",
            JsonValue::Number(self.diagnostics.fraction_visited),
        );
        root.insert("diagnostics", diag);

        root.insert("hom_metrics", self.hom_metrics.to_json());
        root.insert("inhom_metrics", self.inhom_metrics.to_json());
        root.insert("p_hom_small", matrix_to_json(&self.p_hom_small));
        root.insert("p_hom_large", matrix_to_json(&self.p_hom_large));

        let blocks_json: Vec<JsonValue> = self
            .blocks
            .iter()
            .map(|b| {
                let mut o = JsonValue::object();
                o.insert("block_index", JsonValue::Number(b.block_index as f64));
                o.insert("t_start", JsonValue::Number(b.t_start as f64));
                o.insert("t_end", JsonValue::Number(b.t_end as f64));
                o.insert(
                    "n_visited_small",
                    JsonValue::Number(b.n_visited_small as f64),
                );
                o.insert("diag_mean_small", JsonValue::Number(b.diag_mean_small));
                o.insert("lambda_max_small", JsonValue::Number(b.lambda_max_small));
                o.insert("p_small", matrix_to_json(&b.p_small));
                o.insert("p_large", matrix_to_json(&b.p_large));
                o
            })
            .collect();
        root.insert("blocks", JsonValue::Array(blocks_json));
        root.insert("block_rel_diffs", vec_f64_json(&self.block_rel_diffs));
        root.insert(
            "p_laws",
            JsonValue::Array(
                self.p_laws
                    .iter()
                    .map(|l| {
                        let mut o = JsonValue::object();
                        o.insert("src", JsonValue::Number(l.src as f64));
                        o.insert("dst", JsonValue::Number(l.dst as f64));
                        o.insert("block_times", vec_f64_json(&l.block_times));
                        o.insert("values", vec_f64_json(&l.values));
                        o.insert("predicted", vec_f64_json(&l.predicted));
                        o.insert("degree", JsonValue::Number(l.degree as f64));
                        o.insert("rmse", JsonValue::Number(l.rmse));
                        o.insert("r2", JsonValue::Number(l.r2));
                        o.insert("law_label", JsonValue::String(l.law_label.clone()));
                        o
                    })
                    .collect(),
            ),
        );

        let frames_json: Vec<JsonValue> = self
            .frames
            .iter()
            .map(|f| {
                let mut o = JsonValue::object();
                o.insert("timestep", JsonValue::Number(f.timestep as f64));
                o.insert(
                    "coords",
                    JsonValue::Array(f.coords.iter().map(p3).collect()),
                );
                o.insert(
                    "velocities",
                    JsonValue::Array(f.velocities.iter().map(p3).collect()),
                );
                o.insert("diameters", vec_f64_json(&f.diameters));
                o.insert(
                    "is_small",
                    JsonValue::Array(f.is_small.iter().map(|&b| JsonValue::Bool(b)).collect()),
                );
                o.insert(
                    "cell_ids",
                    JsonValue::Array(
                        f.cell_ids
                            .iter()
                            .map(|&id| JsonValue::Number(id as f64))
                            .collect(),
                    ),
                );
                o.insert(
                    "cell_pop",
                    JsonValue::Array(
                        f.cell_pop
                            .iter()
                            .map(|&p| JsonValue::Number(p as f64))
                            .collect(),
                    ),
                );
                o.insert("cell_conc_dem", vec_f64_json(&f.cell_conc_dem));
                o.insert("n_small_total", JsonValue::Number(f.n_small_total as f64));
                o.insert("n_large_total", JsonValue::Number(f.n_large_total as f64));
                o.insert("n_total", JsonValue::Number(f.n_total as f64));
                o
            })
            .collect();
        root.insert("frames", JsonValue::Array(frames_json));

        root.insert(
            "hom_conc_traj",
            JsonValue::Array(self.hom_conc_traj.iter().map(|r| vec_f64_json(r)).collect()),
        );
        root.insert(
            "inhom_conc_traj",
            JsonValue::Array(
                self.inhom_conc_traj
                    .iter()
                    .map(|r| vec_f64_json(r))
                    .collect(),
            ),
        );
        root.insert(
            "dem_conc_traj",
            JsonValue::Array(self.dem_conc_traj.iter().map(|r| vec_f64_json(r)).collect()),
        );
        root.insert(
            "top_cells",
            JsonValue::Array(
                self.top_cells
                    .iter()
                    .map(|&c| JsonValue::Number(c as f64))
                    .collect(),
            ),
        );
        root.insert(
            "top_cell_pop_dem",
            JsonValue::Array(
                self.top_cell_pop_dem
                    .iter()
                    .map(|r| vec_f64_json(r))
                    .collect(),
            ),
        );
        root.insert(
            "top_cell_pop_hom",
            JsonValue::Array(
                self.top_cell_pop_hom
                    .iter()
                    .map(|r| vec_f64_json(r))
                    .collect(),
            ),
        );
        root.insert(
            "top_cell_pop_inhom",
            JsonValue::Array(
                self.top_cell_pop_inhom
                    .iter()
                    .map(|r| vec_f64_json(r))
                    .collect(),
            ),
        );
        root.insert(
            "active_cell_ids",
            JsonValue::Array(
                self.active_cell_ids
                    .iter()
                    .map(|&c| JsonValue::Number(c as f64))
                    .collect(),
            ),
        );
        root.insert(
            "heatmap_times",
            JsonValue::Array(
                self.heatmap_times
                    .iter()
                    .map(|&t| JsonValue::Number(t as f64))
                    .collect(),
            ),
        );
        root.insert(
            "discrepancy_heatmap_hom",
            JsonValue::Array(
                self.discrepancy_heatmap_hom
                    .iter()
                    .map(|r| vec_f64_json(r))
                    .collect(),
            ),
        );
        root.insert(
            "discrepancy_heatmap_inhom",
            JsonValue::Array(
                self.discrepancy_heatmap_inhom
                    .iter()
                    .map(|r| vec_f64_json(r))
                    .collect(),
            ),
        );
        root.insert(
            "top_edges_small",
            JsonValue::Array(
                self.top_edges_small
                    .iter()
                    .map(|&(i, j, p)| {
                        JsonValue::Array(vec![
                            JsonValue::Number(i as f64),
                            JsonValue::Number(j as f64),
                            JsonValue::Number(p),
                        ])
                    })
                    .collect(),
            ),
        );
        root
    }
}
