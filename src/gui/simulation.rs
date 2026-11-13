//! Pure-Rust interactive simulation runner for 3-D mesh extraction,
//! full-duration DEM prediction, and `postprocess.py` / `postprocess_inhomogeneous.py`
//! metric & error comparison (teneur, particle counts, L1, RMS discrepancy,
//! RSD, entropy, Danckwerts, mixing times, inter-block diffs & p_ij(t) laws).

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::data::{DataSource, DemFrame, LocalDataSource};
use crate::error::{DemMcmError, Result};
use crate::gui::matrix_evolution::{
    compute_matrix_evolution, MatrixEvolutionData, TransitionLawItem,
};
use crate::gui::mesh3d::Mesh3DData;
use crate::gui::metrics_eval::{
    build_discrepancy_heatmap, cell_teneur_row, matrix_to_json, vec_f64_json,
    ChainPredictionMetrics, MethodBenchmarkRow,
};
use crate::gui::state::SimulationRequest;
use crate::io::json::JsonValue;
use crate::linalg::{real_eigvals_abs, Matrix};
use crate::metrics::stationary_distribution;
use crate::partitioners::PartitionerDiagnostics;
use crate::research::session::{ResearchSessionCache, TemporalEvalParams};
use crate::run_sweep::detect_species;

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
    pub matrix_evolution_small: MatrixEvolutionData,
    pub matrix_evolution_large: MatrixEvolutionData,
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

/// Run partitioner fitting + 3-D mesh extraction + homogeneous & inhomogeneous
/// model construction + **full-duration DEM prediction** (`ts_min .. ts_max`),
/// without external session caching.
pub fn run_gui_comparison(
    req: &SimulationRequest,
    frames: &BTreeMap<usize, DemFrame>,
) -> Result<ComparisonBundle> {
    let ephemeral = ResearchSessionCache::new();
    run_gui_comparison_cached(req, frames, &ephemeral)
}

/// Cached variant sharing [`ResearchSessionCache`] across interactive runs and
/// parametric research studies so temporal parameter updates skip re-fitting
/// and re-labelling `states_matrix`.
pub fn run_gui_comparison_cached(
    req: &SimulationRequest,
    frames: &BTreeMap<usize, DemFrame>,
    session_cache: &ResearchSessionCache,
) -> Result<ComparisonBundle> {
    if frames.is_empty() {
        return Err(DemMcmError::MissingData(
            "Empty DEM frame collection".to_string(),
        ));
    }

    let session = session_cache.get_or_build(
        &req.dataset,
        &req.method,
        &req.kwargs,
        req.permanent_start,
        false,
        frames,
    )?;

    let sorted_ts = &session.sorted_timesteps;
    let ts_min = session.ts_min;
    let ts_max = session.ts_max;
    let start_index = if session.idx_to_row.contains_key(&req.start_index) {
        req.start_index
    } else {
        ts_min
    };
    let nlt = req.nlt.max(1);
    let step = req.step.max(1);
    let dt = req.dt.max(1);
    let tau = req.tau.max(1);
    let model_t_start = start_index as i64;
    let model_t_end = (start_index + nlt * (step + tau)).min(ts_max) as i64;

    let eval_params = TemporalEvalParams {
        nlt,
        step,
        dt,
        tau,
        start_index,
        no_species: req.no_species,
        target_species: req.target_species.clone(),
        predict_full_duration: req.predict_full_duration,
        build_inhomogeneous: true,
    };
    let ev = session.evaluate_temporal(&eval_params)?;

    let is_single = ev.is_single_species;
    let target_is_large = !is_single && req.target_species == "large";
    let times_i64 = &session.times_i64;
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

    let s_map = if req.no_species {
        &session.s_matrices_all
    } else {
        &session.s_matrices_species
    };
    let s_dem_s = &s_map[&ev.sp_a_key];
    let s_dem_l = &s_map[&ev.sp_b_key];

    let hom_metrics = ev.hom_metrics;
    let times_eval = hom_metrics.times_eval.clone();
    let inhom_metrics = ev
        .inhom_metrics
        .unwrap_or_else(|| hom_metrics.clone());
    let inh_times_eval = inhom_metrics.times_eval.clone();

    let s_mk_hom_s = ev.s_mk_hom_small;
    let s_mk_hom_l = ev.s_mk_hom_large;
    let s_mk_inh_s = ev.s_mk_inhom_small.unwrap_or_else(|| s_mk_hom_s.clone());
    let s_mk_inh_l = ev.s_mk_inhom_large.unwrap_or_else(|| s_mk_hom_l.clone());

    let p_hom_s = ev.p_hom_raw_small;
    let p_hom_l = ev.p_hom_raw_large;
    let ps_clean = ev.p_hom_clean_small;
    let pl_clean = ev.p_hom_clean_large;
    let act_s = ev.activated_small;
    let act_l = ev.activated_large;

    let raw_blocks_s = if ev.p_blocks_raw_small.is_empty() {
        vec![p_hom_s.clone(); nlt]
    } else {
        ev.p_blocks_raw_small
    };
    let raw_blocks_l = if ev.p_blocks_raw_large.is_empty() {
        vec![p_hom_l.clone(); nlt]
    } else {
        ev.p_blocks_raw_large
    };
    let clean_blocks_s = if ev.p_blocks_clean_small.is_empty() {
        vec![ps_clean.clone(); nlt]
    } else {
        ev.p_blocks_clean_small
    };
    let clean_blocks_l = if ev.p_blocks_clean_large.is_empty() {
        vec![pl_clean.clone(); nlt]
    } else {
        ev.p_blocks_clean_large
    };

    // Per-block summaries + consecutive block relative diffs (fig_matrix_differences)
    let mut blocks = Vec::with_capacity(raw_blocks_s.len());
    let mut block_times_f64 = Vec::with_capacity(raw_blocks_s.len());
    for (b, bs) in raw_blocks_s.iter().enumerate() {
        let bl = &raw_blocks_l[b];
        let t0 = ev
            .block_times
            .get(b)
            .map(|&v| v as i64)
            .unwrap_or(start_index as i64 + (b * (step + tau)) as i64);
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

    let block_labels: Vec<String> = if ev.block_labels.len() == blocks.len() {
        ev.block_labels
    } else {
        blocks
            .iter()
            .map(|b| format!("B{} [{}..{}]", b.block_index, b.t_start, b.t_end))
            .collect()
    };

    let matrix_evolution_small = compute_matrix_evolution(
        &clean_blocks_s,
        &block_times_f64,
        &block_labels,
        &ev.sp_a_key,
        0.01,
        2,
        400,
    );
    let matrix_evolution_large = compute_matrix_evolution(
        &clean_blocks_l,
        &block_times_f64,
        &block_labels,
        &ev.sp_b_key,
        0.01,
        2,
        400,
    );

    let block_rel_diffs = if target_is_large {
        matrix_evolution_large.delta_norms_l1.clone()
    } else {
        matrix_evolution_small.delta_norms_l1.clone()
    };

    let target_evol = if target_is_large {
        &matrix_evolution_large
    } else {
        &matrix_evolution_small
    };
    let p_laws: Vec<TransitionLawItem> = target_evol
        .components
        .iter()
        .filter(|c| !c.is_diagonal && c.max_prob > 0.02)
        .take(8)
        .cloned()
        .collect();

    let mesh = session.mesh.clone();
    let diagnostics = session.diagnostics.clone();

    // Downsampled 3-D playback frames across the full DEM duration (up to 48 steps)
    let n_cells = session.n_cells;
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
        let state_row = &session.states_matrix[row_idx];
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
            dem_conc_traj.push(cell_teneur_row(s_dem_l.row(dr), s_dem_s.row(dr)));
            hom_conc_traj.push(cell_teneur_row(s_mk_hom_l.row(ei), s_mk_hom_s.row(ei)));
            inhom_conc_traj.push(cell_teneur_row(s_mk_inh_l.row(ir), s_mk_inh_s.row(ir)));
        } else {
            dem_conc_traj.push(cell_teneur_row(s_dem_s.row(dr), s_dem_l.row(dr)));
            hom_conc_traj.push(cell_teneur_row(s_mk_hom_s.row(ei), s_mk_hom_l.row(ei)));
            inhom_conc_traj.push(cell_teneur_row(s_mk_inh_s.row(ir), s_mk_inh_l.row(ir)));
        }
    }

    // Top-9 populated active cells + their population trajectories
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
                    s_dem_s.get(dr, c) + s_dem_l.get(dr, c)
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
                .map(|&c| s_dem_s.get(dr, c) + s_dem_l.get(dr, c))
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

    // Discrepancy heatmaps (Temps x Cellules)
    let (heatmap_times, discrepancy_heatmap_hom) = build_discrepancy_heatmap(
        if target_is_large { s_dem_l } else { s_dem_s },
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
        if target_is_large { s_dem_l } else { s_dem_s },
        if target_is_large {
            &s_mk_inh_l
        } else {
            &s_mk_inh_s
        },
        times_i64,
        &inh_times_eval,
        &active_cell_ids,
    );

    // Top off-diagonal transition edges for 3-D flow arrows (max 120 strongest edges)
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
        p_hom_small: p_hom_s,
        p_hom_large: p_hom_l,
        blocks,
        block_rel_diffs,
        p_laws,
        matrix_evolution_small,
        matrix_evolution_large,
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
    let ephemeral = ResearchSessionCache::new();
    run_method_benchmark_cached(base_req, methods, frames, &ephemeral)
}

pub fn run_method_benchmark_cached(
    base_req: &SimulationRequest,
    methods: &[String],
    frames: &BTreeMap<usize, DemFrame>,
    session_cache: &ResearchSessionCache,
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
        if let Ok(b) = run_gui_comparison_cached(&r, frames, session_cache) {
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
            JsonValue::Array(self.p_laws.iter().map(|l| l.to_json()).collect()),
        );
        root.insert(
            "matrix_evolution_small",
            self.matrix_evolution_small.to_json(),
        );
        root.insert(
            "matrix_evolution_large",
            self.matrix_evolution_large.to_json(),
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
