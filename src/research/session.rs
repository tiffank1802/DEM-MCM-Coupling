//! Spatial/Temporal decoupling & cached research session (`PreparedMethodSession`).
//!
//! Mirrors and generalizes the `EtudeMethode` pattern from
//! `postprocessing/etudes_librairie.py`: spatial partitioning (`partitioner.fit`,
//! `compute_state_matrices`, `build_s_matrices`) is computed **once** per
//! `(dataset, method, canonical_kwargs, use_bed_frame)` and cached in `Arc`.
//! Any subsequent sweep over `(start_index, nlt, step, dt, tau, no_species, pred_start)`
//! reuses `states_matrix` and `s_matrices` in milliseconds without re-fitting.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::config::{ExperimentData, SpeciesExperimentData};
use crate::data::DemFrame;
use crate::error::{DemMcmError, Result};
use crate::gui::mesh3d::{extract_mesh_3d, Mesh3DData};
use crate::gui::metrics_eval::{evaluate_chain_metrics, ChainEvalInput, ChainPredictionMetrics};
use crate::io::json::JsonValue;
use crate::linalg::{covariance_matrix, jacobi_eigh, real_eigvals_abs, Matrix};
use crate::metrics::{
    clean_transition_matrix, propagate_markov, propagate_markov_inhomogeneous, validate_experiment,
};
use crate::partitioners::{create_partitioner, PartitionerDiagnostics};
use crate::run_sweep::{
    build_inhomogeneous_blocks, build_pairs, build_s_matrices, compute_p_matrix,
    compute_state_matrices, detect_species, ExperimentConfig,
};

/// Canonical key identifying a spatial partitioning session.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SessionKey {
    pub dataset_name: String,
    pub method: String,
    pub canonical_kwargs: String,
    pub permanent_start: usize,
    pub use_bed_frame: bool,
}

impl SessionKey {
    pub fn new(
        dataset_name: impl Into<String>,
        method: impl Into<String>,
        kwargs: &HashMap<String, JsonValue>,
        permanent_start: usize,
        use_bed_frame: bool,
    ) -> Self {
        Self {
            dataset_name: dataset_name.into(),
            method: method.into(),
            canonical_kwargs: canonicalize_kwargs(kwargs),
            permanent_start,
            use_bed_frame,
        }
    }
}

pub fn canonicalize_json(val: &JsonValue) -> String {
    match val {
        JsonValue::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let parts: Vec<String> = keys
                .into_iter()
                .map(|k| format!("\"{k}\":{}", canonicalize_json(&map[k])))
                .collect();
            format!("{{{}}}", parts.join(","))
        }
        JsonValue::Array(arr) => {
            let parts: Vec<String> = arr.iter().map(canonicalize_json).collect();
            format!("[{}]", parts.join(","))
        }
        _ => val.to_string_compact(),
    }
}

pub fn canonicalize_kwargs(kwargs: &HashMap<String, JsonValue>) -> String {
    let mut keys: Vec<&String> = kwargs.keys().collect();
    keys.sort();
    let parts: Vec<String> = keys
        .into_iter()
        .map(|k| format!("\"{k}\":{}", canonicalize_json(&kwargs[k])))
        .collect();
    format!("{{{}}}", parts.join(","))
}

/// Optional orthogonal change-of-frame aligned with the particle bed principal axes.
#[derive(Debug, Clone)]
pub struct BedFrameBasis {
    pub origin: [f64; 3],
    /// Columns are unit basis vectors in row-major `[3 x 3]`.
    pub axes: [[f64; 3]; 3],
}

impl BedFrameBasis {
    pub fn fit_from_coords(coords: &[[f64; 3]]) -> Self {
        if coords.len() < 4 {
            return Self {
                origin: [0.0; 3],
                axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            };
        }
        let n = coords.len() as f64;
        let mut origin = [0.0; 3];
        for c in coords {
            origin[0] += c[0];
            origin[1] += c[1];
            origin[2] += c[2];
        }
        origin[0] /= n;
        origin[1] /= n;
        origin[2] /= n;

        let centered = Matrix::from_vec3(coords);
        let Ok(cov) = covariance_matrix(&centered) else {
            return Self {
                origin,
                axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            };
        };
        let Ok((_, vecs)) = jacobi_eigh(&cov) else {
            return Self {
                origin,
                axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            };
        };
        // jacobi_eigh returns ascending eigenvalues; column 2 has largest variance, col 0 smallest.
        let e0 = [vecs.get(0, 2), vecs.get(1, 2), vecs.get(2, 2)];
        let e1 = [vecs.get(0, 1), vecs.get(1, 1), vecs.get(2, 1)];
        // Right-handed cross product e0 x e1
        let e2 = [
            e0[1] * e1[2] - e0[2] * e1[1],
            e0[2] * e1[0] - e0[0] * e1[2],
            e0[0] * e1[1] - e0[1] * e1[0],
        ];
        Self {
            origin,
            axes: [e0, e1, e2],
        }
    }

    pub fn project_point(&self, p: [f64; 3]) -> [f64; 3] {
        let d = [
            p[0] - self.origin[0],
            p[1] - self.origin[1],
            p[2] - self.origin[2],
        ];
        [
            d[0] * self.axes[0][0] + d[1] * self.axes[0][1] + d[2] * self.axes[0][2],
            d[0] * self.axes[1][0] + d[1] * self.axes[1][1] + d[2] * self.axes[1][2],
            d[0] * self.axes[2][0] + d[1] * self.axes[2][1] + d[2] * self.axes[2][2],
        ]
    }

    pub fn project_vec(&self, v: [f64; 3]) -> [f64; 3] {
        [
            v[0] * self.axes[0][0] + v[1] * self.axes[0][1] + v[2] * self.axes[0][2],
            v[0] * self.axes[1][0] + v[1] * self.axes[1][1] + v[2] * self.axes[1][2],
            v[0] * self.axes[2][0] + v[1] * self.axes[2][1] + v[2] * self.axes[2][2],
        ]
    }
}

/// Cached spatial partitioning session ready for instant multi-parameter temporal sweeps.
#[derive(Debug, Clone)]
pub struct PreparedMethodSession {
    pub key: SessionKey,
    pub label: String,
    pub n_cells: usize,
    pub ts_min: usize,
    pub ts_max: usize,
    pub sorted_timesteps: Vec<usize>,
    pub times_i64: Vec<i64>,
    pub idx_to_row: HashMap<usize, usize>,
    pub states_matrix: Vec<Vec<i64>>,
    pub species_masks: Vec<(String, Vec<bool>)>,
    pub all_mask: Vec<(String, Vec<bool>)>,
    pub s_matrices_species: HashMap<String, Matrix>,
    pub s_matrices_all: HashMap<String, Matrix>,
    pub mesh: Mesh3DData,
    pub diagnostics: PartitionerDiagnostics,
}

/// Temporal evaluation configuration evaluated on a cached [`PreparedMethodSession`].
#[derive(Debug, Clone)]
pub struct TemporalEvalParams {
    pub nlt: usize,
    pub step: usize,
    pub dt: usize,
    pub tau: usize,
    pub start_index: usize,
    pub no_species: bool,
    pub target_species: String,
    pub predict_full_duration: bool,
    pub build_inhomogeneous: bool,
}

/// Result of one fast temporal evaluation against a [`PreparedMethodSession`].
#[derive(Debug, Clone)]
pub struct TemporalEvalOutput {
    pub start_index: usize,
    pub nlt: usize,
    pub step: usize,
    pub dt: usize,
    pub tau: usize,
    pub n_pairs_used: usize,
    pub n_visited_hom: usize,
    pub nan_rows_hom: usize,
    pub nan_rows_inhom_mean: f64,
    pub diag_mean_raw: f64,
    pub lambda_max_hom: f64,
    pub hom_metrics: ChainPredictionMetrics,
    pub inhom_metrics: Option<ChainPredictionMetrics>,
    pub p_hom_raw_small: Matrix,
    pub p_hom_raw_large: Matrix,
    pub p_hom_clean_small: Matrix,
    pub p_hom_clean_large: Matrix,
    pub activated_small: Vec<bool>,
    pub activated_large: Vec<bool>,
    pub p_blocks_raw_small: Vec<Matrix>,
    pub p_blocks_raw_large: Vec<Matrix>,
    pub p_blocks_clean_small: Vec<Matrix>,
    pub p_blocks_clean_large: Vec<Matrix>,
    pub block_times: Vec<f64>,
    pub block_labels: Vec<String>,
    pub is_single_species: bool,
    pub sp_a_key: String,
    pub sp_b_key: String,
    pub s_mk_hom_small: Matrix,
    pub s_mk_hom_large: Matrix,
    pub s_mk_inhom_small: Option<Matrix>,
    pub s_mk_inhom_large: Option<Matrix>,
}

impl TemporalEvalOutput {
    pub fn p_hom_raw_target(&self, target_is_large: bool) -> &Matrix {
        if target_is_large && !self.is_single_species {
            &self.p_hom_raw_large
        } else {
            &self.p_hom_raw_small
        }
    }

    pub fn p_blocks_clean_target(&self, target_is_large: bool) -> &[Matrix] {
        if target_is_large && !self.is_single_species {
            &self.p_blocks_clean_large
        } else {
            &self.p_blocks_clean_small
        }
    }
}

/// Relative Frobenius error `||P_test - P_ref||_F / ||P_ref||_F` on finite entries
/// (Doucet et al., 2008).
pub fn frobenius_relative_error(p_test: &Matrix, p_ref: &Matrix) -> f64 {
    let n = p_test.data.len().min(p_ref.data.len());
    let mut num_sq = 0.0;
    let mut den_sq = 0.0;
    for k in 0..n {
        let a = p_test.data[k];
        let b = p_ref.data[k];
        let af = if a.is_finite() { a } else { 0.0 };
        let bf = if b.is_finite() { b } else { 0.0 };
        let d = af - bf;
        num_sq += d * d;
        den_sq += bf * bf;
    }
    if den_sq <= 1e-24 {
        0.0
    } else {
        (num_sq / den_sq).sqrt()
    }
}

fn count_nan_rows(p: &Matrix) -> usize {
    (0..p.rows)
        .filter(|&r| p.row(r).iter().any(|&v| !v.is_finite()))
        .count()
}

pub fn build_prepared_session(
    dataset_name: &str,
    method: &str,
    kwargs: &HashMap<String, JsonValue>,
    permanent_start: usize,
    use_bed_frame: bool,
    frames: &BTreeMap<usize, DemFrame>,
) -> Result<PreparedMethodSession> {
    if frames.is_empty() {
        return Err(DemMcmError::MissingData(
            "Empty DEM frame collection".to_string(),
        ));
    }

    let sorted_timesteps: Vec<usize> = frames.keys().copied().collect();
    let ts_min = sorted_timesteps[0];
    let ts_max = *sorted_timesteps.last().unwrap();
    let perm_start = if permanent_start >= ts_min && permanent_start <= ts_max {
        permanent_start
    } else {
        ts_min
    };

    let perm_ts: Vec<usize> = sorted_timesteps
        .iter()
        .copied()
        .filter(|&t| t >= perm_start)
        .collect();
    let fit_ts = if perm_ts.is_empty() {
        &sorted_timesteps
    } else {
        &perm_ts
    };

    let approx_pts = frames[&fit_ts[0]].n_particles().max(1);
    let target_frames = (24_000 / approx_pts).max(1);
    let fit_stride = (fit_ts.len() / target_frames).max(1);

    let mut raw_perm_coords = Vec::new();
    let mut raw_perm_vels = Vec::new();
    for (idx, &ts) in fit_ts.iter().enumerate() {
        if idx % fit_stride != 0 && ts != ts_max {
            continue;
        }
        let f = &frames[&ts];
        raw_perm_coords.extend_from_slice(&f.coordinates);
        raw_perm_vels.extend_from_slice(&f.velocities);
    }

    let basis_opt = if use_bed_frame {
        Some(BedFrameBasis::fit_from_coords(&raw_perm_coords))
    } else {
        None
    };

    let (perm_coords, perm_vels, transformed_frames);
    let frames_ref: &BTreeMap<usize, DemFrame> = if let Some(ref basis) = basis_opt {
        perm_coords = raw_perm_coords
            .iter()
            .map(|&p| basis.project_point(p))
            .collect();
        perm_vels = raw_perm_vels
            .iter()
            .map(|&v| basis.project_vec(v))
            .collect();
        let mut tf = BTreeMap::new();
        for (&ts, fr) in frames {
            let coords: Vec<[f64; 3]> = fr
                .coordinates
                .iter()
                .map(|&p| basis.project_point(p))
                .collect();
            let vels: Vec<[f64; 3]> = fr
                .velocities
                .iter()
                .map(|&v| basis.project_vec(v))
                .collect();
            tf.insert(
                ts,
                DemFrame::with_kinematics(ts, coords, vels, fr.diameters.clone()),
            );
        }
        transformed_frames = tf;
        &transformed_frames
    } else {
        perm_coords = raw_perm_coords;
        perm_vels = raw_perm_vels;
        frames
    };

    let mut partitioner = create_partitioner(method, kwargs)?;
    if perm_vels.len() == perm_coords.len() {
        partitioner.set_dem_velocities(Some(&perm_vels));
    }
    partitioner.fit(&perm_coords)?;

    let n_cells = partitioner.n_cells();
    let (states_matrix, sorted_indices) = compute_state_matrices(frames_ref, partitioner.as_mut())?;
    let times_i64: Vec<i64> = sorted_indices.iter().map(|&t| t as i64).collect();
    let idx_to_row: HashMap<usize, usize> = sorted_indices
        .iter()
        .enumerate()
        .map(|(r, &ts)| (ts, r))
        .collect();

    let ref_frame = frames_ref.values().next().unwrap();
    let species_masks = detect_species(ref_frame);
    let all_mask = vec![("all".to_string(), vec![true; ref_frame.n_particles()])];
    let s_matrices_species = build_s_matrices(&states_matrix, &species_masks, n_cells);
    let s_matrices_all = build_s_matrices(&states_matrix, &all_mask, n_cells);

    // 3-D mesh extraction on representative permanent sample
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
    let first_frame = frames_ref
        .get(&perm_start)
        .unwrap_or_else(|| frames_ref.values().next().unwrap());
    let diagnostics =
        partitioner.diagnostics(&first_frame.coordinates, Some(&first_frame.velocities))?;

    Ok(PreparedMethodSession {
        key: SessionKey::new(dataset_name, method, kwargs, perm_start, use_bed_frame),
        label: partitioner.label(),
        n_cells,
        ts_min,
        ts_max,
        sorted_timesteps: sorted_indices,
        times_i64,
        idx_to_row,
        states_matrix,
        species_masks,
        all_mask,
        s_matrices_species,
        s_matrices_all,
        mesh,
        diagnostics,
    })
}

impl PreparedMethodSession {
    /// Evaluate homogeneous (and optionally inhomogeneous) chains for arbitrary
    /// `(start_index, nlt, step, dt, tau, no_species, predict_full_duration)` without
    /// ever re-fitting or re-labelling `states_matrix`.
    pub fn evaluate_temporal(&self, params: &TemporalEvalParams) -> Result<TemporalEvalOutput> {
        let nlt = params.nlt.max(1);
        let step = params.step.max(1);
        let dt = params.dt.max(1);
        let tau = params.tau.max(1);
        let start_index = if self.idx_to_row.contains_key(&params.start_index) {
            params.start_index
        } else {
            self.ts_min
        };

        let masks = if params.no_species {
            &self.all_mask
        } else {
            &self.species_masks
        };
        let s_map = if params.no_species {
            &self.s_matrices_all
        } else {
            &self.s_matrices_species
        };

        let mut cfg = ExperimentConfig {
            method: self.key.method.clone(),
            method_kwargs: HashMap::new(),
            nlt,
            tau,
            step,
            dt,
            start_index,
            particle_diameter: None,
            inhomogeneous: false,
        };

        let pairs = build_pairs(&cfg, self.ts_max, &self.idx_to_row);
        if pairs.is_empty() {
            return Err(DemMcmError::InvalidArgument(format!(
                "Aucune paire de transition valide pour start={start_index}, nlt={nlt}, step={step}, tau={tau}, dt={dt}"
            )));
        }

        let mut species_keys: Vec<String> = masks.iter().map(|(s, _)| s.clone()).collect();
        species_keys.sort();
        let is_single = species_keys.len() < 2;
        let (sp_a_key, sp_b_key) = if is_single {
            (species_keys[0].clone(), species_keys[0].clone())
        } else {
            let small_k = species_keys
                .iter()
                .find(|s| s.as_str() == "small")
                .unwrap_or(&species_keys[0])
                .clone();
            let large_k = species_keys
                .iter()
                .find(|s| s.as_str() == "large")
                .unwrap_or(&species_keys[1])
                .clone();
            (small_k, large_k)
        };
        let target_is_large = !is_single && params.target_species == "large";

        let compute_p_for_pairs = |pair_list: &[(usize, usize)], mask: &[bool]| -> Matrix {
            let mut prev = Vec::with_capacity(pair_list.len() * mask.len() / 2);
            let mut curr = Vec::with_capacity(pair_list.len() * mask.len() / 2);
            for &(s_idx, e_idx) in pair_list {
                let rp = self.idx_to_row[&s_idx];
                let rc = self.idx_to_row[&e_idx];
                let row_p = &self.states_matrix[rp];
                let row_c = &self.states_matrix[rc];
                for (p_idx, &keep) in mask.iter().enumerate() {
                    if keep {
                        prev.push(row_p[p_idx]);
                        curr.push(row_c[p_idx]);
                    }
                }
            }
            compute_p_matrix(&prev, &curr, self.n_cells)
        };

        let mask_a = &masks
            .iter()
            .find(|(s, _)| *s == sp_a_key)
            .unwrap()
            .1;
        let mask_b = &masks
            .iter()
            .find(|(s, _)| *s == sp_b_key)
            .unwrap()
            .1;

        let p_hom_raw_small = compute_p_for_pairs(&pairs, mask_a);
        let p_hom_raw_large = if is_single {
            p_hom_raw_small.clone()
        } else {
            compute_p_for_pairs(&pairs, mask_b)
        };

        let (p_hom_clean_small, act_s) = clean_transition_matrix(&p_hom_raw_small, 0.5);
        let (p_hom_clean_large, act_l) = clean_transition_matrix(&p_hom_raw_large, 0.5);

        let s_dem_s = &s_map[&sp_a_key];
        let s_dem_l = &s_map[&sp_b_key];

        let pred_start = if params.predict_full_duration {
            self.ts_min
        } else {
            start_index
        };
        let i_pred_row = self
            .times_i64
            .iter()
            .position(|&t| t >= pred_start as i64)
            .unwrap_or(0);
        let pred_t_start = self.times_i64[i_pred_row];

        let (s_mk_hom_small, times_eval) = propagate_markov(
            s_dem_s.row(i_pred_row),
            &p_hom_clean_small,
            &self.times_i64,
            pred_t_start,
            tau,
            &act_s,
        )?;
        let (s_mk_hom_large, _) = if is_single {
            (s_mk_hom_small.clone(), times_eval.clone())
        } else {
            propagate_markov(
                s_dem_l.row(i_pred_row),
                &p_hom_clean_large,
                &self.times_i64,
                pred_t_start,
                tau,
                &act_l,
            )?
        };

        let mut hom_exp_map = HashMap::new();
        hom_exp_map.insert(
            sp_a_key.clone(),
            SpeciesExperimentData {
                s_matrix: s_dem_s.clone(),
                times: self.times_i64.clone(),
                p: Some(p_hom_raw_small.clone()),
                p_blocks: None,
            },
        );
        if !is_single {
            hom_exp_map.insert(
                sp_b_key.clone(),
                SpeciesExperimentData {
                    s_matrix: s_dem_l.clone(),
                    times: self.times_i64.clone(),
                    p: Some(p_hom_raw_large.clone()),
                    p_blocks: None,
                },
            );
        }
        let hom_exp = ExperimentData {
            config: cfg.to_json(),
            stats: JsonValue::object(),
            inhomogeneous: false,
            inhomogeneous_metadata: None,
            states_matrix: None,
            species: hom_exp_map,
        };
        let val_hom = validate_experiment("session_hom", &hom_exp, 1e-3);
        let hom_metrics = evaluate_chain_metrics(ChainEvalInput {
            s_dem_small: s_dem_s,
            s_dem_large: s_dem_l,
            s_mk_small: &s_mk_hom_small,
            s_mk_large: &s_mk_hom_large,
            times_dem: &self.times_i64,
            times_eval: &times_eval,
            p_ref_small: &p_hom_raw_small,
            p_ref_large: &p_hom_raw_large,
            activated_small: &act_s,
            activated_large: &act_l,
            target_is_large,
            validation: &val_hom,
        })?;

        let target_raw = if target_is_large && !is_single {
            &p_hom_raw_large
        } else {
            &p_hom_raw_small
        };
        let nan_rows_hom = count_nan_rows(target_raw);
        let n_visited_hom = self.n_cells.saturating_sub(nan_rows_hom);
        let diag_vals: Vec<f64> = (0..target_raw.rows)
            .filter_map(|i| {
                let v = target_raw.get(i, i);
                if v.is_finite() {
                    Some(v)
                } else {
                    None
                }
            })
            .collect();
        let diag_mean_raw = if diag_vals.is_empty() {
            0.0
        } else {
            diag_vals.iter().sum::<f64>() / diag_vals.len() as f64
        };
        let lambda_max_hom = real_eigvals_abs(&p_hom_clean_small.transpose())
            .map(|ev| ev.into_iter().fold(0.0f64, f64::max))
            .unwrap_or(1.0);

        let mut p_blocks_raw_small = Vec::new();
        let mut p_blocks_raw_large = Vec::new();
        let mut p_blocks_clean_small = Vec::new();
        let mut p_blocks_clean_large = Vec::new();
        let mut block_times = Vec::new();
        let mut block_labels = Vec::new();
        let mut inhom_metrics = None;
        let mut s_mk_inhom_small = None;
        let mut s_mk_inhom_large = None;
        let mut nan_rows_inhom_mean = 0.0;

        if params.build_inhomogeneous {
            cfg.inhomogeneous = true;
            let raw_blocks_pairs = build_inhomogeneous_blocks(&cfg, self.ts_max, &self.idx_to_row);
            for (b_idx, bp) in raw_blocks_pairs.iter().enumerate() {
                let ps_b = compute_p_for_pairs(bp, mask_a);
                let pl_b = if is_single {
                    ps_b.clone()
                } else {
                    compute_p_for_pairs(bp, mask_b)
                };
                let t0 = bp.first().map(|p| p.0).unwrap_or(start_index) as i64;
                let t1 = bp.last().map(|p| p.1).unwrap_or(start_index + tau) as i64;
                block_times.push(t0 as f64);
                block_labels.push(format!("B{b_idx} [{t0}..{t1}]"));
                p_blocks_raw_small.push(ps_b);
                p_blocks_raw_large.push(pl_b);
            }
            if !p_blocks_raw_small.is_empty() {
                let nan_tot: usize = p_blocks_raw_small.iter().map(count_nan_rows).sum();
                nan_rows_inhom_mean = nan_tot as f64 / p_blocks_raw_small.len() as f64;
                p_blocks_clean_small =
                    stabilize_blocks(&p_blocks_raw_small, &p_hom_clean_small, &act_s);
                p_blocks_clean_large =
                    stabilize_blocks(&p_blocks_raw_large, &p_hom_clean_large, &act_l);

                let (s_inh_s, inh_eval_times) = propagate_markov_inhomogeneous(
                    s_dem_s.row(i_pred_row),
                    &p_blocks_clean_small,
                    &self.times_i64,
                    pred_t_start,
                    tau,
                    &act_s,
                    Some(step),
                    Some(nlt),
                )?;
                let (s_inh_l, _) = if is_single {
                    (s_inh_s.clone(), inh_eval_times.clone())
                } else {
                    propagate_markov_inhomogeneous(
                        s_dem_l.row(i_pred_row),
                        &p_blocks_clean_large,
                        &self.times_i64,
                        pred_t_start,
                        tau,
                        &act_l,
                        Some(step),
                        Some(nlt),
                    )?
                };

                let mut inh_exp_map = HashMap::new();
                inh_exp_map.insert(
                    sp_a_key.clone(),
                    SpeciesExperimentData {
                        s_matrix: s_dem_s.clone(),
                        times: self.times_i64.clone(),
                        p: None,
                        p_blocks: Some(p_blocks_raw_small.clone()),
                    },
                );
                if !is_single {
                    inh_exp_map.insert(
                        sp_b_key.clone(),
                        SpeciesExperimentData {
                            s_matrix: s_dem_l.clone(),
                            times: self.times_i64.clone(),
                            p: None,
                            p_blocks: Some(p_blocks_raw_large.clone()),
                        },
                    );
                }
                let inh_exp = ExperimentData {
                    config: cfg.to_json(),
                    stats: JsonValue::object(),
                    inhomogeneous: true,
                    inhomogeneous_metadata: None,
                    states_matrix: None,
                    species: inh_exp_map,
                };
                let val_inh = validate_experiment("session_inhom", &inh_exp, 1e-3);
                let im = evaluate_chain_metrics(ChainEvalInput {
                    s_dem_small: s_dem_s,
                    s_dem_large: s_dem_l,
                    s_mk_small: &s_inh_s,
                    s_mk_large: &s_inh_l,
                    times_dem: &self.times_i64,
                    times_eval: &inh_eval_times,
                    p_ref_small: &p_blocks_raw_small[0],
                    p_ref_large: &p_blocks_raw_large[0],
                    activated_small: &act_s,
                    activated_large: &act_l,
                    target_is_large,
                    validation: &val_inh,
                })?;
                inhom_metrics = Some(im);
                s_mk_inhom_small = Some(s_inh_s);
                s_mk_inhom_large = Some(s_inh_l);
            }
        }

        Ok(TemporalEvalOutput {
            start_index,
            nlt,
            step,
            dt,
            tau,
            n_pairs_used: pairs.len(),
            n_visited_hom,
            nan_rows_hom,
            nan_rows_inhom_mean,
            diag_mean_raw,
            lambda_max_hom,
            hom_metrics,
            inhom_metrics,
            p_hom_raw_small,
            p_hom_raw_large,
            p_hom_clean_small,
            p_hom_clean_large,
            activated_small: act_s,
            activated_large: act_l,
            p_blocks_raw_small,
            p_blocks_raw_large,
            p_blocks_clean_small,
            p_blocks_clean_large,
            block_times,
            block_labels,
            is_single_species: is_single,
            sp_a_key,
            sp_b_key,
            s_mk_hom_small,
            s_mk_hom_large,
            s_mk_inhom_small,
            s_mk_inhom_large,
        })
    }
}

pub fn stabilize_blocks(raw: &[Matrix], p_fallback: &Matrix, act: &[bool]) -> Vec<Matrix> {
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
}

/// Thread-safe bounded LRU cache of [`PreparedMethodSession`] objects.
#[derive(Debug, Default)]
pub struct ResearchSessionCache {
    inner: Mutex<SessionCacheInner>,
}

#[derive(Debug, Default)]
struct SessionCacheInner {
    map: HashMap<SessionKey, Arc<PreparedMethodSession>>,
    order: VecDeque<SessionKey>,
}

impl ResearchSessionCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.inner.lock().map(|g| g.map.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get_or_build(
        &self,
        dataset_name: &str,
        method: &str,
        kwargs: &HashMap<String, JsonValue>,
        permanent_start: usize,
        use_bed_frame: bool,
        frames: &BTreeMap<usize, DemFrame>,
    ) -> Result<Arc<PreparedMethodSession>> {
        let sorted_min = frames.keys().next().copied().unwrap_or(0);
        let sorted_max = frames.keys().next_back().copied().unwrap_or(0);
        let perm = if permanent_start >= sorted_min && permanent_start <= sorted_max {
            permanent_start
        } else {
            sorted_min
        };
        let key = SessionKey::new(dataset_name, method, kwargs, perm, use_bed_frame);

        if let Ok(guard) = self.inner.lock() {
            if let Some(cached) = guard.map.get(&key) {
                return Ok(Arc::clone(cached));
            }
        }

        let built = Arc::new(build_prepared_session(
            dataset_name,
            method,
            kwargs,
            perm,
            use_bed_frame,
            frames,
        )?);

        if let Ok(mut guard) = self.inner.lock() {
            if let Some(existing) = guard.map.get(&key) {
                return Ok(Arc::clone(existing));
            }
            if guard.order.len() >= 48 {
                if let Some(old) = guard.order.pop_front() {
                    guard.map.remove(&old);
                }
            }
            guard.order.push_back(key.clone());
            guard.map.insert(key, Arc::clone(&built));
        }

        Ok(built)
    }
}
