//! Markovian sweep pipelines (homogeneous, no-species, and inhomogeneous).
//!
//! Mirrors `dem_mcm_coupling/run_sweep.py`.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::config::{
    validate_partitioning_method, PartitioningMethod, N_PARTICLES_PER_TIMESTEP, PERMANENT_START,
};
use crate::data::{DataSource, DemFrame, ExperimentWritePayload};
use crate::error::{DemMcmError, Result};
use crate::io::json::JsonValue;
use crate::io::npy::NpyArray;
use crate::linalg::Matrix;
use crate::partitioners::{create_partitioner, Partitioner};

pub const BASE_OUTPUT_DIR: &str = "RaffinageTemporel";

/// Configuration of a single Markov experiment.
#[derive(Debug, Clone)]
pub struct ExperimentConfig {
    pub method: String,
    pub method_kwargs: HashMap<String, JsonValue>,
    pub nlt: usize,
    pub tau: usize,
    pub step: usize,
    pub dt: usize,
    pub start_index: usize,
    pub particle_diameter: Option<f64>,
    pub inhomogeneous: bool,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self::new("cartesian")
    }
}

impl ExperimentConfig {
    pub fn new(method: impl Into<String>) -> Self {
        let step = 157usize;
        Self {
            method: method.into(),
            method_kwargs: HashMap::new(),
            nlt: 2,
            tau: 157,
            step,
            dt: (step / 100).max(1),
            start_index: 157,
            particle_diameter: None,
            inhomogeneous: false,
        }
    }

    pub fn with_dt_auto(mut self) -> Self {
        self.dt = (self.step / 100).max(1);
        self
    }

    /// Return the experiment folder name (without path prefix).
    pub fn output_folder(&self, sample_coords: Option<&[[f64; 3]]>) -> Result<String> {
        let mut partitioner = create_partitioner(&self.method, &self.method_kwargs)?;
        if let Some(coords) = sample_coords {
            partitioner.fit(coords)?;
        }
        let mut folder = format!(
            "{}_NLT{}_step{}_dt{}_tau{}_start{}",
            partitioner.label(),
            self.nlt,
            self.step,
            self.dt,
            self.tau,
            self.start_index
        );
        if let Some(d) = self.particle_diameter {
            let d_str = format!("{d}").replace('.', "");
            folder.push_str(&format!("_d{d_str}"));
        }
        if self.inhomogeneous {
            folder = format!("inhomogeneous_{folder}");
        }
        Ok(folder)
    }

    pub fn to_json(&self) -> JsonValue {
        let mut obj = JsonValue::object();
        obj.insert("method", JsonValue::String(self.method.clone()));
        let mut kw = JsonValue::object();
        for (k, v) in &self.method_kwargs {
            kw.insert(k, v.clone());
        }
        obj.insert("method_kwargs", kw);
        obj.insert("nlt", JsonValue::Number(self.nlt as f64));
        obj.insert("tau", JsonValue::Number(self.tau as f64));
        obj.insert("step", JsonValue::Number(self.step as f64));
        obj.insert("dt", JsonValue::Number(self.dt as f64));
        obj.insert("start_index", JsonValue::Number(self.start_index as f64));
        obj.insert(
            "particle_diameter",
            self.particle_diameter
                .map_or(JsonValue::Null, JsonValue::Number),
        );
        obj.insert("inhomogeneous", JsonValue::Bool(self.inhomogeneous));
        obj
    }
}

fn kw_u(pairs: &[(&str, usize)]) -> HashMap<String, JsonValue> {
    pairs
        .iter()
        .map(|&(k, v)| (k.to_string(), JsonValue::Number(v as f64)))
        .collect()
}

pub fn default_kwargs(method: &str) -> HashMap<String, JsonValue> {
    let mut m = HashMap::new();
    match method {
        "cartesian" => return kw_u(&[("nx", 5), ("ny", 5), ("nz", 5)]),
        "cylindrical" => {
            m.insert("nr".into(), JsonValue::Number(3.0));
            m.insert("ntheta".into(), JsonValue::Number(8.0));
            m.insert("nz".into(), JsonValue::Number(1.0));
            m.insert("radial_mode".into(), JsonValue::String("equal_area".into()));
        }
        "voronoi" => return kw_u(&[("n_cells", 40)]),
        "quantile" => return kw_u(&[("nx", 5), ("ny", 5), ("nz", 5)]),
        "octree" => return kw_u(&[("max_particles", 100), ("max_depth", 1)]),
        "physics" | "physics_full_vel" => {
            m.insert("n_cells".into(), JsonValue::Number(30.0));
            m.insert("velocity_weight".into(), JsonValue::Number(0.5));
        }
        "spectral" => {
            m.insert("n_cells".into(), JsonValue::Number(20.0));
            m.insert("velocity_weight".into(), JsonValue::Number(0.5));
            m.insert("n_neighbors".into(), JsonValue::Number(15.0));
            m.insert("max_samples".into(), JsonValue::Number(5000.0));
        }
        "gmm" => {
            m.insert("n_cells".into(), JsonValue::Number(20.0));
            m.insert("velocity_weight".into(), JsonValue::Number(0.5));
        }
        "spectral_biclustering" => {
            m.insert("n_cells".into(), JsonValue::Number(30.0));
            m.insert("n_col_clusters".into(), JsonValue::Number(3.0));
            m.insert("method".into(), JsonValue::String("log".into()));
            m.insert("velocity_weight".into(), JsonValue::Number(0.6));
        }
        "adaptive" => {
            m.insert("y_split".into(), JsonValue::Number(0.90));
            m.insert("y_split_mode".into(), JsonValue::String("quantile".into()));
            m.insert("top_method".into(), JsonValue::String("single".into()));
            m.insert("top_kwargs".into(), JsonValue::object());
            m.insert("bottom_method".into(), JsonValue::String("voronoi".into()));
            let mut bk = JsonValue::object();
            bk.insert("n_cells", JsonValue::Number(100.0));
            m.insert("bottom_kwargs".into(), bk);
        }
        _ => {}
    }
    m
}

/// Generate the deduplicated spatial + temporal sweep configurations for `method`.
pub fn get_configs(method: &str, particle_diameter: Option<f64>) -> Result<Vec<ExperimentConfig>> {
    let _ = validate_partitioning_method(method)?;
    let mut configs: Vec<ExperimentConfig> = Vec::new();
    let make = |kw: HashMap<String, JsonValue>| {
        let mut c = ExperimentConfig::new(method);
        c.method_kwargs = kw;
        c.particle_diameter = particle_diameter;
        c
    };

    match method {
        "cartesian" => {
            for n in [2, 3, 4, 5] {
                configs.push(make(kw_u(&[("nx", n), ("ny", n), ("nz", n)])));
            }
        }
        "cylindrical" => {
            for nr in [3, 4, 5, 6] {
                let mut kw = kw_u(&[("nr", nr), ("ntheta", 3), ("nz", 3)]);
                kw.insert("radial_mode".into(), JsonValue::String("equal_area".into()));
                configs.push(make(kw));
            }
        }
        "dbscan" => {
            for ms in [3, 4, 5, 6] {
                configs.push(make(kw_u(&[("min_samples", ms)])));
            }
        }
        "voronoi" => {
            for nc in [10, 15, 20, 25, 30] {
                configs.push(make(kw_u(&[("n_cells", nc)])));
            }
        }
        "quantile" => {
            for n in [2, 3, 4, 5, 6] {
                configs.push(make(kw_u(&[("nx", n), ("ny", n), ("nz", 1)])));
            }
        }
        "octree" => {
            for mp in [20, 40, 80, 16, 32, 64, 28, 50, 60, 70, 100] {
                configs.push(make(kw_u(&[("max_particles", mp), ("max_depth", 2)])));
            }
            for md in [2, 1] {
                configs.push(make(kw_u(&[("max_particles", 100), ("max_depth", md)])));
            }
        }
        "physics" => {
            for nc in [10, 15, 20, 25, 30] {
                let mut kw = kw_u(&[("n_cells", nc)]);
                kw.insert("velocity_weight".into(), JsonValue::Number(0.5));
                configs.push(make(kw));
            }
            for vw in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0] {
                let mut kw = kw_u(&[("n_cells", 30)]);
                kw.insert("velocity_weight".into(), JsonValue::Number(vw));
                configs.push(make(kw));
            }
        }
        "physics_full_vel" => {
            for nc in [10, 15, 20, 25, 30] {
                let mut kw = kw_u(&[("n_cells", nc)]);
                kw.insert("velocity_weight".into(), JsonValue::Number(0.5));
                configs.push(make(kw));
            }
            for vw in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0] {
                let mut kw = kw_u(&[("n_cells", 30)]);
                kw.insert("velocity_weight".into(), JsonValue::Number(vw));
                configs.push(make(kw));
            }
        }
        "spectral" => {
            for nc in [10, 15, 20, 25, 30] {
                let mut kw = kw_u(&[("n_cells", nc), ("n_neighbors", 15)]);
                kw.insert("velocity_weight".into(), JsonValue::Number(0.5));
                configs.push(make(kw));
            }
            for vw in [0.1, 0.3, 0.5, 0.7, 1.0] {
                let mut kw = kw_u(&[("n_cells", 20), ("n_neighbors", 15)]);
                kw.insert("velocity_weight".into(), JsonValue::Number(vw));
                configs.push(make(kw));
            }
            for k in [5, 10, 15, 20, 30] {
                let mut kw = kw_u(&[("n_cells", 20), ("n_neighbors", k)]);
                kw.insert("velocity_weight".into(), JsonValue::Number(0.5));
                configs.push(make(kw));
            }
        }
        "gmm" => {
            for nc in [10, 15, 20, 25, 30] {
                let mut kw = kw_u(&[("n_cells", nc)]);
                kw.insert("velocity_weight".into(), JsonValue::Number(0.5));
                configs.push(make(kw));
            }
            for vw in [0.1, 0.3, 0.5, 0.7, 1.0] {
                let mut kw = kw_u(&[("n_cells", 20)]);
                kw.insert("velocity_weight".into(), JsonValue::Number(vw));
                configs.push(make(kw));
            }
        }
        "spectral_biclustering" => {
            for (nc, ncol) in [(20, 3), (30, 3), (40, 4), (50, 4)] {
                let mut kw = kw_u(&[("n_cells", nc), ("n_col_clusters", ncol)]);
                kw.insert("method".into(), JsonValue::String("log".into()));
                kw.insert("velocity_weight".into(), JsonValue::Number(0.6));
                configs.push(make(kw));
            }
        }
        "adaptive" => {
            for y_q in [0.5, 0.6, 0.7, 0.8, 0.9] {
                let mut kw = HashMap::new();
                kw.insert("y_split".into(), JsonValue::Number(y_q));
                kw.insert("y_split_mode".into(), JsonValue::String("quantile".into()));
                kw.insert("top_method".into(), JsonValue::String("single".into()));
                kw.insert(
                    "bottom_method".into(),
                    JsonValue::String("cylindrical".into()),
                );
                let mut bk = JsonValue::object();
                bk.insert("nr", JsonValue::Number(1.0));
                bk.insert("ntheta", JsonValue::Number(36.0));
                bk.insert("nz", JsonValue::Number(1.0));
                bk.insert("radial_mode", JsonValue::String("equal_area".into()));
                kw.insert("bottom_kwargs".into(), bk);
                configs.push(make(kw));
            }
            for nc in [10, 20, 30, 50, 64] {
                let mut kw = HashMap::new();
                kw.insert("y_split".into(), JsonValue::Number(0.90));
                kw.insert("y_split_mode".into(), JsonValue::String("quantile".into()));
                kw.insert("top_method".into(), JsonValue::String("single".into()));
                kw.insert("bottom_method".into(), JsonValue::String("voronoi".into()));
                let mut bk = JsonValue::object();
                bk.insert("n_cells", JsonValue::Number(nc as f64));
                kw.insert("bottom_kwargs".into(), bk);
                configs.push(make(kw));
            }
        }
        "single" => {
            configs.push(make(HashMap::new()));
        }
        _ => {}
    }

    // Temporal sweep with default spatial kwargs
    let def_kw = default_kwargs(method);
    let mut temporal = Vec::new();
    for nlt in [1, 2, 3, 5] {
        let mut c = make(def_kw.clone());
        c.nlt = nlt;
        temporal.push(c);
    }
    for step in [20, 30, 40] {
        let mut c = make(def_kw.clone());
        c.step = step;
        c = c.with_dt_auto();
        temporal.push(c);
    }
    for dt in [1, 2, 3, 4] {
        let mut c = make(def_kw.clone());
        c.step = 20;
        c.dt = dt;
        temporal.push(c);
    }
    for tau in [20, 50, 100, 200] {
        let mut c = make(def_kw.clone());
        c.tau = tau;
        temporal.push(c);
    }
    for (nlt, step, dt, tau) in [(3, 100, 1, 50), (5, 20, 2, 100), (2, 20, 1, 100)] {
        let mut c = make(def_kw.clone());
        c.nlt = nlt;
        c.step = step;
        c.dt = dt;
        c.tau = tau;
        temporal.push(c);
    }

    configs.extend(temporal);

    let mut seen: HashSet<String> = HashSet::new();
    let mut unique = Vec::new();
    for cfg in configs {
        let key = if cfg.method == "adaptive" || cfg.method == "multizone" {
            let mut kw_j = JsonValue::object();
            for (k, v) in &cfg.method_kwargs {
                kw_j.insert(k, v.clone());
            }
            format!(
                "{}_{}_NLT{}_step{}_dt{}_tau{}_start{}",
                cfg.method,
                kw_j.to_string_compact(),
                cfg.nlt,
                cfg.step,
                cfg.dt,
                cfg.tau,
                cfg.start_index
            )
        } else {
            cfg.output_folder(None)?
        };
        if seen.insert(key) {
            unique.push(cfg);
        }
    }
    Ok(unique)
}

// ============================================================================
// TRANSITION MATRIX ESTIMATION
// ============================================================================

/// Compute the row-stochastic transition matrix `P` from `(states_prev, states_curr)`.
///
/// Matches `compute_P_matrix_torch` bit-for-bit:
/// - `P[i, j] = #(i -> j) / #(i)`
/// - out-of-zone (`< 0` or `>= n_states`) states are skipped
/// - unvisited source rows (`#(i) == 0`) contain `NaN` (`0.0 / 0.0`) to signal
///   unvisited rows until cleaned by `clean_transition_matrix`.
pub fn compute_p_matrix(states_prev: &[i64], states_curr: &[i64], n_states: usize) -> Matrix {
    let n = states_prev.len().min(states_curr.len());
    let mut counts = vec![0u64; n_states * n_states];
    let mut denom = vec![0u64; n_states];

    for k in 0..n {
        let sp = states_prev[k];
        let sc = states_curr[k];
        if sp >= 0 && (sp as usize) < n_states {
            let i = sp as usize;
            denom[i] += 1;
            if sc >= 0 && (sc as usize) < n_states {
                let j = sc as usize;
                counts[i * n_states + j] += 1;
            }
        }
    }

    let mut p = Matrix::zeros(n_states, n_states);
    for i in 0..n_states {
        let d = denom[i] as f64;
        for j in 0..n_states {
            p.set(i, j, (counts[i * n_states + j] as f64) / d);
        }
    }
    p
}

// ============================================================================
// SPECIES DETECTION & STATE MATRICES
// ============================================================================

/// Auto-detect particle species by diameter on a single DEM frame.
pub fn detect_species(frame: &DemFrame) -> Vec<(String, Vec<bool>)> {
    let n = frame.n_particles();
    if frame.diameters.len() != n {
        return vec![("all".to_string(), vec![true; n])];
    }

    let mut unique: Vec<f64> = Vec::new();
    for &d in &frame.diameters {
        if !unique.iter().any(|&u| (u - d).abs() < 1e-12) {
            unique.push(d);
        }
    }
    unique.sort_by(|a, b| a.total_cmp(b));

    if unique.len() <= 1 {
        return vec![("all".to_string(), vec![true; n])];
    }

    let labels: Vec<String> = if unique.len() == 2 {
        vec!["small".to_string(), "large".to_string()]
    } else {
        unique
            .iter()
            .map(|d| format!("d{}", format!("{d}").replace('.', "")))
            .collect()
    };

    labels
        .into_iter()
        .zip(unique)
        .map(|(lbl, diam)| {
            let mask = frame
                .diameters
                .iter()
                .map(|&d| (d - diam).abs() < 1e-12)
                .collect();
            (lbl, mask)
        })
        .collect()
}

fn compute_state_matrices(
    timestep_dict: &BTreeMap<usize, DemFrame>,
    partitioner: &mut dyn Partitioner,
) -> Result<(Vec<Vec<i64>>, Vec<usize>)> {
    let sorted_indices: Vec<usize> = timestep_dict.keys().copied().collect();
    if sorted_indices.is_empty() {
        return Err(DemMcmError::InvalidArgument(
            "Empty timestep dictionary".to_string(),
        ));
    }
    let n_timesteps = sorted_indices.len();
    let n_particles = timestep_dict[&sorted_indices[0]].n_particles();

    let uses_vel = matches!(
        partitioner.method(),
        PartitioningMethod::Physics
            | PartitioningMethod::PhysicsFullVel
            | PartitioningMethod::SpectralBiclustering
            | PartitioningMethod::Gmm
            | PartitioningMethod::Dbscan
            | PartitioningMethod::Spectral
    );

    let mut all_coords = Vec::with_capacity(n_timesteps * n_particles);
    let mut all_vels = if uses_vel {
        Vec::with_capacity(n_timesteps * n_particles)
    } else {
        Vec::new()
    };

    for &ts in &sorted_indices {
        let frame = &timestep_dict[&ts];
        all_coords.extend_from_slice(&frame.coordinates);
        if uses_vel {
            if frame.velocities.len() == frame.coordinates.len() {
                all_vels.extend_from_slice(&frame.velocities);
            } else {
                all_vels.resize(all_coords.len(), [0.0; 3]);
            }
        }
    }

    let states_flat = if uses_vel {
        partitioner.set_use_velocity(true);
        partitioner.set_dem_velocities(Some(&all_vels));
        partitioner.compute_states(&all_coords, Some(&all_vels))?
    } else {
        partitioner.compute_states(&all_coords, None)?
    };

    let mut states_matrix = Vec::with_capacity(n_timesteps);
    let mut cursor = 0usize;
    for &ts in &sorted_indices {
        let np = timestep_dict[&ts].n_particles();
        states_matrix.push(states_flat[cursor..cursor + np].to_vec());
        cursor += np;
    }

    Ok((states_matrix, sorted_indices))
}

fn build_s_matrices(
    states_matrix: &[Vec<i64>],
    species_masks: &[(String, Vec<bool>)],
    n_states: usize,
) -> HashMap<String, Matrix> {
    let n_timesteps = states_matrix.len();
    let mut out = HashMap::new();
    for (sp, mask) in species_masks {
        let mut s_mat = Matrix::zeros(n_timesteps, n_states);
        for (t, row) in states_matrix.iter().enumerate() {
            for (p, &st) in row.iter().enumerate() {
                if mask[p] && st >= 0 && (st as usize) < n_states {
                    let cur = s_mat.get(t, st as usize);
                    s_mat.set(t, st as usize, cur + 1.0);
                }
            }
        }
        out.insert(sp.clone(), s_mat);
    }
    out
}

fn build_pairs(
    config: &ExperimentConfig,
    max_ts: usize,
    idx_to_row: &HashMap<usize, usize>,
) -> Vec<(usize, usize)> {
    let dt = config.dt.max(1);
    let mut pairs = Vec::new();
    for nlt_idx in 0..config.nlt {
        let current_start_base = config.start_index + nlt_idx * (config.step + config.tau);
        let Some(max_start_possible) = max_ts.checked_sub(config.tau) else {
            break;
        };
        if current_start_base > max_start_possible {
            break;
        }
        let n_app = if nlt_idx + 1 == config.nlt {
            let rem = max_start_possible - current_start_base;
            ((config.step + config.tau) / dt).min(rem / dt) + 1
        } else {
            (config.step + config.tau) / dt
        };
        for i in 0..n_app {
            let s_idx = current_start_base + i * dt;
            let e_idx = s_idx + config.tau;
            if !idx_to_row.contains_key(&s_idx) || !idx_to_row.contains_key(&e_idx) {
                break;
            }
            pairs.push((s_idx, e_idx));
        }
    }
    pairs
}

fn build_inhomogeneous_blocks(
    config: &ExperimentConfig,
    max_ts: usize,
    idx_to_row: &HashMap<usize, usize>,
) -> Vec<Vec<(usize, usize)>> {
    let dt = config.dt.max(1);
    let mut blocks = Vec::new();
    for nlt_idx in 0..config.nlt {
        let current_start_base = config.start_index + nlt_idx * (config.step + config.tau);
        let Some(max_start_possible) = max_ts.checked_sub(config.tau) else {
            break;
        };
        if current_start_base > max_start_possible {
            break;
        }
        let n_app = if nlt_idx + 1 == config.nlt {
            let rem = max_start_possible - current_start_base;
            (config.tau / dt).min(rem / dt) + 1
        } else {
            config.tau / dt
        };
        let mut block_pairs = Vec::new();
        for i in 0..n_app {
            let s_idx = current_start_base + i * dt;
            let e_idx = s_idx + config.tau;
            if !idx_to_row.contains_key(&s_idx) || !idx_to_row.contains_key(&e_idx) {
                break;
            }
            block_pairs.push((s_idx, e_idx));
        }
        if !block_pairs.is_empty() {
            blocks.push(block_pairs);
        }
    }
    blocks
}

// ============================================================================
// RESULT CONTAINERS & RUNNERS
// ============================================================================

#[derive(Debug, Clone)]
pub struct SpeciesRunOutput {
    pub p: Option<Matrix>,
    pub p_blocks: Option<Vec<Matrix>>,
    pub s_matrix: Matrix,
    pub times: Vec<i64>,
}

#[derive(Debug, Clone)]
pub struct ExperimentRunResults {
    pub states_matrix: Vec<Vec<i64>>,
    pub species: BTreeMap<String, SpeciesRunOutput>,
    pub stats: JsonValue,
}

fn finite_diag_mean_std(p: &Matrix) -> (f64, f64) {
    let vals: Vec<f64> = (0..p.rows)
        .filter_map(|i| {
            let v = p.get(i, i);
            if v.is_finite() {
                Some(v)
            } else {
                None
            }
        })
        .collect();
    if vals.is_empty() {
        return (0.0, 0.0);
    }
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    let var = vals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
    (mean, var.sqrt())
}

fn run_experiment_with_masks(
    config: &ExperimentConfig,
    partitioner: &mut dyn Partitioner,
    timestep_dict: &BTreeMap<usize, DemFrame>,
    species_masks: &[(String, Vec<bool>)],
    species_masks_applied: bool,
) -> Result<ExperimentRunResults> {
    let n_states = partitioner.n_cells();
    let (states_matrix, sorted_indices) = compute_state_matrices(timestep_dict, partitioner)?;
    let n_timesteps = sorted_indices.len();
    let idx_to_row: HashMap<usize, usize> = sorted_indices
        .iter()
        .enumerate()
        .map(|(r, &ts)| (ts, r))
        .collect();
    let max_ts = *sorted_indices.last().unwrap();

    let s_matrices = build_s_matrices(&states_matrix, species_masks, n_states);
    let pairs = build_pairs(config, max_ts, &idx_to_row);
    if pairs.is_empty() {
        return Err(DemMcmError::InvalidArgument(
            "No valid transition pair with these parameters".to_string(),
        ));
    }

    let times_i64: Vec<i64> = sorted_indices.iter().map(|&t| t as i64).collect();
    let mut species_out = BTreeMap::new();

    for (sp, mask) in species_masks {
        let mut prev = Vec::new();
        let mut curr = Vec::new();
        for &(s_idx, e_idx) in &pairs {
            let rp = idx_to_row[&s_idx];
            let rc = idx_to_row[&e_idx];
            for (p_idx, &keep) in mask.iter().enumerate() {
                if keep {
                    prev.push(states_matrix[rp][p_idx]);
                    curr.push(states_matrix[rc][p_idx]);
                }
            }
        }
        let p = compute_p_matrix(&prev, &curr, n_states);
        let s_matrix = s_matrices[sp].clone();
        species_out.insert(
            sp.clone(),
            SpeciesRunOutput {
                p: Some(p),
                p_blocks: None,
                s_matrix,
                times: times_i64.clone(),
            },
        );
    }

    let first_sp = &species_masks[0].0;
    let p_ref = species_out[first_sp].p.as_ref().unwrap();
    let n_visited = (0..n_states)
        .filter(|&r| {
            let rs: f64 = p_ref.row(r).iter().sum();
            rs > 0.0
        })
        .count();
    let (diag_mean, diag_std) = finite_diag_mean_std(p_ref);
    let n_paires_par_bloc = (config.step + config.tau) / config.dt.max(1);
    let fraction_visited = if n_states > 0 {
        ((n_visited as f64 / n_states as f64) * 10000.0).round() / 10000.0
    } else {
        0.0
    };

    let sp_list: Vec<JsonValue> = species_masks
        .iter()
        .map(|(s, _)| JsonValue::String(s.clone()))
        .collect();
    let mut stats = JsonValue::object();
    stats.insert("n_pairs_used", JsonValue::Number(pairs.len() as f64));
    stats.insert("n_nlt_requested", JsonValue::Number(config.nlt as f64));
    stats.insert(
        "n_blocs_complets",
        JsonValue::Number((pairs.len() / n_paires_par_bloc.max(1)) as f64),
    );
    stats.insert(
        "n_paires_dernier_bloc",
        JsonValue::Number((pairs.len() % n_paires_par_bloc.max(1)) as f64),
    );
    stats.insert("n_states", JsonValue::Number(n_states as f64));
    stats.insert("n_states_visited", JsonValue::Number(n_visited as f64));
    stats.insert(
        "n_states_empty",
        JsonValue::Number((n_states.saturating_sub(n_visited)) as f64),
    );
    stats.insert("fraction_visited", JsonValue::Number(fraction_visited));
    stats.insert("diagonal_mean", JsonValue::Number(diag_mean));
    stats.insert("diagonal_std", JsonValue::Number(diag_std));
    stats.insert("method", JsonValue::String(config.method.clone()));
    stats.insert("species", JsonValue::Array(sp_list.clone()));
    stats.insert("species_list", JsonValue::Array(sp_list));
    stats.insert(
        "species_masks_applied",
        JsonValue::Bool(species_masks_applied),
    );
    stats.insert("n_timesteps", JsonValue::Number(n_timesteps as f64));
    stats.insert("tau", JsonValue::Number(config.tau as f64));
    stats.insert("step", JsonValue::Number(config.step as f64));
    stats.insert("dt", JsonValue::Number(config.dt as f64));
    stats.insert(
        "raffinage_ratio",
        JsonValue::Number(n_paires_par_bloc as f64),
    );
    stats.insert(
        "plage_temporelle",
        JsonValue::Number((pairs.last().unwrap().1 - pairs[0].0) as f64),
    );
    stats.insert("start_index", JsonValue::Number(config.start_index as f64));
    stats.insert(
        "first_pair",
        JsonValue::Array(vec![
            JsonValue::Number(pairs[0].0 as f64),
            JsonValue::Number(pairs[0].1 as f64),
        ]),
    );
    stats.insert(
        "last_pair",
        JsonValue::Array(vec![
            JsonValue::Number(pairs.last().unwrap().0 as f64),
            JsonValue::Number(pairs.last().unwrap().1 as f64),
        ]),
    );
    stats.insert(
        "particle_diameter",
        config
            .particle_diameter
            .map_or(JsonValue::Null, JsonValue::Number),
    );

    Ok(ExperimentRunResults {
        states_matrix,
        species: species_out,
        stats,
    })
}

/// Build per-species homogeneous transition matrices from DEM trajectories.
pub fn run_experiment(
    config: &ExperimentConfig,
    partitioner: &mut dyn Partitioner,
    timestep_dict: &BTreeMap<usize, DemFrame>,
) -> Result<ExperimentRunResults> {
    let df_init = timestep_dict.get(&config.start_index).ok_or_else(|| {
        DemMcmError::MissingData(format!(
            "Timestep start_base={} absent from the data",
            config.start_index
        ))
    })?;
    let species_masks = detect_species(df_init);
    run_experiment_with_masks(config, partitioner, timestep_dict, &species_masks, true)
}

/// Build ONE homogeneous transition matrix for all particles combined (no species mask).
pub fn run_no_species_experiment(
    config: &ExperimentConfig,
    partitioner: &mut dyn Partitioner,
    timestep_dict: &BTreeMap<usize, DemFrame>,
) -> Result<ExperimentRunResults> {
    let df_init = timestep_dict.get(&config.start_index).ok_or_else(|| {
        DemMcmError::MissingData(format!(
            "Timestep start_base={} absent from the data",
            config.start_index
        ))
    })?;
    let species_masks = vec![("all".to_string(), vec![true; df_init.n_particles()])];
    run_experiment_with_masks(config, partitioner, timestep_dict, &species_masks, false)
}

/// Build one transition matrix `P_k` per NLT block (inhomogeneous Markov chain).
pub fn run_inhomogeneous_experiment(
    config: &ExperimentConfig,
    partitioner: &mut dyn Partitioner,
    timestep_dict: &BTreeMap<usize, DemFrame>,
) -> Result<ExperimentRunResults> {
    let df_init = timestep_dict.get(&config.start_index).ok_or_else(|| {
        DemMcmError::MissingData(format!(
            "Timestep start_base={} absent from the data",
            config.start_index
        ))
    })?;
    let species_masks = detect_species(df_init);
    let n_states = partitioner.n_cells();
    let (states_matrix, sorted_indices) = compute_state_matrices(timestep_dict, partitioner)?;
    let n_timesteps = sorted_indices.len();
    let idx_to_row: HashMap<usize, usize> = sorted_indices
        .iter()
        .enumerate()
        .map(|(r, &ts)| (ts, r))
        .collect();
    let max_ts = *sorted_indices.last().unwrap();

    let s_matrices = build_s_matrices(&states_matrix, &species_masks, n_states);
    let blocks = build_inhomogeneous_blocks(config, max_ts, &idx_to_row);
    if blocks.is_empty() {
        return Err(DemMcmError::InvalidArgument(
            "No valid pair — cannot build the blocks".to_string(),
        ));
    }

    let times_i64: Vec<i64> = sorted_indices.iter().map(|&t| t as i64).collect();
    let mut p_blocks_by_sp: HashMap<String, Vec<Matrix>> = species_masks
        .iter()
        .map(|(sp, _)| (sp.clone(), Vec::with_capacity(blocks.len())))
        .collect();

    for block_pairs in &blocks {
        for (sp, mask) in &species_masks {
            let mut prev = Vec::new();
            let mut curr = Vec::new();
            for &(s_idx, e_idx) in block_pairs {
                let rp = idx_to_row[&s_idx];
                let rc = idx_to_row[&e_idx];
                for (p_idx, &keep) in mask.iter().enumerate() {
                    if keep {
                        prev.push(states_matrix[rp][p_idx]);
                        curr.push(states_matrix[rc][p_idx]);
                    }
                }
            }
            let pk = compute_p_matrix(&prev, &curr, n_states);
            p_blocks_by_sp.get_mut(sp).unwrap().push(pk);
        }
    }

    let mut species_out = BTreeMap::new();
    for (sp, _) in &species_masks {
        species_out.insert(
            sp.clone(),
            SpeciesRunOutput {
                p: None,
                p_blocks: Some(p_blocks_by_sp.remove(sp).unwrap()),
                s_matrix: s_matrices[sp].clone(),
                times: times_i64.clone(),
            },
        );
    }

    let first_sp = &species_masks[0].0;
    let first_blocks = species_out[first_sp].p_blocks.as_ref().unwrap();
    let n_visited_total = (0..n_states)
        .filter(|&r| {
            first_blocks
                .iter()
                .any(|b| b.row(r).iter().sum::<f64>() > 0.0)
        })
        .count();
    let fraction_visited = if n_states > 0 {
        ((n_visited_total as f64 / n_states as f64) * 10000.0).round() / 10000.0
    } else {
        0.0
    };

    let sp_list: Vec<JsonValue> = species_masks
        .iter()
        .map(|(s, _)| JsonValue::String(s.clone()))
        .collect();
    let mut stats = JsonValue::object();
    stats.insert("n_blocks", JsonValue::Number(blocks.len() as f64));
    stats.insert(
        "n_pairs_per_block",
        JsonValue::Array(
            blocks
                .iter()
                .map(|b| JsonValue::Number(b.len() as f64))
                .collect(),
        ),
    );
    stats.insert("n_nlt_requested", JsonValue::Number(config.nlt as f64));
    stats.insert("n_states", JsonValue::Number(n_states as f64));
    stats.insert(
        "n_states_visited",
        JsonValue::Number(n_visited_total as f64),
    );
    stats.insert("fraction_visited", JsonValue::Number(fraction_visited));
    stats.insert("method", JsonValue::String(config.method.clone()));
    stats.insert("species", JsonValue::Array(sp_list.clone()));
    stats.insert("species_list", JsonValue::Array(sp_list));
    stats.insert("n_timesteps", JsonValue::Number(n_timesteps as f64));
    stats.insert("tau", JsonValue::Number(config.tau as f64));
    stats.insert("step", JsonValue::Number(config.step as f64));
    stats.insert("dt", JsonValue::Number(config.dt as f64));
    stats.insert("start_index", JsonValue::Number(config.start_index as f64));
    stats.insert(
        "particle_diameter",
        config
            .particle_diameter
            .map_or(JsonValue::Null, JsonValue::Number),
    );
    stats.insert("inhomogeneous", JsonValue::Bool(true));

    Ok(ExperimentRunResults {
        states_matrix,
        species: species_out,
        stats,
    })
}

/// Persist homogeneous experiment results to a [`DataSource`].
pub fn save_results(
    config: &ExperimentConfig,
    partitioner: &dyn Partitioner,
    results: &ExperimentRunResults,
    folder_name: Option<&str>,
    data_source: &mut dyn DataSource,
) -> Result<String> {
    let fname = match folder_name {
        Some(f) => f.to_string(),
        None => config.output_folder(None)?,
    };
    let mut species_arrays = HashMap::new();
    species_arrays.insert(
        "states_matrix".to_string(),
        NpyArray::from_i64_2d(&results.states_matrix)?,
    );
    for (sp, data) in &results.species {
        if let Some(p) = &data.p {
            species_arrays.insert(format!("transitionmatrix_{sp}"), NpyArray::from_matrix(p));
        }
        species_arrays.insert(
            format!("S_matrix_{sp}"),
            NpyArray::from_matrix(&data.s_matrix),
        );
        species_arrays.insert(
            format!("times_{sp}"),
            NpyArray::from_i64_vec(data.times.clone()),
        );
    }

    let mut art = partitioner.save_artifacts()?;
    let mut meta = JsonValue::object();
    meta.insert(
        "type",
        JsonValue::String(partitioner.type_name().to_string()),
    );
    meta.insert("label", JsonValue::String(partitioner.label()));
    meta.insert("n_cells", JsonValue::Number(partitioner.n_cells() as f64));
    art.json_files.insert("partitioner_meta".to_string(), meta);

    let cfg_json = config.to_json();
    data_source.write_experiment(ExperimentWritePayload {
        folder_name: &fname,
        stats: &results.stats,
        config: &cfg_json,
        species_arrays,
        partitioner_arrays: art.arrays,
        partitioner_json: art.json_files,
        image_data: HashMap::new(),
        prefix: None,
        inhomogeneous_metadata: None,
    })?;

    Ok(fname)
}

/// Persist inhomogeneous experiment results (`P_blocks_{sp}.npy` + metadata).
pub fn save_inhomogeneous_results(
    config: &ExperimentConfig,
    partitioner: &dyn Partitioner,
    results: &ExperimentRunResults,
    folder_name: Option<&str>,
    data_source: &mut dyn DataSource,
) -> Result<String> {
    let fname = match folder_name {
        Some(f) => f.to_string(),
        None => config.output_folder(None)?,
    };
    let mut species_arrays = HashMap::new();
    species_arrays.insert(
        "states_matrix".to_string(),
        NpyArray::from_i64_2d(&results.states_matrix)?,
    );
    for (sp, data) in &results.species {
        if let Some(blocks) = &data.p_blocks {
            species_arrays.insert(format!("P_blocks_{sp}"), NpyArray::from_matrix_3d(blocks)?);
        }
        species_arrays.insert(
            format!("S_matrix_{sp}"),
            NpyArray::from_matrix(&data.s_matrix),
        );
        species_arrays.insert(
            format!("times_{sp}"),
            NpyArray::from_i64_vec(data.times.clone()),
        );
    }

    let n_blocks = results
        .stats
        .get("n_blocks")
        .and_then(|v| v.as_usize())
        .unwrap_or(0);
    let mut inh_meta = JsonValue::object();
    inh_meta.insert("n_blocks", JsonValue::Number(n_blocks as f64));
    if let Some(npb) = results.stats.get("n_pairs_per_block") {
        inh_meta.insert("n_pairs_per_block", npb.clone());
    }
    if let Some(spl) = results.stats.get("species_list") {
        inh_meta.insert("species_list", spl.clone());
    }
    inh_meta.insert(
        "block_start_indices",
        JsonValue::Array(
            (0..n_blocks)
                .map(|i| {
                    JsonValue::Number((config.start_index + i * (config.step + config.tau)) as f64)
                })
                .collect(),
        ),
    );

    let mut art = partitioner.save_artifacts()?;
    let mut meta = JsonValue::object();
    meta.insert(
        "type",
        JsonValue::String(partitioner.type_name().to_string()),
    );
    meta.insert("label", JsonValue::String(partitioner.label()));
    meta.insert("n_cells", JsonValue::Number(partitioner.n_cells() as f64));
    art.json_files.insert("partitioner_meta".to_string(), meta);

    let cfg_json = config.to_json();
    data_source.write_experiment(ExperimentWritePayload {
        folder_name: &fname,
        stats: &results.stats,
        config: &cfg_json,
        species_arrays,
        partitioner_arrays: art.arrays,
        partitioner_json: art.json_files,
        image_data: HashMap::new(),
        prefix: None,
        inhomogeneous_metadata: Some(&inh_meta),
    })?;

    Ok(fname)
}

#[derive(Debug, Clone)]
pub struct SweepOutcome {
    pub folder_name: String,
    pub success: bool,
    pub stats: Option<JsonValue>,
    pub error: Option<String>,
}

fn fit_partitioner_for_sweep(
    partitioner: &mut dyn Partitioner,
    method: &str,
    sample_coords: &[[f64; 3]],
    sample_vels: &[[f64; 3]],
) -> Result<()> {
    let permanent_start = PERMANENT_START * N_PARTICLES_PER_TIMESTEP;
    let start_row = if sample_coords.len() > permanent_start {
        permanent_start
    } else {
        0
    };
    let fitted_with_vel = matches!(
        method,
        "physics" | "physics_full_vel" | "spectral_biclustering" | "dbscan"
    );
    if fitted_with_vel {
        partitioner.set_use_velocity(true);
        if sample_vels.len() == sample_coords.len() {
            partitioner.set_dem_velocities(Some(&sample_vels[start_row..]));
        }
        partitioner.fit(&sample_coords[start_row..])
    } else if method == "cartesian" {
        partitioner.fit(sample_coords)
    } else {
        partitioner.fit(&sample_coords[start_row..])
    }
}

/// Sweep mode selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SweepMode {
    Homogeneous,
    NoSpecies,
    Inhomogeneous,
}

/// Run a sweep over one or all (`method = "all"`) partitioning methods against `data_source`.
pub fn run_sweep(
    method: &str,
    mode: SweepMode,
    explicit_configs: Option<Vec<ExperimentConfig>>,
    particle_diameter: Option<f64>,
    data_source: &mut dyn DataSource,
) -> Result<Vec<SweepOutcome>> {
    let timestep_dict = data_source.read_timesteps(None)?;
    let mut sample_coords = Vec::new();
    let mut sample_vels = Vec::new();
    for frame in timestep_dict.values() {
        sample_coords.extend_from_slice(&frame.coordinates);
        if frame.velocities.len() == frame.coordinates.len() {
            sample_vels.extend_from_slice(&frame.velocities);
        } else {
            sample_vels.resize(sample_coords.len(), [0.0; 3]);
        }
    }

    let configs = match explicit_configs {
        Some(c) => c,
        None => {
            let diam = if mode == SweepMode::NoSpecies {
                None
            } else {
                particle_diameter
            };
            let methods: Vec<&str> = if method == "all" {
                PartitioningMethod::all()
                    .iter()
                    .map(|m| m.as_str())
                    .collect()
            } else {
                vec![method]
            };
            let mut list = Vec::new();
            for m in methods {
                for mut c in get_configs(m, diam)? {
                    if mode == SweepMode::Inhomogeneous {
                        c.inhomogeneous = true;
                    }
                    list.push(c);
                }
            }
            list
        }
    };

    let mut outcomes = Vec::with_capacity(configs.len());
    for cfg in configs {
        let mut partitioner = match create_partitioner(&cfg.method, &cfg.method_kwargs) {
            Ok(p) => p,
            Err(e) => {
                outcomes.push(SweepOutcome {
                    folder_name: cfg.method.clone(),
                    success: false,
                    stats: None,
                    error: Some(e.to_string()),
                });
                continue;
            }
        };

        if let Err(e) = fit_partitioner_for_sweep(
            partitioner.as_mut(),
            &cfg.method,
            &sample_coords,
            &sample_vels,
        ) {
            outcomes.push(SweepOutcome {
                folder_name: cfg.method.clone(),
                success: false,
                stats: None,
                error: Some(e.to_string()),
            });
            continue;
        }

        let base_folder = cfg.output_folder(Some(&sample_coords))?;
        let folder_name = if mode == SweepMode::NoSpecies {
            format!("nospecies_{base_folder}")
        } else {
            base_folder
        };

        let res = match mode {
            SweepMode::Homogeneous => run_experiment(&cfg, partitioner.as_mut(), &timestep_dict)
                .and_then(|r| {
                    save_results(
                        &cfg,
                        partitioner.as_ref(),
                        &r,
                        Some(&folder_name),
                        data_source,
                    )?;
                    Ok(r)
                }),
            SweepMode::NoSpecies => {
                run_no_species_experiment(&cfg, partitioner.as_mut(), &timestep_dict).and_then(
                    |r| {
                        save_results(
                            &cfg,
                            partitioner.as_ref(),
                            &r,
                            Some(&folder_name),
                            data_source,
                        )?;
                        Ok(r)
                    },
                )
            }
            SweepMode::Inhomogeneous => {
                run_inhomogeneous_experiment(&cfg, partitioner.as_mut(), &timestep_dict).and_then(
                    |r| {
                        save_inhomogeneous_results(
                            &cfg,
                            partitioner.as_ref(),
                            &r,
                            Some(&folder_name),
                            data_source,
                        )?;
                        Ok(r)
                    },
                )
            }
        };

        match res {
            Ok(r) => outcomes.push(SweepOutcome {
                folder_name,
                success: true,
                stats: Some(r.stats),
                error: None,
            }),
            Err(e) => outcomes.push(SweepOutcome {
                folder_name,
                success: false,
                stats: None,
                error: Some(e.to_string()),
            }),
        }
    }

    Ok(outcomes)
}
