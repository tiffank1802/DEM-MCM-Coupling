//! Markovian experiment loading, DEM vs MCM comparison, and parametric studies.
//!
//! Mirrors `dem_mcm_coupling/analyze_results.py` and `postprocessing/etudes_librairie.py`.

use std::collections::{BTreeMap, HashMap};

use crate::config::{ExperimentData, TIMESTEP_TO_SECONDS};
use crate::data::{DataSource, DemFrame};
use crate::error::{DemMcmError, Result};
use crate::linalg::Matrix;
use crate::metrics::{
    clean_transition_matrix, entropy_concentration, intensity_of_segregation, mixing_times,
    propagate_markov, propagate_markov_inhomogeneous, rsd_concentration,
};
use crate::partitioners::Partitioner;

/// Species labelling criterion matching `MarkovAnalyzer.label_species`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeciesCriterion {
    Small,
    FirstHalf,
    SpatialBottom,
}

/// Evaluation metrics between DEM reference RSD and Markov-predicted RSD.
#[derive(Debug, Clone, PartialEq)]
pub struct ComparisonMetrics {
    pub rsd_dem: Vec<f64>,
    pub rsd_markov: Vec<f64>,
    pub entropy_dem: Vec<f64>,
    pub entropy_markov: Vec<f64>,
    pub intensity_dem: Vec<f64>,
    pub intensity_markov: Vec<f64>,
    pub times_markov: Vec<i64>,
    pub rmse_rsd: f64,
    pub mae_rsd: f64,
    pub t50_dem: Option<f64>,
    pub t90_dem: Option<f64>,
    pub t50_markov: Option<f64>,
    pub t90_markov: Option<f64>,
}

/// Loader and analyzer of homogeneous & inhomogeneous Markov experiments.
#[derive(Default)]
pub struct MarkovAnalyzer {
    pub results: BTreeMap<String, ExperimentData>,
    pub by_method: HashMap<String, Vec<String>>,
    pub dem_frames: BTreeMap<usize, DemFrame>,
    pub species_labels: Option<Vec<bool>>,
}

/// DEM trajectory metrics tuple `(times, s_small, s_large, rsd, entropy, danckwerts)`.
pub type DemMetricsTrajectory = (Vec<i64>, Matrix, Matrix, Vec<f64>, Vec<f64>, Vec<f64>);

impl MarkovAnalyzer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Detect the partitioning method from an experiment folder name and/or config JSON.
    pub fn detect_method(folder_name: &str, config: Option<&crate::io::json::JsonValue>) -> String {
        if let Some(cfg) = config {
            if let Some(m) = cfg.get("method").and_then(|v| v.as_str()) {
                return m.to_string();
            }
            if cfg.get("nx").is_some() {
                return "cartesian".to_string();
            }
        }
        let mut stripped = folder_name;
        for variant_prefix in ["inhomogeneous_", "nospecies_"] {
            if let Some(rest) = stripped.strip_prefix(variant_prefix) {
                stripped = rest;
                break;
            }
        }
        let prefixes: &[(&str, &[&str])] = &[
            ("cartesian", &["cartesian_", "NLT_"]),
            ("cylindrical", &["cylindrical_"]),
            ("voronoi", &["voronoi_"]),
            ("quantile", &["quantile_"]),
            ("octree", &["octree_"]),
            ("physics_full_vel", &["fullvel_kmeans_"]),
            ("physics", &["physics_"]),
            ("spectral_biclustering", &["spectral_biclustering_"]),
            ("spectral", &["spectral_"]),
            ("gmm", &["gmm_full_"]),
            ("dbscan", &["dbscan_"]),
            ("adaptive", &["adaptive_"]),
            ("multizone", &["multizone_"]),
            ("single", &["single_"]),
        ];
        for &(method, pfxs) in prefixes {
            for &pfx in pfxs {
                if stripped.starts_with(pfx) {
                    return method.to_string();
                }
            }
        }
        "unknown".to_string()
    }

    /// Load every experiment matching `method` from `source`.
    pub fn load_method(&mut self, source: &dyn DataSource, method: Option<&str>) -> Result<usize> {
        let folders = source.list_experiments(None)?;
        let mut count = 0usize;
        for folder in folders {
            if folder.starts_with('_') {
                continue;
            }
            let exp = match source.read_experiment(&folder, None) {
                Ok(e) => e,
                Err(_) => continue,
            };
            let detected = Self::detect_method(&folder, Some(&exp.config));
            if method.is_some_and(|m| m != detected) {
                continue;
            }
            self.by_method
                .entry(detected)
                .or_default()
                .push(folder.clone());
            self.results.insert(folder, exp);
            count += 1;
        }
        Ok(count)
    }

    /// Filter loaded experiments by `fraction_visited >= threshold`.
    pub fn list_available_models(
        &self,
        method: Option<&str>,
        particle_diameter: Option<f64>,
        fraction_visited_threshold: f64,
    ) -> Vec<(String, f64)> {
        let mut out = Vec::new();
        for (folder, exp) in &self.results {
            let det = Self::detect_method(folder, Some(&exp.config));
            if method.is_some_and(|m| m != det) {
                continue;
            }
            if let Some(d_req) = particle_diameter {
                let d_got = exp.stats.get("particle_diameter").and_then(|v| v.as_f64());
                if !d_got.is_some_and(|d| (d - d_req).abs() < 1e-9) {
                    continue;
                }
            }
            let fv = exp
                .stats
                .get("fraction_visited")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0);
            if fv >= fraction_visited_threshold {
                out.push((folder.clone(), fv));
            }
        }
        out
    }

    /// Load DEM snapshots from `source`.
    pub fn load_dem_snapshots(
        &mut self,
        source: &dyn DataSource,
        timestep_indices: Option<&[usize]>,
    ) -> Result<usize> {
        self.dem_frames = source.read_timesteps(timestep_indices)?;
        Ok(self.dem_frames.len())
    }

    /// Label particles as species A (`true`) or species B (`false`).
    pub fn label_species(&mut self, criterion: SpeciesCriterion) -> Result<&[bool]> {
        let frame0 = self.dem_frames.values().next().ok_or_else(|| {
            DemMcmError::MissingData(
                "No DEM snapshot loaded. Call load_dem_snapshots() first.".to_string(),
            )
        })?;
        let n = frame0.n_particles();
        let labels = match criterion {
            SpeciesCriterion::Small => {
                if frame0.diameters.len() == n {
                    frame0.diameters.iter().map(|&d| d < 0.006).collect()
                } else {
                    (0..n).map(|i| i < n / 2).collect()
                }
            }
            SpeciesCriterion::FirstHalf => (0..n).map(|i| i < n / 2).collect(),
            SpeciesCriterion::SpatialBottom => {
                let mut zs: Vec<f64> = frame0.coordinates.iter().map(|c| c[2]).collect();
                zs.sort_by(|a, b| a.total_cmp(b));
                let med = zs[n / 2];
                frame0.coordinates.iter().map(|c| c[2] < med).collect()
            }
        };
        self.species_labels = Some(labels);
        Ok(self.species_labels.as_deref().unwrap())
    }

    /// Compute DEM reference concentration RSD, entropy, and Danckwerts segregation intensity
    /// across loaded snapshots using `partitioner`.
    pub fn compute_dem_metrics(
        &self,
        partitioner: &dyn Partitioner,
        species_labels: &[bool],
    ) -> Result<DemMetricsTrajectory> {
        if self.dem_frames.is_empty() {
            return Err(DemMcmError::MissingData(
                "No DEM snapshots loaded".to_string(),
            ));
        }
        let n_snaps = self.dem_frames.len();
        let n_states = partitioner.n_cells();
        let mut times = Vec::with_capacity(n_snaps);
        let mut s_small = Matrix::zeros(n_snaps, n_states);
        let mut s_large = Matrix::zeros(n_snaps, n_states);

        for (t, (&ts, frame)) in self.dem_frames.iter().enumerate() {
            times.push(ts as i64);
            let vel = if frame.velocities.len() == frame.coordinates.len() {
                Some(frame.velocities.as_slice())
            } else {
                None
            };
            let states = partitioner.compute_states(&frame.coordinates, vel)?;
            for (p, &st) in states.iter().enumerate() {
                if st >= 0 && (st as usize) < n_states {
                    let s = st as usize;
                    if species_labels[p] {
                        s_small.set(t, s, s_small.get(t, s) + 1.0);
                    } else {
                        s_large.set(t, s, s_large.get(t, s) + 1.0);
                    }
                }
            }
        }

        let act = vec![true; n_states];
        let rsd = rsd_concentration(&s_small, &s_large, &act, &act);
        let ent = entropy_concentration(&s_small, &s_large, &act, &act, true);
        let iseg = intensity_of_segregation(&s_small, &s_large, &act, &act);
        Ok((times, s_small, s_large, rsd, ent, iseg))
    }

    /// Evaluate a stored homogeneous or inhomogeneous experiment against its own
    /// DEM state matrices `S_matrix_{small,large}` at Markov timesteps `times_markov`.
    pub fn evaluate_experiment(&self, exp: &ExperimentData) -> Result<ComparisonMetrics> {
        let da = exp
            .species
            .get("small")
            .ok_or_else(|| DemMcmError::MissingData("Experiment missing 'small' species".into()))?;
        let db = exp
            .species
            .get("large")
            .ok_or_else(|| DemMcmError::MissingData("Experiment missing 'large' species".into()))?;

        let start_idx = exp
            .config
            .get("start_index")
            .and_then(|v| v.as_i64())
            .unwrap_or(250);
        let tau = exp
            .config
            .get("tau")
            .and_then(|v| v.as_usize())
            .unwrap_or(50);
        let step = exp.config.get("step").and_then(|v| v.as_usize());
        let nlt = exp.config.get("nlt").and_then(|v| v.as_usize());

        let row0 = da.times.partition_point(|&t| t < start_idx);
        if row0 >= da.s_matrix.rows || row0 >= db.s_matrix.rows {
            return Err(DemMcmError::MissingData(
                "start_index beyond S_matrix length".to_string(),
            ));
        }

        let (traj_s, traj_l, act_s, act_l, times_markov) = if exp.inhomogeneous {
            let pb_s = da.p_blocks.as_ref().ok_or_else(|| {
                DemMcmError::MissingData("Inhomogeneous experiment missing P_blocks_small".into())
            })?;
            let pb_l = db.p_blocks.as_ref().ok_or_else(|| {
                DemMcmError::MissingData("Inhomogeneous experiment missing P_blocks_large".into())
            })?;
            let (_, act_s) = clean_transition_matrix(&pb_s[0], 0.5);
            let (_, act_l) = clean_transition_matrix(&pb_l[0], 0.5);
            let (ts, tm) = propagate_markov_inhomogeneous(
                da.s_matrix.row(row0),
                pb_s,
                &da.times,
                start_idx,
                tau,
                &act_s,
                step,
                nlt,
            )?;
            let (tl, _) = propagate_markov_inhomogeneous(
                db.s_matrix.row(row0),
                pb_l,
                &db.times,
                start_idx,
                tau,
                &act_l,
                step,
                nlt,
            )?;
            (ts, tl, act_s, act_l, tm)
        } else {
            let p_s = da.p.as_ref().ok_or_else(|| {
                DemMcmError::MissingData("Homogeneous experiment missing P for small".into())
            })?;
            let p_l = db.p.as_ref().ok_or_else(|| {
                DemMcmError::MissingData("Homogeneous experiment missing P for large".into())
            })?;
            let (pc_s, act_s) = clean_transition_matrix(p_s, 0.5);
            let (pc_l, act_l) = clean_transition_matrix(p_l, 0.5);
            let (ts, tm) = propagate_markov(
                da.s_matrix.row(row0),
                &pc_s,
                &da.times,
                start_idx,
                tau,
                &act_s,
            )?;
            let (tl, _) = propagate_markov(
                db.s_matrix.row(row0),
                &pc_l,
                &db.times,
                start_idx,
                tau,
                &act_l,
                // wait
            )?;
            (ts, tl, act_s, act_l, tm)
        };

        // Slice DEM S_matrices at the exact `times_markov` indices
        let time_to_row: HashMap<i64, usize> =
            da.times.iter().enumerate().map(|(r, &t)| (t, r)).collect();
        let dem_rows: Vec<usize> = times_markov
            .iter()
            .filter_map(|t| time_to_row.get(t).copied())
            .collect();
        let s_dem_s = da.s_matrix.select_rows(&dem_rows);
        let s_dem_l = db.s_matrix.select_rows(&dem_rows);

        let rsd_dem = rsd_concentration(&s_dem_s, &s_dem_l, &act_s, &act_l);
        let rsd_markov = rsd_concentration(&traj_s, &traj_l, &act_s, &act_l);
        let entropy_dem = entropy_concentration(&s_dem_s, &s_dem_l, &act_s, &act_l, true);
        let entropy_markov = entropy_concentration(&traj_s, &traj_l, &act_s, &act_l, true);
        let intensity_dem = intensity_of_segregation(&s_dem_s, &s_dem_l, &act_s, &act_l);
        let intensity_markov = intensity_of_segregation(&traj_s, &traj_l, &act_s, &act_l);

        let n_cmp = rsd_dem.len().min(rsd_markov.len());
        let (rmse_rsd, mae_rsd) = if n_cmp > 0 {
            let sq: f64 = rsd_dem[..n_cmp]
                .iter()
                .zip(rsd_markov[..n_cmp].iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum();
            let ab: f64 = rsd_dem[..n_cmp]
                .iter()
                .zip(rsd_markov[..n_cmp].iter())
                .map(|(&a, &b)| (a - b).abs())
                .sum();
            ((sq / n_cmp as f64).sqrt(), ab / n_cmp as f64)
        } else {
            (0.0, 0.0)
        };

        let times_sec: Vec<f64> = times_markov
            .iter()
            .map(|&t| t as f64 * TIMESTEP_TO_SECONDS)
            .collect();
        let mt_dem = mixing_times(&rsd_dem, &times_sec, &[0.5, 0.1]);
        let mt_mk = mixing_times(&rsd_markov, &times_sec, &[0.5, 0.1]);

        Ok(ComparisonMetrics {
            rsd_dem,
            rsd_markov,
            entropy_dem,
            entropy_markov,
            intensity_dem,
            intensity_markov,
            times_markov,
            rmse_rsd,
            mae_rsd,
            t50_dem: mt_dem[0].1,
            t90_dem: mt_dem[1].1,
            t50_markov: mt_mk[0].1,
            t90_markov: mt_mk[1].1,
        })
    }
}
