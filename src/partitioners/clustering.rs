//! Unsupervised clustering spatial partitioners:
//! - [`SpectralClusteringPartitioner`]
//! - [`DbscanPartitioner`]
//! - [`GaussianMixturePartitioner`]
//! - [`SpectralBiclusteringPartitioner`]

use std::fs;
use std::path::Path;

use crate::config::PartitioningMethod;
use crate::error::{DemMcmError, Result};
use crate::io::json::JsonValue;
use crate::io::npy::NpyArray;
use crate::linalg::Matrix;
use crate::ml::{
    Dbscan, GaussianMixture, GaussianMixtureModel, SpectralBiclustering, SpectralClustering,
};
use crate::partitioners::physics::{
    apply_saved_scaling, compute_relative_velocity, scale_and_weight_features,
};
use crate::partitioners::{Partitioner, PartitionerArtifacts};
use crate::rng::Rng;
use crate::spatial::KdTree;

// ============================================================================
// 8. SPECTRAL CLUSTERING PARTITIONER
// ============================================================================

#[derive(Debug, Clone)]
pub struct SpectralClusteringPartitioner {
    pub n_cells_cfg: usize,
    pub velocity_weight: f64,
    pub n_neighbors: usize,
    pub max_samples: usize,
    pub random_state: u64,
    pub use_velocity: bool,
    pub dem_velocities: Option<Vec<[f64; 3]>>,

    pub support_data: Option<Matrix>,
    pub support_labels: Option<Vec<i64>>,
    pub mean: Option<Vec<f64>>,
    pub std: Option<Vec<f64>>,
    pub n_features: usize,
    tree: Option<KdTree>,
}

impl SpectralClusteringPartitioner {
    pub fn new(
        n_cells: usize,
        velocity_weight: f64,
        n_neighbors: usize,
        max_samples: usize,
        random_state: u64,
    ) -> Self {
        Self {
            n_cells_cfg: n_cells,
            velocity_weight,
            n_neighbors,
            max_samples,
            random_state,
            use_velocity: true,
            dem_velocities: None,
            support_data: None,
            support_labels: None,
            mean: None,
            std: None,
            n_features: 4,
            tree: None,
        }
    }

    pub fn fit_with_velocities(
        &mut self,
        coordinates: &[[f64; 3]],
        use_velocities: Option<bool>,
    ) -> Result<()> {
        if coordinates.is_empty() {
            return Err(DemMcmError::InvalidArgument(
                "SpectralClusteringPartitioner::fit requires non-empty coordinates".to_string(),
            ));
        }
        let use_vel = use_velocities.unwrap_or(self.use_velocity);
        let features = if use_vel
            && self
                .dem_velocities
                .as_ref()
                .is_some_and(|v| v.len() == coordinates.len())
        {
            let vel = self.dem_velocities.as_ref().unwrap();
            let v_rel = compute_relative_velocity(coordinates, vel, 4.0);
            let mut m = Matrix::zeros(coordinates.len(), 4);
            for i in 0..coordinates.len() {
                let spd = (v_rel[i][0] * v_rel[i][0]
                    + v_rel[i][1] * v_rel[i][1]
                    + v_rel[i][2] * v_rel[i][2])
                    .sqrt();
                m.set(i, 0, coordinates[i][0]);
                m.set(i, 1, coordinates[i][1]);
                m.set(i, 2, coordinates[i][2]);
                m.set(i, 3, spd);
            }
            m
        } else {
            let mut m = Matrix::zeros(coordinates.len(), 3);
            for (i, c) in coordinates.iter().enumerate() {
                m.set(i, 0, c[0]);
                m.set(i, 1, c[1]);
                m.set(i, 2, c[2]);
            }
            m
        };

        self.n_features = features.cols;
        let (x_scaled, mean, std) = scale_and_weight_features(&features, 3, self.velocity_weight);
        self.mean = Some(mean);
        self.std = Some(std);

        let mut rng = Rng::seed_from_u64(self.random_state);
        let n_sub = self.max_samples.min(x_scaled.rows);
        let idx = rng.sample_indices(x_scaled.rows, n_sub);
        let x_sub = x_scaled.select_rows(&idx);

        let sc = SpectralClustering {
            n_clusters: self.n_cells_cfg.min(x_sub.rows).max(1),
            n_neighbors: self.n_neighbors,
            gamma: 1.0,
            random_state: self.random_state,
        };
        let labels = sc.fit_predict(&x_sub)?;
        let labels_i64: Vec<i64> = labels.into_iter().map(|l| l as i64).collect();
        self.tree = Some(KdTree::build(&x_sub.data, x_sub.cols));
        self.support_data = Some(x_sub);
        self.support_labels = Some(labels_i64);
        Ok(())
    }
}

impl Partitioner for SpectralClusteringPartitioner {
    fn method(&self) -> PartitioningMethod {
        PartitioningMethod::Spectral
    }

    fn type_name(&self) -> &'static str {
        "SpectralClusteringPartitioner"
    }

    fn n_cells(&self) -> usize {
        self.n_cells_cfg
    }

    fn uses_velocity(&self) -> bool {
        self.use_velocity
    }

    fn set_dem_velocities(&mut self, velocities: Option<&[[f64; 3]]>) {
        self.dem_velocities = velocities.map(|v| v.to_vec());
    }

    fn set_use_velocity(&mut self, use_vel: bool) {
        self.use_velocity = use_vel;
    }

    fn label(&self) -> String {
        let uv = if self.use_velocity { "True" } else { "False" };
        format!(
            "spectral_{}cells_vw{}_k{}_{uv}_{}",
            self.n_cells_cfg, self.velocity_weight, self.n_neighbors, self.n_features
        )
    }

    fn fit(&mut self, coordinates: &[[f64; 3]]) -> Result<()> {
        self.fit_with_velocities(coordinates, None)
    }

    fn compute_states(
        &self,
        coordinates: &[[f64; 3]],
        velocities: Option<&[[f64; 3]]>,
    ) -> Result<Vec<i64>> {
        let (tree, labels, mean, std) =
            match (&self.tree, &self.support_labels, &self.mean, &self.std) {
                (Some(t), Some(l), Some(m), Some(s)) => (t, l, m, s),
                _ => {
                    return Err(DemMcmError::NotFitted(
                        "SpectralClusteringPartitioner must be fitted before use".to_string(),
                    ))
                }
            };

        let mut features = Matrix::zeros(coordinates.len(), self.n_features);
        let v_rel_opt = velocities
            .filter(|v| v.len() == coordinates.len())
            .map(|v| compute_relative_velocity(coordinates, v, 4.0));
        for (i, c) in coordinates.iter().enumerate() {
            features.set(i, 0, c[0]);
            features.set(i, 1, c[1]);
            features.set(i, 2, c[2]);
            if self.n_features == 4 {
                let spd = v_rel_opt.as_ref().map_or(0.0, |vr| {
                    (vr[i][0] * vr[i][0] + vr[i][1] * vr[i][1] + vr[i][2] * vr[i][2]).sqrt()
                });
                features.set(i, 3, spd);
            }
        }
        let x_scaled = apply_saved_scaling(&features, mean, std, 3, self.velocity_weight);
        let mut states = Vec::with_capacity(coordinates.len());
        for r in 0..x_scaled.rows {
            let (idx, _) = tree.query_nearest(x_scaled.row(r)).unwrap();
            states.push(labels[idx]);
        }
        Ok(states)
    }

    fn save_artifacts(&self) -> Result<PartitionerArtifacts> {
        let (sd, sl, m, s) = match (
            &self.support_data,
            &self.support_labels,
            &self.mean,
            &self.std,
        ) {
            (Some(sd), Some(sl), Some(m), Some(s)) => (sd, sl, m, s),
            _ => {
                return Err(DemMcmError::NotFitted(
                    "SpectralClusteringPartitioner must be fitted before save()".to_string(),
                ))
            }
        };
        let mut art = PartitionerArtifacts::default();
        art.arrays
            .insert("support_data".to_string(), NpyArray::from_matrix(sd));
        art.arrays.insert(
            "support_labels".to_string(),
            NpyArray::from_i64_vec(sl.clone()),
        );
        art.arrays
            .insert("mean".to_string(), NpyArray::from_f64_vec(m.clone()));
        art.arrays
            .insert("std".to_string(), NpyArray::from_f64_vec(s.clone()));
        let mut params = JsonValue::object();
        params.insert("n_features", JsonValue::Number(self.n_features as f64));
        params.insert("n_cells", JsonValue::Number(self.n_cells_cfg as f64));
        params.insert("vw", JsonValue::Number(self.velocity_weight));
        params.insert("k", JsonValue::Number(self.n_neighbors as f64));
        art.json_files.insert("params".to_string(), params);
        Ok(art)
    }

    fn load(&mut self, dir: &Path) -> Result<()> {
        let sd = NpyArray::load(&dir.join("support_data.npy"))?.to_matrix()?;
        let sl = NpyArray::load(&dir.join("support_labels.npy"))?.to_i64_vec()?;
        let m = NpyArray::load(&dir.join("mean.npy"))?.to_f64_vec();
        let s = NpyArray::load(&dir.join("std.npy"))?.to_f64_vec();
        self.tree = Some(KdTree::build(&sd.data, sd.cols));
        self.support_data = Some(sd);
        self.support_labels = Some(sl);
        self.mean = Some(m);
        self.std = Some(s);
        let p = JsonValue::parse(&fs::read_to_string(dir.join("params.json"))?)?;
        self.n_features = p.get("n_features").and_then(|v| v.as_usize()).unwrap_or(3);
        Ok(())
    }
}

// ============================================================================
// 9. DBSCAN PARTITIONER
// ============================================================================

#[derive(Debug, Clone)]
pub struct DbscanPartitioner {
    pub eps: f64,
    pub min_samples: usize,
    pub velocity_weight: f64,
    pub max_samples: usize,
    pub random_state: u64,
    pub use_velocity: bool,
    pub dem_velocities: Option<Vec<[f64; 3]>>,

    pub support_data: Option<Matrix>,
    pub support_labels: Option<Vec<i64>>,
    pub mean: Option<Vec<f64>>,
    pub std: Option<Vec<f64>>,
    pub n_features: usize,
    pub n_cells_found: usize,
    tree: Option<KdTree>,
}

impl DbscanPartitioner {
    pub fn new(
        eps: f64,
        min_samples: usize,
        velocity_weight: f64,
        max_samples: usize,
        random_state: u64,
    ) -> Self {
        Self {
            eps,
            min_samples,
            velocity_weight,
            max_samples,
            random_state,
            use_velocity: false,
            dem_velocities: None,
            support_data: None,
            support_labels: None,
            mean: None,
            std: None,
            n_features: 6,
            n_cells_found: 0,
            tree: None,
        }
    }

    pub fn fit_with_velocities(
        &mut self,
        coordinates: &[[f64; 3]],
        use_velocities: Option<bool>,
    ) -> Result<()> {
        if coordinates.is_empty() {
            return Err(DemMcmError::InvalidArgument(
                "DbscanPartitioner::fit requires non-empty coordinates".to_string(),
            ));
        }
        let use_vel = use_velocities.unwrap_or(self.use_velocity);
        let features = if use_vel
            && self
                .dem_velocities
                .as_ref()
                .is_some_and(|v| v.len() == coordinates.len())
        {
            let vel = self.dem_velocities.as_ref().unwrap();
            let mut m = Matrix::zeros(coordinates.len(), 6);
            for i in 0..coordinates.len() {
                m.set(i, 0, coordinates[i][0]);
                m.set(i, 1, coordinates[i][1]);
                m.set(i, 2, coordinates[i][2]);
                m.set(i, 3, vel[i][0]);
                m.set(i, 4, vel[i][1]);
                m.set(i, 5, vel[i][2]);
            }
            m
        } else {
            let mut m = Matrix::zeros(coordinates.len(), 3);
            for (i, c) in coordinates.iter().enumerate() {
                m.set(i, 0, c[0]);
                m.set(i, 1, c[1]);
                m.set(i, 2, c[2]);
            }
            m
        };

        self.n_features = features.cols;
        let (x_scaled, mean, std) = scale_and_weight_features(&features, 3, self.velocity_weight);
        self.mean = Some(mean);
        self.std = Some(std);

        let mut rng = Rng::seed_from_u64(self.random_state);
        let n_sub = self.max_samples.min(x_scaled.rows);
        let idx = rng.sample_indices(x_scaled.rows, n_sub);
        let x_sub = x_scaled.select_rows(&idx);

        let db = Dbscan::new(self.eps, self.min_samples);
        let (labels_sub, n_clusters) = db.fit_predict(&x_sub);
        if n_clusters == 0 {
            return Err(DemMcmError::InvalidArgument(format!(
                "DBSCAN found no cluster (everything is noise). Increase eps (current={}) or decrease min_samples (current={}).",
                self.eps, self.min_samples
            )));
        }
        self.n_cells_found = n_clusters;

        self.tree = Some(KdTree::build(&x_sub.data, x_sub.cols));
        self.support_data = Some(x_sub);
        self.support_labels = Some(labels_sub);
        Ok(())
    }
}

impl Partitioner for DbscanPartitioner {
    fn method(&self) -> PartitioningMethod {
        PartitioningMethod::Dbscan
    }

    fn type_name(&self) -> &'static str {
        "DBSCANPartitioner"
    }

    fn n_cells(&self) -> usize {
        self.n_cells_found
    }

    fn uses_velocity(&self) -> bool {
        self.use_velocity
    }

    fn set_dem_velocities(&mut self, velocities: Option<&[[f64; 3]]>) {
        self.dem_velocities = velocities.map(|v| v.to_vec());
    }

    fn set_use_velocity(&mut self, use_vel: bool) {
        self.use_velocity = use_vel;
    }

    fn label(&self) -> String {
        format!(
            "dbscan_eps{}_min{}_vw{}",
            self.eps, self.min_samples, self.velocity_weight
        )
    }

    fn fit(&mut self, coordinates: &[[f64; 3]]) -> Result<()> {
        self.fit_with_velocities(coordinates, None)
    }

    fn compute_states(
        &self,
        coordinates: &[[f64; 3]],
        velocities: Option<&[[f64; 3]]>,
    ) -> Result<Vec<i64>> {
        let (tree, labels, mean, std) =
            match (&self.tree, &self.support_labels, &self.mean, &self.std) {
                (Some(t), Some(l), Some(m), Some(s)) => (t, l, m, s),
                _ => {
                    return Err(DemMcmError::NotFitted(
                        "DbscanPartitioner must be fitted before use".to_string(),
                    ))
                }
            };

        let mut features = Matrix::zeros(coordinates.len(), self.n_features);
        for (i, c) in coordinates.iter().enumerate() {
            features.set(i, 0, c[0]);
            features.set(i, 1, c[1]);
            features.set(i, 2, c[2]);
            if self.n_features == 6 {
                let v = velocities
                    .filter(|vs| vs.len() == coordinates.len())
                    .map_or([0.0; 3], |vs| vs[i]);
                features.set(i, 3, v[0]);
                features.set(i, 4, v[1]);
                features.set(i, 5, v[2]);
            }
        }
        let x_scaled = apply_saved_scaling(&features, mean, std, 3, self.velocity_weight);
        let mut states = Vec::with_capacity(coordinates.len());
        for r in 0..x_scaled.rows {
            let (idx, _) = tree.query_nearest(x_scaled.row(r)).unwrap();
            states.push(labels[idx]);
        }
        Ok(states)
    }

    fn save_artifacts(&self) -> Result<PartitionerArtifacts> {
        let (sd, sl, m, s) = match (
            &self.support_data,
            &self.support_labels,
            &self.mean,
            &self.std,
        ) {
            (Some(sd), Some(sl), Some(m), Some(s)) => (sd, sl, m, s),
            _ => {
                return Err(DemMcmError::NotFitted(
                    "DbscanPartitioner must be fitted before save()".to_string(),
                ))
            }
        };
        let mut art = PartitionerArtifacts::default();
        art.arrays
            .insert("support_data".to_string(), NpyArray::from_matrix(sd));
        art.arrays.insert(
            "support_labels".to_string(),
            NpyArray::from_i64_vec(sl.clone()),
        );
        art.arrays
            .insert("mean".to_string(), NpyArray::from_f64_vec(m.clone()));
        art.arrays
            .insert("std".to_string(), NpyArray::from_f64_vec(s.clone()));
        let mut params = JsonValue::object();
        params.insert("n_features", JsonValue::Number(self.n_features as f64));
        params.insert("n_cells", JsonValue::Number(self.n_cells_found as f64));
        params.insert("eps", JsonValue::Number(self.eps));
        params.insert("min_samples", JsonValue::Number(self.min_samples as f64));
        params.insert("vw", JsonValue::Number(self.velocity_weight));
        art.json_files.insert("params".to_string(), params);
        Ok(art)
    }

    fn load(&mut self, dir: &Path) -> Result<()> {
        let sd = NpyArray::load(&dir.join("support_data.npy"))?.to_matrix()?;
        let sl = NpyArray::load(&dir.join("support_labels.npy"))?.to_i64_vec()?;
        let m = NpyArray::load(&dir.join("mean.npy"))?.to_f64_vec();
        let s = NpyArray::load(&dir.join("std.npy"))?.to_f64_vec();
        self.tree = Some(KdTree::build(&sd.data, sd.cols));
        self.support_data = Some(sd);
        self.support_labels = Some(sl);
        self.mean = Some(m);
        self.std = Some(s);
        let p = JsonValue::parse(&fs::read_to_string(dir.join("params.json"))?)?;
        self.n_features = p.get("n_features").and_then(|v| v.as_usize()).unwrap_or(3);
        self.n_cells_found = p.get("n_cells").and_then(|v| v.as_usize()).unwrap_or(0);
        if let Some(eps) = p.get("eps").and_then(|v| v.as_f64()) {
            self.eps = eps;
        }
        if let Some(ms) = p.get("min_samples").and_then(|v| v.as_usize()) {
            self.min_samples = ms;
        }
        if let Some(vw) = p.get("vw").and_then(|v| v.as_f64()) {
            self.velocity_weight = vw;
        }
        Ok(())
    }
}

// ============================================================================
// 10. GAUSSIAN MIXTURE MODEL PARTITIONER
// ============================================================================

#[derive(Debug, Clone)]
pub struct GaussianMixturePartitioner {
    pub n_cells_cfg: usize,
    pub velocity_weight: f64,
    pub random_state: u64,
    pub max_fit_samples: usize,
    pub use_velocity: bool,
    pub dem_velocities: Option<Vec<[f64; 3]>>,

    pub gmm: Option<GaussianMixtureModel>,
    pub mean: Option<Vec<f64>>,
    pub std: Option<Vec<f64>>,
    pub n_features: usize,
}

impl GaussianMixturePartitioner {
    pub fn new(
        n_cells: usize,
        velocity_weight: f64,
        random_state: u64,
        max_fit_samples: usize,
    ) -> Self {
        Self {
            n_cells_cfg: n_cells,
            velocity_weight,
            random_state,
            max_fit_samples,
            use_velocity: true,
            dem_velocities: None,
            gmm: None,
            mean: None,
            std: None,
            n_features: 6,
        }
    }

    pub fn fit_with_velocities(
        &mut self,
        coordinates: &[[f64; 3]],
        use_velocities: Option<bool>,
    ) -> Result<()> {
        if coordinates.is_empty() {
            return Err(DemMcmError::InvalidArgument(
                "GaussianMixturePartitioner::fit requires non-empty coordinates".to_string(),
            ));
        }
        let use_vel = use_velocities.unwrap_or(self.use_velocity);
        let features = if use_vel
            && self
                .dem_velocities
                .as_ref()
                .is_some_and(|v| v.len() == coordinates.len())
        {
            let vel = self.dem_velocities.as_ref().unwrap();
            let mut m = Matrix::zeros(coordinates.len(), 6);
            for i in 0..coordinates.len() {
                m.set(i, 0, coordinates[i][0]);
                m.set(i, 1, coordinates[i][1]);
                m.set(i, 2, coordinates[i][2]);
                m.set(i, 3, vel[i][0]);
                m.set(i, 4, vel[i][1]);
                m.set(i, 5, vel[i][2]);
            }
            m
        } else {
            let mut m = Matrix::zeros(coordinates.len(), 3);
            for (i, c) in coordinates.iter().enumerate() {
                m.set(i, 0, c[0]);
                m.set(i, 1, c[1]);
                m.set(i, 2, c[2]);
            }
            m
        };

        self.n_features = features.cols;
        let (x_scaled, mean, std) = scale_and_weight_features(&features, 3, self.velocity_weight);
        self.mean = Some(mean);
        self.std = Some(std);

        let mut rng = Rng::seed_from_u64(self.random_state);
        let x_fit = if x_scaled.rows > self.max_fit_samples {
            let idx = rng.sample_indices(x_scaled.rows, self.max_fit_samples);
            x_scaled.select_rows(&idx)
        } else {
            x_scaled
        };

        let k = self.n_cells_cfg.min(x_fit.rows).max(1);
        let mut gmm = GaussianMixture::new(k).with_random_state(self.random_state);
        gmm.max_iter = 50;
        gmm.tol = 1e-3;
        let model = gmm.fit(&x_fit)?;
        self.gmm = Some(model);
        Ok(())
    }
}

impl Partitioner for GaussianMixturePartitioner {
    fn method(&self) -> PartitioningMethod {
        PartitioningMethod::Gmm
    }

    fn type_name(&self) -> &'static str {
        "GaussianMixturePartitioner"
    }

    fn splitting_method(&self) -> String {
        "gmm_full".to_string()
    }

    fn n_cells(&self) -> usize {
        self.n_cells_cfg
    }

    fn uses_velocity(&self) -> bool {
        self.use_velocity
    }

    fn set_dem_velocities(&mut self, velocities: Option<&[[f64; 3]]>) {
        self.dem_velocities = velocities.map(|v| v.to_vec());
    }

    fn set_use_velocity(&mut self, use_vel: bool) {
        self.use_velocity = use_vel;
    }

    fn label(&self) -> String {
        format!(
            "gmm_full_{}cells_vw{}",
            self.n_cells_cfg, self.velocity_weight
        )
    }

    fn fit(&mut self, coordinates: &[[f64; 3]]) -> Result<()> {
        self.fit_with_velocities(coordinates, None)
    }

    fn compute_states(
        &self,
        coordinates: &[[f64; 3]],
        velocities: Option<&[[f64; 3]]>,
    ) -> Result<Vec<i64>> {
        let (gmm, mean, std) = match (&self.gmm, &self.mean, &self.std) {
            (Some(g), Some(m), Some(s)) => (g, m, s),
            _ => {
                return Err(DemMcmError::NotFitted(
                    "GaussianMixturePartitioner must be fitted before use".to_string(),
                ))
            }
        };

        let mut features = Matrix::zeros(coordinates.len(), self.n_features);
        for (i, c) in coordinates.iter().enumerate() {
            features.set(i, 0, c[0]);
            features.set(i, 1, c[1]);
            features.set(i, 2, c[2]);
            if self.n_features == 6 {
                let v = velocities
                    .filter(|vs| vs.len() == coordinates.len())
                    .map_or([0.0; 3], |vs| vs[i]);
                features.set(i, 3, v[0]);
                features.set(i, 4, v[1]);
                features.set(i, 5, v[2]);
            }
        }
        let x_scaled = apply_saved_scaling(&features, mean, std, 3, self.velocity_weight);
        let preds = gmm.predict(&x_scaled)?;
        Ok(preds.into_iter().map(|p| p as i64).collect())
    }

    fn save_artifacts(&self) -> Result<PartitionerArtifacts> {
        let (gmm, m, s) = match (&self.gmm, &self.mean, &self.std) {
            (Some(g), Some(m), Some(s)) => (g, m, s),
            _ => {
                return Err(DemMcmError::NotFitted(
                    "GaussianMixturePartitioner must be fitted before save()".to_string(),
                ))
            }
        };
        let mut art = PartitionerArtifacts::default();
        art.arrays
            .insert("mean".to_string(), NpyArray::from_f64_vec(m.clone()));
        art.arrays
            .insert("std".to_string(), NpyArray::from_f64_vec(s.clone()));
        art.arrays
            .insert("gmm_means".to_string(), NpyArray::from_matrix(&gmm.means));
        art.arrays.insert(
            "gmm_weights".to_string(),
            NpyArray::from_f64_vec(gmm.weights.clone()),
        );
        art.arrays.insert(
            "gmm_covariances".to_string(),
            NpyArray::from_matrix_3d(&gmm.covariances)?,
        );
        let mut params = JsonValue::object();
        params.insert("n_features", JsonValue::Number(self.n_features as f64));
        params.insert("n_cells", JsonValue::Number(self.n_cells_cfg as f64));
        params.insert("vw", JsonValue::Number(self.velocity_weight));
        art.json_files.insert("gmm_params".to_string(), params);
        Ok(art)
    }

    fn load(&mut self, dir: &Path) -> Result<()> {
        self.mean = Some(NpyArray::load(&dir.join("mean.npy"))?.to_f64_vec());
        self.std = Some(NpyArray::load(&dir.join("std.npy"))?.to_f64_vec());
        let means = NpyArray::load(&dir.join("gmm_means.npy"))?.to_matrix()?;
        let weights = NpyArray::load(&dir.join("gmm_weights.npy"))?.to_f64_vec();
        let covs = NpyArray::load(&dir.join("gmm_covariances.npy"))?.to_matrix_3d()?;
        let p = JsonValue::parse(&fs::read_to_string(dir.join("gmm_params.json"))?)?;
        self.n_features = p.get("n_features").and_then(|v| v.as_usize()).unwrap_or(3);
        let model = GaussianMixtureModel::from_params(weights, means, covs, 1e-6)?;
        self.gmm = Some(model);
        Ok(())
    }
}

// ============================================================================
// 11. SPECTRAL BICLUSTERING PARTITIONER
// ============================================================================

#[derive(Debug, Clone)]
pub struct SpectralBiclusteringPartitioner {
    pub n_cells_cfg: usize,
    pub n_col_clusters: usize,
    pub method_name: String,
    pub velocity_weight: f64,
    pub n_components: usize,
    pub n_best: usize,
    pub max_samples: usize,
    pub random_state: u64,
    pub use_velocity: bool,
    pub dem_velocities: Option<Vec<[f64; 3]>>,

    pub support_data: Option<Matrix>,
    pub support_labels: Option<Vec<i64>>,
    pub col_labels: Option<Vec<i64>>,
    pub mean: Option<Vec<f64>>,
    pub std: Option<Vec<f64>>,
    pub log_shift: Option<Vec<f64>>,
    pub n_features: usize,
    tree: Option<KdTree>,
}

impl SpectralBiclusteringPartitioner {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_cells: usize,
        n_col_clusters: usize,
        method: impl Into<String>,
        velocity_weight: f64,
        n_components: usize,
        n_best: usize,
        max_samples: usize,
        random_state: u64,
    ) -> Self {
        Self {
            n_cells_cfg: n_cells,
            n_col_clusters,
            method_name: method.into(),
            velocity_weight,
            n_components,
            n_best,
            max_samples,
            random_state,
            use_velocity: true,
            dem_velocities: None,
            support_data: None,
            support_labels: None,
            col_labels: None,
            mean: None,
            std: None,
            log_shift: None,
            n_features: 7,
            tree: None,
        }
    }

    fn build_features(coords: &[[f64; 3]], velocities: Option<&[[f64; 3]]>) -> Matrix {
        let has_vel = velocities.is_some_and(|v| v.len() == coords.len());
        let ncols = if has_vel { 7 } else { 3 };
        let mut m = Matrix::zeros(coords.len(), ncols);
        for (i, c) in coords.iter().enumerate() {
            let r_cyl = (c[0] * c[0] + c[1] * c[1]).sqrt();
            let theta = c[1].atan2(c[0]);
            m.set(i, 0, r_cyl);
            m.set(i, 1, theta);
            m.set(i, 2, c[2]);
            if let Some(vs) = velocities.filter(|_| has_vel) {
                let v = vs[i];
                let v_norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                m.set(i, 3, v_norm);
                m.set(i, 4, v[0]);
                m.set(i, 5, v[1]);
                m.set(i, 6, v[2]);
            }
        }
        m
    }
}

impl Partitioner for SpectralBiclusteringPartitioner {
    fn method(&self) -> PartitioningMethod {
        PartitioningMethod::SpectralBiclustering
    }

    fn type_name(&self) -> &'static str {
        "SpectralBiclusteringPartitioner"
    }

    fn n_cells(&self) -> usize {
        self.n_cells_cfg
    }

    fn uses_velocity(&self) -> bool {
        self.use_velocity
    }

    fn set_dem_velocities(&mut self, velocities: Option<&[[f64; 3]]>) {
        self.dem_velocities = velocities.map(|v| v.to_vec());
    }

    fn set_use_velocity(&mut self, use_vel: bool) {
        self.use_velocity = use_vel;
    }

    fn label(&self) -> String {
        format!(
            "spectral_biclustering_{}cells_nc{}_m{}_vw{}",
            self.n_cells_cfg, self.n_col_clusters, self.method_name, self.velocity_weight
        )
    }

    fn fit(&mut self, coordinates: &[[f64; 3]]) -> Result<()> {
        if coordinates.is_empty() {
            return Err(DemMcmError::InvalidArgument(
                "SpectralBiclusteringPartitioner::fit requires non-empty coordinates".to_string(),
            ));
        }
        let vel = self
            .dem_velocities
            .as_deref()
            .filter(|v| v.len() == coordinates.len());
        let features = Self::build_features(coordinates, vel);
        self.n_features = features.cols;

        let (x_scaled, mean, std) = scale_and_weight_features(&features, 3, self.velocity_weight);
        self.mean = Some(mean);
        self.std = Some(std);

        let mut rng = Rng::seed_from_u64(self.random_state);
        let n_sub = self.max_samples.min(x_scaled.rows);
        let idx = rng.sample_indices(x_scaled.rows, n_sub);
        let mut x_sub = x_scaled.select_rows(&idx);

        if self.method_name == "log" {
            let mut shift = vec![f64::INFINITY; x_sub.cols];
            for r in 0..x_sub.rows {
                for (c, s) in shift.iter_mut().enumerate() {
                    *s = s.min(x_sub.get(r, c));
                }
            }
            for r in 0..x_sub.rows {
                for (c, s) in shift.iter().enumerate() {
                    x_sub.set(r, c, x_sub.get(r, c) - s + 1e-6);
                }
            }
            self.log_shift = Some(shift);
        } else {
            self.log_shift = None;
        }

        let sb = SpectralBiclustering {
            n_row_clusters: self.n_cells_cfg.min(x_sub.rows).max(1),
            n_col_clusters: self.n_col_clusters.min(x_sub.cols).max(1),
            method: self.method_name.clone(),
            n_components: self
                .n_components
                .max(self.n_cells_cfg + self.n_col_clusters),
            n_best: self.n_best,
            random_state: self.random_state,
        };
        let sb_res = sb.fit_predict(&x_sub)?;
        let row_i64: Vec<i64> = sb_res.row_labels.into_iter().map(|v| v as i64).collect();
        let col_i64: Vec<i64> = sb_res.column_labels.into_iter().map(|v| v as i64).collect();

        self.tree = Some(KdTree::build(&x_sub.data, x_sub.cols));
        self.support_data = Some(x_sub);
        self.support_labels = Some(row_i64);
        self.col_labels = Some(col_i64);
        Ok(())
    }

    fn compute_states(
        &self,
        coordinates: &[[f64; 3]],
        velocities: Option<&[[f64; 3]]>,
    ) -> Result<Vec<i64>> {
        let (tree, labels, mean, std) =
            match (&self.tree, &self.support_labels, &self.mean, &self.std) {
                (Some(t), Some(l), Some(m), Some(s)) => (t, l, m, s),
                _ => {
                    return Err(DemMcmError::NotFitted(
                        "SpectralBiclusteringPartitioner must be fitted before use".to_string(),
                    ))
                }
            };

        let zero_vel;
        let vel_arg = if self.n_features == 7 {
            if let Some(vs) = velocities.filter(|v| v.len() == coordinates.len()) {
                Some(vs)
            } else {
                zero_vel = vec![[0.0; 3]; coordinates.len()];
                Some(zero_vel.as_slice())
            }
        } else {
            None
        };
        let features = Self::build_features(coordinates, vel_arg);
        let mut x_scaled = apply_saved_scaling(&features, mean, std, 3, self.velocity_weight);

        if self.method_name == "log" {
            if let Some(shift) = &self.log_shift {
                for r in 0..x_scaled.rows {
                    for (c, s) in shift.iter().enumerate() {
                        x_scaled.set(r, c, x_scaled.get(r, c) - s + 1e-6);
                    }
                }
            }
        }

        let mut states = Vec::with_capacity(coordinates.len());
        for r in 0..x_scaled.rows {
            let (idx, _) = tree.query_nearest(x_scaled.row(r)).unwrap();
            states.push(labels[idx]);
        }
        Ok(states)
    }

    fn save_artifacts(&self) -> Result<PartitionerArtifacts> {
        let (sd, sl, cl, m, s) = match (
            &self.support_data,
            &self.support_labels,
            &self.col_labels,
            &self.mean,
            &self.std,
        ) {
            (Some(sd), Some(sl), Some(cl), Some(m), Some(s)) => (sd, sl, cl, m, s),
            _ => {
                return Err(DemMcmError::NotFitted(
                    "SpectralBiclusteringPartitioner must be fitted before save()".to_string(),
                ))
            }
        };
        let mut art = PartitionerArtifacts::default();
        art.arrays
            .insert("support_data".to_string(), NpyArray::from_matrix(sd));
        art.arrays.insert(
            "support_labels".to_string(),
            NpyArray::from_i64_vec(sl.clone()),
        );
        art.arrays
            .insert("col_labels".to_string(), NpyArray::from_i64_vec(cl.clone()));
        art.arrays
            .insert("mean".to_string(), NpyArray::from_f64_vec(m.clone()));
        art.arrays
            .insert("std".to_string(), NpyArray::from_f64_vec(s.clone()));
        if let Some(ls) = &self.log_shift {
            art.arrays
                .insert("log_shift".to_string(), NpyArray::from_f64_vec(ls.clone()));
        }
        let mut params = JsonValue::object();
        params.insert("n_features", JsonValue::Number(self.n_features as f64));
        params.insert("n_cells", JsonValue::Number(self.n_cells_cfg as f64));
        params.insert(
            "n_col_clusters",
            JsonValue::Number(self.n_col_clusters as f64),
        );
        params.insert("method", JsonValue::String(self.method_name.clone()));
        params.insert("velocity_weight", JsonValue::Number(self.velocity_weight));
        params.insert("n_components", JsonValue::Number(self.n_components as f64));
        params.insert("n_best", JsonValue::Number(self.n_best as f64));
        art.json_files.insert("params".to_string(), params);
        Ok(art)
    }

    fn load(&mut self, dir: &Path) -> Result<()> {
        let sd = NpyArray::load(&dir.join("support_data.npy"))?.to_matrix()?;
        let sl = NpyArray::load(&dir.join("support_labels.npy"))?.to_i64_vec()?;
        let cl = NpyArray::load(&dir.join("col_labels.npy"))?.to_i64_vec()?;
        let m = NpyArray::load(&dir.join("mean.npy"))?.to_f64_vec();
        let s = NpyArray::load(&dir.join("std.npy"))?.to_f64_vec();
        let ls_path = dir.join("log_shift.npy");
        if ls_path.exists() {
            self.log_shift = Some(NpyArray::load(&ls_path)?.to_f64_vec());
        }
        self.tree = Some(KdTree::build(&sd.data, sd.cols));
        self.support_data = Some(sd);
        self.support_labels = Some(sl);
        self.col_labels = Some(cl);
        self.mean = Some(m);
        self.std = Some(s);
        let p = JsonValue::parse(&fs::read_to_string(dir.join("params.json"))?)?;
        self.n_features = p.get("n_features").and_then(|v| v.as_usize()).unwrap_or(7);
        self.n_cells_cfg = p.get("n_cells").and_then(|v| v.as_usize()).unwrap_or(30);
        self.n_col_clusters = p
            .get("n_col_clusters")
            .and_then(|v| v.as_usize())
            .unwrap_or(3);
        if let Some(mname) = p.get("method").and_then(|v| v.as_str()) {
            self.method_name = mname.to_string();
        }
        Ok(())
    }
}
