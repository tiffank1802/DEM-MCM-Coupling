//! Physics-aware spatial partitioners using DEM positions and velocities:
//! - [`PhysicsAwarePartitioner`]
//! - [`FullVectorVelocityKMeansPartitioner`]

use std::fs;
use std::path::Path;

use crate::config::PartitioningMethod;
use crate::error::{DemMcmError, Result};
use crate::io::json::JsonValue;
use crate::io::npy::NpyArray;
use crate::linalg::Matrix;
use crate::ml::{KMeans, MiniBatchKMeans};
use crate::partitioners::{Partitioner, PartitionerArtifacts, PartitionerGeometry};
use crate::rng::Rng;
use crate::spatial::KdTree;

pub(crate) fn scale_and_weight_features(
    features: &Matrix,
    n_pos: usize,
    velocity_weight: f64,
) -> (Matrix, Vec<f64>, Vec<f64>) {
    let mean = features.col_means();
    let mut std = features.col_stds();
    for s in &mut std {
        if *s == 0.0 {
            *s = 1.0;
        }
    }
    let mut scaled = Matrix::zeros(features.rows, features.cols);
    for r in 0..features.rows {
        for c in 0..features.cols {
            let mut val = (features.get(r, c) - mean[c]) / std[c];
            if c >= n_pos {
                val *= velocity_weight;
            }
            scaled.set(r, c, val);
        }
    }
    (scaled, mean, std)
}

pub(crate) fn apply_saved_scaling(
    features: &Matrix,
    mean: &[f64],
    std: &[f64],
    n_pos: usize,
    velocity_weight: f64,
) -> Matrix {
    let mut scaled = Matrix::zeros(features.rows, features.cols);
    for r in 0..features.rows {
        for c in 0..features.cols {
            let mut val = (features.get(r, c) - mean[c]) / std[c];
            if c >= n_pos {
                val *= velocity_weight;
            }
            scaled.set(r, c, val);
        }
    }
    scaled
}

pub(crate) fn compute_relative_velocity(
    coords: &[[f64; 3]],
    vel: &[[f64; 3]],
    omega: f64,
) -> Vec<[f64; 3]> {
    coords
        .iter()
        .zip(vel.iter())
        .map(|(c, v)| [v[0] - (-omega * c[1]), v[1] - (omega * c[0]), v[2]])
        .collect()
}

// ============================================================================
// 6. PHYSICS-AWARE PARTITIONER
// ============================================================================

#[derive(Debug, Clone)]
pub struct PhysicsAwarePartitioner {
    pub n_cells_cfg: usize,
    pub velocity_weight: f64,
    pub random_state: u64,
    pub velocity_mode: String,
    pub use_velocity: bool,
    pub dem_velocities: Option<Vec<[f64; 3]>>,

    pub centroids: Option<Matrix>,
    pub mean: Option<Vec<f64>>,
    pub std: Option<Vec<f64>>,
    pub n_features: usize,
    tree: Option<KdTree>,
}

impl PhysicsAwarePartitioner {
    pub const OMEGA: f64 = 4.0;

    pub fn new(
        n_cells: usize,
        velocity_weight: f64,
        random_state: u64,
        velocity_mode: impl Into<String>,
    ) -> Self {
        Self {
            n_cells_cfg: n_cells,
            velocity_weight,
            random_state,
            velocity_mode: velocity_mode.into(),
            use_velocity: false,
            dem_velocities: None,
            centroids: None,
            mean: None,
            std: None,
            n_features: 3,
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
                "PhysicsAwarePartitioner::fit requires non-empty coordinates".to_string(),
            ));
        }
        let use_vel = use_velocities.unwrap_or(self.use_velocity);

        let features = if use_vel {
            if let Some(vel) = &self.dem_velocities {
                if vel.len() == coordinates.len() {
                    let v_rel = compute_relative_velocity(coordinates, vel, Self::OMEGA);
                    match self.velocity_mode.as_str() {
                        "norm" => {
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
                        }
                        "components" => {
                            let mut m = Matrix::zeros(coordinates.len(), 6);
                            for i in 0..coordinates.len() {
                                m.set(i, 0, coordinates[i][0]);
                                m.set(i, 1, coordinates[i][1]);
                                m.set(i, 2, coordinates[i][2]);
                                m.set(i, 3, v_rel[i][0]);
                                m.set(i, 4, v_rel[i][1]);
                                m.set(i, 5, v_rel[i][2]);
                            }
                            m
                        }
                        other => {
                            return Err(DemMcmError::InvalidArgument(format!(
                                "Unknown velocity_mode: {other:?}. Choose 'norm' or 'components'."
                            )))
                        }
                    }
                } else {
                    Self::coords_matrix(coordinates)
                }
            } else {
                Self::coords_matrix(coordinates)
            }
        } else {
            Self::coords_matrix(coordinates)
        };

        self.fit_internal(&features, 3)
    }

    fn coords_matrix(coordinates: &[[f64; 3]]) -> Matrix {
        let mut m = Matrix::zeros(coordinates.len(), 3);
        for (i, c) in coordinates.iter().enumerate() {
            m.set(i, 0, c[0]);
            m.set(i, 1, c[1]);
            m.set(i, 2, c[2]);
        }
        m
    }

    fn fit_internal(&mut self, features: &Matrix, n_pos: usize) -> Result<()> {
        self.n_features = features.cols;
        let (x_scaled, mean, std) =
            scale_and_weight_features(features, n_pos, self.velocity_weight);
        self.mean = Some(mean);
        self.std = Some(std);

        let mut rng = Rng::seed_from_u64(self.random_state);
        let x_fit = if x_scaled.rows > 500_000 {
            let idx = rng.sample_indices(x_scaled.rows, 500_000);
            x_scaled.select_rows(&idx)
        } else {
            x_scaled
        };

        let k = self.n_cells_cfg.min(x_fit.rows).max(1);
        let km = KMeans::new(k)
            .with_random_state(self.random_state)
            .fit(&x_fit)?;
        let centers = km.centroids;
        self.tree = Some(KdTree::build(&centers.data, centers.cols));
        self.centroids = Some(centers);
        Ok(())
    }
}

impl Partitioner for PhysicsAwarePartitioner {
    fn method(&self) -> PartitioningMethod {
        PartitioningMethod::Physics
    }

    fn type_name(&self) -> &'static str {
        "PhysicsAwarePartitioner"
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
        let suffix = if !self.use_velocity || self.n_features == 3 {
            "pos"
        } else if self.velocity_mode == "norm" {
            "normvel"
        } else {
            "compvel"
        };
        format!(
            "physics_{}cells_{suffix}_vw{}",
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
        let (tree, mean, std) = match (&self.tree, &self.mean, &self.std) {
            (Some(t), Some(m), Some(s)) => (t, m, s),
            _ => {
                return Err(DemMcmError::NotFitted(
                    "PhysicsAwarePartitioner must be fitted before use".to_string(),
                ))
            }
        };

        let mut features = Matrix::zeros(coordinates.len(), self.n_features);
        let v_rel_opt = velocities
            .filter(|v| v.len() == coordinates.len())
            .map(|v| compute_relative_velocity(coordinates, v, Self::OMEGA));

        for (i, c) in coordinates.iter().enumerate() {
            features.set(i, 0, c[0]);
            features.set(i, 1, c[1]);
            features.set(i, 2, c[2]);
            match self.n_features {
                4 => {
                    let spd = v_rel_opt.as_ref().map_or(0.0, |vr| {
                        (vr[i][0] * vr[i][0] + vr[i][1] * vr[i][1] + vr[i][2] * vr[i][2]).sqrt()
                    });
                    features.set(i, 3, spd);
                }
                6 => {
                    let vr = v_rel_opt.as_ref().map_or([0.0; 3], |v| v[i]);
                    features.set(i, 3, vr[0]);
                    features.set(i, 4, vr[1]);
                    features.set(i, 5, vr[2]);
                }
                _ => {}
            }
        }

        let x_scaled = apply_saved_scaling(&features, mean, std, 3, self.velocity_weight);
        let mut states = Vec::with_capacity(coordinates.len());
        for r in 0..x_scaled.rows {
            let (idx, _) = tree.query_nearest(x_scaled.row(r)).unwrap();
            states.push(idx as i64);
        }
        Ok(states)
    }

    fn save_artifacts(&self) -> Result<PartitionerArtifacts> {
        let (c, m, s) = match (&self.centroids, &self.mean, &self.std) {
            (Some(c), Some(m), Some(s)) => (c, m, s),
            _ => {
                return Err(DemMcmError::NotFitted(
                    "PhysicsAwarePartitioner must be fitted before save()".to_string(),
                ))
            }
        };
        let mut art = PartitionerArtifacts::default();
        art.arrays
            .insert("centroids".to_string(), NpyArray::from_matrix(c));
        art.arrays
            .insert("mean".to_string(), NpyArray::from_f64_vec(m.clone()));
        art.arrays
            .insert("std".to_string(), NpyArray::from_f64_vec(s.clone()));
        let mut params = JsonValue::object();
        params.insert("n_features", JsonValue::Number(self.n_features as f64));
        params.insert(
            "velocity_mode",
            JsonValue::String(self.velocity_mode.clone()),
        );
        art.json_files.insert("physics_params".to_string(), params);
        Ok(art)
    }

    fn geometry_hint(&self) -> PartitionerGeometry {
        if let (Some(c), Some(m), Some(s)) = (&self.centroids, &self.mean, &self.std) {
            if c.cols >= 3 && m.len() >= 3 && s.len() >= 3 {
                let seeds = (0..c.rows)
                    .map(|r| {
                        [
                            c.get(r, 0) * s[0] + m[0],
                            c.get(r, 1) * s[1] + m[1],
                            c.get(r, 2) * s[2] + m[2],
                        ]
                    })
                    .collect();
                return PartitionerGeometry::VoronoiSeeds(seeds);
            }
        }
        PartitionerGeometry::PointClusters
    }

    fn load(&mut self, dir: &Path) -> Result<()> {
        let c = NpyArray::load(&dir.join("centroids.npy"))?.to_matrix()?;
        let m = NpyArray::load(&dir.join("mean.npy"))?.to_f64_vec();
        let s = NpyArray::load(&dir.join("std.npy"))?.to_f64_vec();
        self.tree = Some(KdTree::build(&c.data, c.cols));
        self.n_cells_cfg = c.rows;
        self.centroids = Some(c);
        self.mean = Some(m);
        self.std = Some(s);
        let p = JsonValue::parse(&fs::read_to_string(dir.join("physics_params.json"))?)?;
        self.n_features = p.get("n_features").and_then(|v| v.as_usize()).unwrap_or(3);
        if let Some(vm) = p.get("velocity_mode").and_then(|v| v.as_str()) {
            self.velocity_mode = vm.to_string();
        }
        Ok(())
    }
}

// ============================================================================
// 7. FULL-VECTOR VELOCITY K-MEANS PARTITIONER
// ============================================================================

#[derive(Debug, Clone)]
pub struct FullVectorVelocityKMeansPartitioner {
    pub n_cells_cfg: usize,
    pub velocity_weight: f64,
    pub random_state: u64,
    pub use_velocity: bool,
    pub dem_velocities: Option<Vec<[f64; 3]>>,

    pub centroids: Option<Matrix>,
    pub mean: Option<Vec<f64>>,
    pub std: Option<Vec<f64>>,
    pub n_features: usize,
    tree: Option<KdTree>,
}

impl FullVectorVelocityKMeansPartitioner {
    pub fn new(n_cells: usize, velocity_weight: f64, random_state: u64) -> Self {
        Self {
            n_cells_cfg: n_cells,
            velocity_weight,
            random_state,
            use_velocity: true,
            dem_velocities: None,
            centroids: None,
            mean: None,
            std: None,
            n_features: 6,
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
                "FullVectorVelocityKMeansPartitioner::fit requires non-empty coordinates"
                    .to_string(),
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
        let x_fit = if x_scaled.rows > 500_000 {
            let idx = rng.sample_indices(x_scaled.rows, 500_000);
            x_scaled.select_rows(&idx)
        } else {
            x_scaled
        };

        let k = self.n_cells_cfg.min(x_fit.rows).max(1);
        let mb = MiniBatchKMeans::new(k, 10_000.min(x_fit.rows), self.random_state);
        let model = mb.fit(&x_fit)?;
        let centers = model.centroids;
        self.tree = Some(KdTree::build(&centers.data, centers.cols));
        self.centroids = Some(centers);
        Ok(())
    }
}

impl Partitioner for FullVectorVelocityKMeansPartitioner {
    fn method(&self) -> PartitioningMethod {
        PartitioningMethod::PhysicsFullVel
    }

    fn type_name(&self) -> &'static str {
        "FullVectorVelocityKMeansPartitioner"
    }

    fn splitting_method(&self) -> String {
        "fullvel_kmeans".to_string()
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
            "fullvel_kmeans_{}cells_vw{}",
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
        let (tree, mean, std) = match (&self.tree, &self.mean, &self.std) {
            (Some(t), Some(m), Some(s)) => (t, m, s),
            _ => {
                return Err(DemMcmError::NotFitted(
                    "FullVectorVelocityKMeansPartitioner must be fitted before use".to_string(),
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
            states.push(idx as i64);
        }
        Ok(states)
    }

    fn save_artifacts(&self) -> Result<PartitionerArtifacts> {
        let (c, m, s) = match (&self.centroids, &self.mean, &self.std) {
            (Some(c), Some(m), Some(s)) => (c, m, s),
            _ => {
                return Err(DemMcmError::NotFitted(
                    "FullVectorVelocityKMeansPartitioner must be fitted before save()".to_string(),
                ))
            }
        };
        let mut art = PartitionerArtifacts::default();
        art.arrays
            .insert("centroids".to_string(), NpyArray::from_matrix(c));
        art.arrays
            .insert("mean".to_string(), NpyArray::from_f64_vec(m.clone()));
        art.arrays
            .insert("std".to_string(), NpyArray::from_f64_vec(s.clone()));
        let mut params = JsonValue::object();
        params.insert("n_features", JsonValue::Number(self.n_features as f64));
        params.insert("n_cells", JsonValue::Number(self.n_cells_cfg as f64));
        params.insert("vw", JsonValue::Number(self.velocity_weight));
        art.json_files.insert("params".to_string(), params);
        Ok(art)
    }

    fn geometry_hint(&self) -> PartitionerGeometry {
        if let (Some(c), Some(m), Some(s)) = (&self.centroids, &self.mean, &self.std) {
            if c.cols >= 3 && m.len() >= 3 && s.len() >= 3 {
                let seeds = (0..c.rows)
                    .map(|r| {
                        [
                            c.get(r, 0) * s[0] + m[0],
                            c.get(r, 1) * s[1] + m[1],
                            c.get(r, 2) * s[2] + m[2],
                        ]
                    })
                    .collect();
                return PartitionerGeometry::VoronoiSeeds(seeds);
            }
        }
        PartitionerGeometry::PointClusters
    }

    fn load(&mut self, dir: &Path) -> Result<()> {
        let c = NpyArray::load(&dir.join("centroids.npy"))?.to_matrix()?;
        let m = NpyArray::load(&dir.join("mean.npy"))?.to_f64_vec();
        let s = NpyArray::load(&dir.join("std.npy"))?.to_f64_vec();
        self.tree = Some(KdTree::build(&c.data, c.cols));
        self.centroids = Some(c);
        self.mean = Some(m);
        self.std = Some(s);
        let p = JsonValue::parse(&fs::read_to_string(dir.join("params.json"))?)?;
        self.n_features = p.get("n_features").and_then(|v| v.as_usize()).unwrap_or(6);
        Ok(())
    }
}
