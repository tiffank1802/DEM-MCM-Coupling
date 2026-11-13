//! Spatial partitioning methods for Markov chain models.
//!
//! Mirrors `dem_mcm_coupling/partitioners.py` and exposes all 14 partitioning
//! strategies behind a unified [`Partitioner`] trait and [`create_partitioner`]
//! factory registry:
//!
//! - `cartesian` — Regular `(x, y, z)` grid
//! - `cylindrical` — Cylindrical `(r, theta, z)` grid (`equal_area` / `equal_dr`)
//! - `voronoi` — K-means Voronoi tessellation (reference MCM)
//! - `quantile` — Equi-population quantile grid
//! - `octree` — Adaptive axial 8-ary octree or binary oblique tree (`pca`, `kmeans2`, `2medians`, `random`, `svm`)
//! - `physics` — K-means on `(x, y, z)` + rotating-frame relative velocity (`norm` or `components`)
//! - `physics_full_vel` — MiniBatchKMeans on `(x, y, z, vx, vy, vz)`
//! - `spectral` — Spectral clustering on k-NN/RBF graph + 1-NN support extension
//! - `dbscan` — Density-based clustering (`-1` noise support)
//! - `gmm` — Full-covariance Gaussian mixture model
//! - `spectral_biclustering` — Position x velocity spectral biclustering
//! - `adaptive` — Coarse-top / fine-bottom adaptive split along `y`
//! - `multizone` — Multi-zone hybrid partitioner along `y`
//! - `single` — Degenerate 1-cell baseline

pub mod clustering;
pub mod geometric;
pub mod hybrid;
pub mod physics;

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::config::{validate_partitioning_method, PartitioningMethod};
use crate::error::Result;
use crate::io::json::JsonValue;
use crate::io::npy::NpyArray;

pub use clustering::{
    DbscanPartitioner, GaussianMixturePartitioner, SpectralBiclusteringPartitioner,
    SpectralClusteringPartitioner,
};
pub use geometric::{
    CartesianPartitioner, CylindricalPartitioner, OctreePartitioner, QuantileGridPartitioner,
    VoronoiPartitioner,
};
pub use hybrid::{AdaptivePartitioner, MultiZonePartitioner, SingleCellPartitioner, ZoneSpec};
pub use physics::{FullVectorVelocityKMeansPartitioner, PhysicsAwarePartitioner};

/// Per-cell population diagnostics returned by [`Partitioner::diagnostics`].
#[derive(Debug, Clone, PartialEq)]
pub struct PartitionerDiagnostics {
    pub pop_min: usize,
    pub pop_max: usize,
    pub pop_mean: f64,
    pub pop_std: f64,
    pub n_empty: usize,
    pub n_visited: usize,
    pub fraction_visited: f64,
}

/// Oblique leaf descriptor in original 3-D DEM coordinates for mesh visualization.
#[derive(Debug, Clone)]
pub struct ObliqueCellGeometry {
    pub bounds: [f64; 6],
    pub centroid: [f64; 3],
    pub halfspaces: Vec<([f64; 3], f64, bool)>,
}

/// Structural 3-D geometry descriptor exported by a fitted [`Partitioner`].
#[derive(Debug, Clone)]
pub enum PartitionerGeometry {
    CartesianGrid {
        nx: usize,
        ny: usize,
        nz: usize,
        bounds: [f64; 6],
    },
    CylindricalGrid {
        x_center: f64,
        y_center: f64,
        r_edges: Vec<f64>,
        theta_edges: Vec<f64>,
        z_edges: Vec<f64>,
    },
    TensorGrid {
        x_edges: Vec<f64>,
        y_edges: Vec<f64>,
        z_edges: Vec<f64>,
    },
    AxisBoxes(Vec<[f64; 6]>),
    ObliquePolyhedra {
        domain_bounds: [f64; 6],
        cells: Vec<ObliqueCellGeometry>,
    },
    VoronoiSeeds(Vec<[f64; 3]>),
    AdaptiveZones {
        y_split: f64,
        y_min: f64,
        y_max: f64,
        n_bottom: usize,
        bottom: Box<PartitionerGeometry>,
        top: Option<Box<PartitionerGeometry>>,
    },
    MultiZone {
        zones: Vec<(f64, f64, usize, PartitionerGeometry)>,
    },
    SingleDomain,
    PointClusters,
}

/// Serialized artifacts for a fitted partitioner (stored under `partitioner/`).
#[derive(Debug, Default, Clone)]
pub struct PartitionerArtifacts {
    pub arrays: HashMap<String, NpyArray>,
    pub json_files: HashMap<String, JsonValue>,
    pub text_files: HashMap<String, String>,
}

/// Common interface implemented by all 14 spatial partitioners.
pub trait Partitioner: Send + Sync {
    /// Partitioning method enum variant.
    fn method(&self) -> PartitioningMethod;

    /// Python class name (stored in `partitioner_meta.json["type"]`).
    fn type_name(&self) -> &'static str;

    /// Total number of partition cells (Markov states).
    fn n_cells(&self) -> usize;

    /// Unique identifier string used for experiment folder naming.
    fn label(&self) -> String;

    /// Short splitting method tag (mirrors Python `splitting_method`).
    fn splitting_method(&self) -> String {
        self.method().as_str().to_string()
    }

    /// Whether this partitioner makes use of velocity features.
    fn uses_velocity(&self) -> bool {
        false
    }

    /// Supply DEM velocities `(N, 3)` before calling [`Partitioner::fit`].
    fn set_dem_velocities(&mut self, _velocities: Option<&[[f64; 3]]>) {}

    /// Toggle optional velocity feature usage before calling [`Partitioner::fit`].
    fn set_use_velocity(&mut self, _use_vel: bool) {}

    /// Learn cell boundaries from 3-D particle coordinates.
    fn fit(&mut self, coordinates: &[[f64; 3]]) -> Result<()>;

    /// Assign each particle to its cell state index (in `0..n_cells`, or `-1`
    /// when outside the active zone / noise).
    fn compute_states(
        &self,
        coordinates: &[[f64; 3]],
        velocities: Option<&[[f64; 3]]>,
    ) -> Result<Vec<i64>>;

    /// Compute per-cell population statistics for the supplied particles.
    fn diagnostics(
        &self,
        coordinates: &[[f64; 3]],
        velocities: Option<&[[f64; 3]]>,
    ) -> Result<PartitionerDiagnostics> {
        let states = self.compute_states(coordinates, velocities)?;
        let nc = self.n_cells();
        let mut counts = vec![0usize; nc];
        for &s in &states {
            if s >= 0 && (s as usize) < nc {
                counts[s as usize] += 1;
            }
        }
        let n_visited = counts.iter().filter(|&&c| c > 0).count();
        let n_empty = nc.saturating_sub(n_visited);
        let pop_min = counts.iter().copied().filter(|&c| c > 0).min().unwrap_or(0);
        let pop_max = counts.iter().copied().max().unwrap_or(0);
        let pop_mean = if nc > 0 {
            counts.iter().sum::<usize>() as f64 / nc as f64
        } else {
            0.0
        };
        let pop_std = if nc > 0 {
            let var = counts
                .iter()
                .map(|&c| {
                    let d = c as f64 - pop_mean;
                    d * d
                })
                .sum::<f64>()
                / nc as f64;
            var.sqrt()
        } else {
            0.0
        };
        let fraction_visited = if nc > 0 {
            n_visited as f64 / nc as f64
        } else {
            0.0
        };
        Ok(PartitionerDiagnostics {
            pop_min,
            pop_max,
            pop_mean,
            pop_std,
            n_empty,
            n_visited,
            fraction_visited,
        })
    }

    /// Structural 3-D geometric description of cells (used for 3-D mesh visualization).
    fn geometry_hint(&self) -> PartitionerGeometry {
        PartitionerGeometry::PointClusters
    }

    /// Collect in-memory artifacts for saving into a `DataSource`.
    fn save_artifacts(&self) -> Result<PartitionerArtifacts> {
        Ok(PartitionerArtifacts::default())
    }

    /// Persist the fitted partitioner state to a directory (`partitioner_meta.json` + artifacts).
    fn save(&self, dir: &Path) -> Result<()> {
        fs::create_dir_all(dir)?;
        let mut meta = JsonValue::object();
        meta.insert("type", JsonValue::String(self.type_name().to_string()));
        meta.insert("label", JsonValue::String(self.label()));
        meta.insert("n_cells", JsonValue::Number(self.n_cells() as f64));
        fs::write(dir.join("partitioner_meta.json"), meta.to_string_pretty())?;

        let artifacts = self.save_artifacts()?;
        for (name, arr) in artifacts.arrays {
            let fname = if name.ends_with(".npy") {
                name
            } else {
                format!("{name}.npy")
            };
            arr.save(&dir.join(fname))?;
        }
        for (name, val) in artifacts.json_files {
            let fname = if name.ends_with(".json") {
                name
            } else {
                format!("{name}.json")
            };
            fs::write(dir.join(fname), val.to_string_pretty())?;
        }
        for (name, text) in artifacts.text_files {
            fs::write(dir.join(name), text)?;
        }
        Ok(())
    }

    /// Restore a previously saved partitioner from a directory.
    fn load(&mut self, dir: &Path) -> Result<()>;
}

/// Instantiate a [`Partitioner`] from a method name and JSON keyword map.
pub fn create_partitioner(
    method: &str,
    kwargs: &HashMap<String, JsonValue>,
) -> Result<Box<dyn Partitioner>> {
    let m = validate_partitioning_method(method)?;
    create_partitioner_by_method(m, kwargs)
}

/// Instantiate a [`Partitioner`] from a [`PartitioningMethod`] and JSON keyword map.
pub fn create_partitioner_by_method(
    method: PartitioningMethod,
    kwargs: &HashMap<String, JsonValue>,
) -> Result<Box<dyn Partitioner>> {
    let get_usize = |k: &str, def: usize| kwargs.get(k).and_then(|v| v.as_usize()).unwrap_or(def);
    let get_u64 = |k: &str, def: u64| {
        kwargs
            .get(k)
            .and_then(|v| v.as_i64())
            .map(|x| x as u64)
            .unwrap_or(def)
    };
    let get_f64 = |k: &str, def: f64| kwargs.get(k).and_then(|v| v.as_f64()).unwrap_or(def);
    let get_opt_f64 = |k: &str| kwargs.get(k).and_then(|v| v.as_f64());
    let get_str = |k: &str| kwargs.get(k).and_then(|v| v.as_str()).map(str::to_string);

    match method {
        PartitioningMethod::Cartesian => Ok(Box::new(CartesianPartitioner::new(
            get_usize("nx", 5),
            get_usize("ny", 5),
            get_usize("nz", 5),
        ))),
        PartitioningMethod::Cylindrical => Ok(Box::new(CylindricalPartitioner {
            nr: get_usize("nr", 5),
            ntheta: get_usize("ntheta", 8),
            nz: get_usize("nz", 5),
            radial_mode: get_str("radial_mode").unwrap_or_else(|| "equal_area".to_string()),
            theta_min_input: get_opt_f64("theta_min"),
            theta_max_input: get_opt_f64("theta_max"),
            z_min_limit_input: get_opt_f64("z_min_limit"),
            z_max_limit_input: get_opt_f64("z_max_limit"),
            r_min_limit_input: get_opt_f64("r_min_limit"),
            r_max_limit_input: get_opt_f64("r_max_limit"),
            ..CylindricalPartitioner::default()
        })),
        PartitioningMethod::Voronoi => Ok(Box::new(VoronoiPartitioner::new(
            get_usize("n_cells", 125),
            get_u64("random_state", 42),
        ))),
        PartitioningMethod::Quantile => Ok(Box::new(QuantileGridPartitioner::new(
            get_usize("nx", 5),
            get_usize("ny", 5),
            get_usize("nz", 5),
        ))),
        PartitioningMethod::Octree => Ok(Box::new(OctreePartitioner::new(
            get_usize("max_particles", 100),
            get_usize("max_depth", 5),
            get_str("transform_type"),
            get_str("oblique_method"),
        ))),
        PartitioningMethod::Physics => {
            let mut p = PhysicsAwarePartitioner::new(
                get_usize("n_cells", 125),
                get_f64("velocity_weight", 0.5),
                get_u64("random_state", 42),
                get_str("velocity_mode").unwrap_or_else(|| "norm".to_string()),
            );
            if let Some(uv) = kwargs.get("use_velocity").and_then(|v| v.as_bool()) {
                p.use_velocity = uv;
            }
            Ok(Box::new(p))
        }
        PartitioningMethod::PhysicsFullVel => {
            Ok(Box::new(FullVectorVelocityKMeansPartitioner::new(
                get_usize("n_cells", 125),
                get_f64("velocity_weight", 0.5),
                get_u64("random_state", 42),
            )))
        }
        PartitioningMethod::Spectral => Ok(Box::new(SpectralClusteringPartitioner::new(
            get_usize("n_cells", 125),
            get_f64("velocity_weight", 1.0),
            get_usize("n_neighbors", 15),
            get_usize("max_samples", 5000),
            get_u64("random_state", 42),
        ))),
        PartitioningMethod::Dbscan => Ok(Box::new(DbscanPartitioner::new(
            get_f64("eps", 0.1),
            get_usize("min_samples", 10),
            get_f64("velocity_weight", 0.5),
            get_usize("max_samples", 5000),
            get_u64("random_state", 42),
        ))),
        PartitioningMethod::Gmm => Ok(Box::new(GaussianMixturePartitioner::new(
            get_usize("n_cells", 125),
            get_f64("velocity_weight", 0.5),
            get_u64("random_state", 42),
            get_usize("max_fit_samples", 50_000),
        ))),
        PartitioningMethod::SpectralBiclustering => {
            Ok(Box::new(SpectralBiclusteringPartitioner::new(
                get_usize("n_cells", 30),
                get_usize("n_col_clusters", 3),
                get_str("method").unwrap_or_else(|| "log".to_string()),
                get_f64("velocity_weight", 0.6),
                get_usize("n_components", 10),
                get_usize("n_best", 5),
                get_usize("max_samples", 4000),
                get_u64("random_state", 42),
            )))
        }
        PartitioningMethod::Adaptive => {
            let top_kwargs = kwargs
                .get("top_kwargs")
                .and_then(|v| v.as_object())
                .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                .unwrap_or_default();
            let bottom_kwargs = kwargs
                .get("bottom_kwargs")
                .and_then(|v| v.as_object())
                .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                .unwrap_or_default();
            Ok(Box::new(AdaptivePartitioner::new(
                get_opt_f64("y_split"),
                get_str("y_split_mode").unwrap_or_else(|| "quantile".to_string()),
                get_str("top_method").unwrap_or_else(|| "single".to_string()),
                top_kwargs,
                get_str("bottom_method").unwrap_or_else(|| "cylindrical".to_string()),
                bottom_kwargs,
            )))
        }
        PartitioningMethod::Multizone => {
            let zones = kwargs
                .get("zones")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(ZoneSpec::from_json).collect())
                .unwrap_or_default();
            let y_mode = get_str("y_mode").unwrap_or_else(|| "absolute".to_string());
            Ok(Box::new(MultiZonePartitioner::new(zones, y_mode)))
        }
        PartitioningMethod::Single => Ok(Box::new(SingleCellPartitioner::new())),
    }
}
