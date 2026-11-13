//! Geometric spatial partitioners:
//! - [`CartesianPartitioner`]
//! - [`CylindricalPartitioner`]
//! - [`VoronoiPartitioner`]
//! - [`QuantileGridPartitioner`]
//! - [`OctreePartitioner`]

use std::collections::HashMap;
use std::f64::consts::PI;
use std::fs;
use std::path::Path;

use crate::config::PartitioningMethod;
use crate::error::{DemMcmError, Result};
use crate::io::json::JsonValue;
use crate::io::npy::NpyArray;
use crate::linalg::{covariance_matrix, jacobi_eigh, Matrix};
use crate::ml::{KMeans, LinearSvc};
use crate::partitioners::{
    ObliqueCellGeometry, Partitioner, PartitionerArtifacts, PartitionerGeometry,
};
use crate::rng::Rng;
use crate::spatial::KdTree;

// ============================================================================
// HELPERS
// ============================================================================

#[inline]
fn median_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

pub(crate) fn median(vals: &[f64]) -> f64 {
    let mut s = vals.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    median_sorted(&s)
}

/// Linear interpolation quantile matching `numpy.quantile(..., method="linear")`.
pub(crate) fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let q = q.clamp(0.0, 1.0);
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

pub(crate) fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![start; n];
    }
    (0..n)
        .map(|i| start + (stop - start) * (i as f64) / ((n - 1) as f64))
        .collect()
}

/// Equivalent to `np.searchsorted(edges, val, side="right")`.
#[inline]
fn searchsorted_right(edges: &[f64], val: f64) -> usize {
    edges.partition_point(|&e| e <= val)
}

// ============================================================================
// 1. CARTESIAN PARTITIONER
// ============================================================================

#[derive(Debug, Clone)]
pub struct CartesianPartitioner {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub bounds: Option<[f64; 6]>, // xmin, xmax, ymin, ymax, zmin, zmax
}

impl Default for CartesianPartitioner {
    fn default() -> Self {
        Self::new(5, 5, 5)
    }
}

impl CartesianPartitioner {
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            nx,
            ny,
            nz,
            bounds: None,
        }
    }
}

impl Partitioner for CartesianPartitioner {
    fn method(&self) -> PartitioningMethod {
        PartitioningMethod::Cartesian
    }

    fn type_name(&self) -> &'static str {
        "CartesianPartitioner"
    }

    fn n_cells(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    fn label(&self) -> String {
        format!("cartesian_nx{}_ny{}_nz{}", self.nx, self.ny, self.nz)
    }

    fn fit(&mut self, coordinates: &[[f64; 3]]) -> Result<()> {
        if coordinates.is_empty() {
            return Err(DemMcmError::InvalidArgument(
                "CartesianPartitioner::fit requires non-empty coordinates".to_string(),
            ));
        }
        let eps = 0.001;
        let mut mins = [f64::INFINITY; 3];
        let mut maxs = [f64::NEG_INFINITY; 3];
        for c in coordinates {
            for d in 0..3 {
                mins[d] = mins[d].min(c[d]);
                maxs[d] = maxs[d].max(c[d]);
            }
        }
        self.bounds = Some([
            mins[0] - eps,
            maxs[0] + eps,
            mins[1] - eps,
            maxs[1] + eps,
            mins[2] - eps,
            maxs[2] + eps,
        ]);
        Ok(())
    }

    fn compute_states(
        &self,
        coordinates: &[[f64; 3]],
        _velocities: Option<&[[f64; 3]]>,
    ) -> Result<Vec<i64>> {
        let [xmin, xmax, ymin, ymax, zmin, zmax] = self.bounds.ok_or_else(|| {
            DemMcmError::NotFitted("CartesianPartitioner must be fitted before use".to_string())
        })?;
        let nx_f = self.nx as f64;
        let ny_f = self.ny as f64;
        let nz_f = self.nz as f64;
        let nx_max = (self.nx - 1) as f64;
        let ny_max = (self.ny - 1) as f64;
        let nz_max = (self.nz - 1) as f64;

        let mut states = Vec::with_capacity(coordinates.len());
        for c in coordinates {
            let ix = ((c[0] - xmin) * nx_f / (xmax - xmin))
                .clamp(0.0, nx_max)
                .round_ties_even() as i64;
            let iy = ((c[1] - ymin) * ny_f / (ymax - ymin))
                .clamp(0.0, ny_max)
                .round_ties_even() as i64;
            let iz = ((c[2] - zmin) * nz_f / (zmax - zmin))
                .clamp(0.0, nz_max)
                .round_ties_even() as i64;
            states.push(ix + iy * (self.nx as i64) + iz * ((self.nx * self.ny) as i64));
        }
        Ok(states)
    }

    fn save_artifacts(&self) -> Result<PartitionerArtifacts> {
        let b = self.bounds.ok_or_else(|| {
            DemMcmError::NotFitted("CartesianPartitioner must be fitted before save()".to_string())
        })?;
        let mut art = PartitionerArtifacts::default();
        art.arrays
            .insert("bounds".to_string(), NpyArray::from_f64_vec(b.to_vec()));
        Ok(art)
    }

    fn geometry_hint(&self) -> PartitionerGeometry {
        if let Some(bounds) = self.bounds {
            PartitionerGeometry::CartesianGrid {
                nx: self.nx,
                ny: self.ny,
                nz: self.nz,
                bounds,
            }
        } else {
            PartitionerGeometry::PointClusters
        }
    }

    fn load(&mut self, dir: &Path) -> Result<()> {
        let b = NpyArray::load(&dir.join("bounds.npy"))?.to_f64_vec();
        if b.len() != 6 {
            return Err(DemMcmError::ShapeMismatch(
                "bounds.npy must have 6 elements".to_string(),
            ));
        }
        self.bounds = Some([b[0], b[1], b[2], b[3], b[4], b[5]]);
        Ok(())
    }
}

// ============================================================================
// 2. CYLINDRICAL PARTITIONER
// ============================================================================

#[derive(Debug, Clone)]
pub struct CylindricalPartitioner {
    pub nr: usize,
    pub ntheta: usize,
    pub nz: usize,
    pub radial_mode: String,
    pub theta_min_input: Option<f64>,
    pub theta_max_input: Option<f64>,
    pub z_min_limit_input: Option<f64>,
    pub z_max_limit_input: Option<f64>,
    pub r_min_limit_input: Option<f64>,
    pub r_max_limit_input: Option<f64>,

    pub theta_min: Option<f64>,
    pub theta_max: Option<f64>,
    pub z_min_limit: Option<f64>,
    pub z_max_limit: Option<f64>,
    pub r_min_limit: Option<f64>,
    pub r_max_limit: Option<f64>,
    pub x_center: Option<f64>,
    pub y_center: Option<f64>,
    pub r_edges: Option<Vec<f64>>,
    pub theta_edges: Option<Vec<f64>>,
    pub z_edges: Option<Vec<f64>>,
}

impl Default for CylindricalPartitioner {
    fn default() -> Self {
        Self::new(5, 8, 5, "equal_area")
    }
}

impl CylindricalPartitioner {
    pub fn new(nr: usize, ntheta: usize, nz: usize, radial_mode: impl Into<String>) -> Self {
        Self {
            nr,
            ntheta,
            nz,
            radial_mode: radial_mode.into(),
            theta_min_input: None,
            theta_max_input: None,
            z_min_limit_input: None,
            z_max_limit_input: None,
            r_min_limit_input: None,
            r_max_limit_input: None,
            theta_min: None,
            theta_max: None,
            z_min_limit: None,
            z_max_limit: None,
            r_min_limit: None,
            r_max_limit: None,
            x_center: None,
            y_center: None,
            r_edges: None,
            theta_edges: None,
            z_edges: None,
        }
    }
}

impl Partitioner for CylindricalPartitioner {
    fn method(&self) -> PartitioningMethod {
        PartitioningMethod::Cylindrical
    }

    fn type_name(&self) -> &'static str {
        "CylindricalPartitioner"
    }

    fn n_cells(&self) -> usize {
        self.nr * self.ntheta * self.nz
    }

    fn label(&self) -> String {
        let tmin = self
            .theta_min
            .map_or_else(|| "auto".to_string(), |v| format!("{v:.2}"));
        let tmax = self
            .theta_max
            .map_or_else(|| "auto".to_string(), |v| format!("{v:.2}"));
        format!(
            "cylindrical_nr{}_nth{}_nz{}_{}_theta{tmin}_{tmax}",
            self.nr, self.ntheta, self.nz, self.radial_mode
        )
    }

    fn fit(&mut self, coordinates: &[[f64; 3]]) -> Result<()> {
        if coordinates.is_empty() {
            return Err(DemMcmError::InvalidArgument(
                "CylindricalPartitioner::fit requires non-empty coordinates".to_string(),
            ));
        }
        self.x_center = Some(0.0);
        self.y_center = Some(0.0);

        let mut r_max_data = 0.0f64;
        let mut z_min_data = f64::INFINITY;
        let mut z_max_data = f64::NEG_INFINITY;
        for c in coordinates {
            let r = (c[0] * c[0] + c[1] * c[1]).sqrt();
            r_max_data = r_max_data.max(r);
            z_min_data = z_min_data.min(c[2]);
            z_max_data = z_max_data.max(c[2]);
        }

        let r_min = self.r_min_limit_input.unwrap_or(0.0);
        let r_max = self.r_max_limit_input.unwrap_or(r_max_data);
        let t_min = self.theta_min_input.unwrap_or(0.0);
        let t_max = self.theta_max_input.unwrap_or(2.0 * PI);
        let z_min = self.z_min_limit_input.unwrap_or(z_min_data);
        let z_max = self.z_max_limit_input.unwrap_or(z_max_data);

        self.r_min_limit = Some(r_min);
        self.r_max_limit = Some(r_max);
        self.theta_min = Some(t_min);
        self.theta_max = Some(t_max);
        self.z_min_limit = Some(z_min);
        self.z_max_limit = Some(z_max);

        let mut r_edges = match self.radial_mode.as_str() {
            "equal_area" => linspace(r_min * r_min, r_max * r_max, self.nr + 1)
                .into_iter()
                .map(|v| v.sqrt())
                .collect::<Vec<_>>(),
            "equal_dr" => linspace(r_min, r_max, self.nr + 1),
            other => {
                return Err(DemMcmError::InvalidArgument(format!(
                    "Unknown radial_mode: {other}"
                )))
            }
        };
        r_edges[0] = r_min;
        if let Some(last) = r_edges.last_mut() {
            *last = r_max * (1.0 + 1e-9);
        }

        let mut theta_edges = linspace(t_min, t_max, self.ntheta + 1);
        if let Some(last) = theta_edges.last_mut() {
            *last += 1e-9;
        }

        let mut z_edges = linspace(z_min, z_max, self.nz + 1);
        if let Some(last) = z_edges.last_mut() {
            if *last > 0.0 {
                *last *= 1.0 + 1e-9;
            }
            *last += 1e-9;
        }

        self.r_edges = Some(r_edges);
        self.theta_edges = Some(theta_edges);
        self.z_edges = Some(z_edges);
        Ok(())
    }

    fn compute_states(
        &self,
        coordinates: &[[f64; 3]],
        _velocities: Option<&[[f64; 3]]>,
    ) -> Result<Vec<i64>> {
        let (r_edges, theta_edges, z_edges) =
            match (&self.r_edges, &self.theta_edges, &self.z_edges) {
                (Some(r), Some(t), Some(z)) => (r, t, z),
                _ => {
                    return Err(DemMcmError::NotFitted(
                        "CylindricalPartitioner must be fitted before use".to_string(),
                    ))
                }
            };
        let x0 = self.x_center.unwrap_or(0.0);
        let y0 = self.y_center.unwrap_or(0.0);
        let r_min = self.r_min_limit.unwrap_or(0.0);
        let r_top = *r_edges.last().unwrap();
        let t_min = self.theta_min.unwrap_or(0.0);
        let t_top = *theta_edges.last().unwrap();
        let z_min = self.z_min_limit.unwrap_or(f64::NEG_INFINITY);
        let z_top = *z_edges.last().unwrap();
        let two_pi = 2.0 * PI;

        let mut states = vec![-1i64; coordinates.len()];
        for (idx, c) in coordinates.iter().enumerate() {
            let dx = c[0] - x0;
            let dy = c[1] - y0;
            let r = (dx * dx + dy * dy).sqrt();
            let theta = (dy.atan2(dx) + two_pi).rem_euclid(two_pi);
            let z = c[2];

            if r >= r_min && r < r_top && theta >= t_min && theta < t_top && z >= z_min && z < z_top
            {
                let ir = searchsorted_right(r_edges, r)
                    .saturating_sub(1)
                    .min(self.nr - 1);
                let itheta = searchsorted_right(theta_edges, theta)
                    .saturating_sub(1)
                    .min(self.ntheta - 1);
                let iz = searchsorted_right(z_edges, z)
                    .saturating_sub(1)
                    .min(self.nz - 1);
                states[idx] = (ir + itheta * self.nr + iz * self.nr * self.ntheta) as i64;
            }
        }
        Ok(states)
    }

    fn save_artifacts(&self) -> Result<PartitionerArtifacts> {
        let mut obj = JsonValue::object();
        obj.insert("nr", JsonValue::Number(self.nr as f64));
        obj.insert("ntheta", JsonValue::Number(self.ntheta as f64));
        obj.insert("nz", JsonValue::Number(self.nz as f64));
        obj.insert("radial_mode", JsonValue::String(self.radial_mode.clone()));
        let opt_num = |v: Option<f64>| v.map_or(JsonValue::Null, JsonValue::Number);
        obj.insert("theta_min", opt_num(self.theta_min));
        obj.insert("theta_max", opt_num(self.theta_max));
        obj.insert("z_min_limit", opt_num(self.z_min_limit));
        obj.insert("z_max_limit", opt_num(self.z_max_limit));
        obj.insert("r_min_limit", opt_num(self.r_min_limit));
        obj.insert("r_max_limit", opt_num(self.r_max_limit));
        obj.insert("x_center", opt_num(self.x_center));
        obj.insert("y_center", opt_num(self.y_center));
        let vec_json = |v: &Option<Vec<f64>>| {
            JsonValue::Array(
                v.as_deref()
                    .unwrap_or(&[])
                    .iter()
                    .map(|&x| JsonValue::Number(x))
                    .collect(),
            )
        };
        obj.insert("r_edges", vec_json(&self.r_edges));
        obj.insert("theta_edges", vec_json(&self.theta_edges));
        obj.insert("z_edges", vec_json(&self.z_edges));

        let mut art = PartitionerArtifacts::default();
        art.json_files.insert("cylindrical_data".to_string(), obj);
        Ok(art)
    }

    fn geometry_hint(&self) -> PartitionerGeometry {
        if let (Some(re), Some(te), Some(ze)) = (&self.r_edges, &self.theta_edges, &self.z_edges) {
            PartitionerGeometry::CylindricalGrid {
                x_center: self.x_center.unwrap_or(0.0),
                y_center: self.y_center.unwrap_or(0.0),
                r_edges: re.clone(),
                theta_edges: te.clone(),
                z_edges: ze.clone(),
            }
        } else {
            PartitionerGeometry::PointClusters
        }
    }

    fn load(&mut self, dir: &Path) -> Result<()> {
        let s = fs::read_to_string(dir.join("cylindrical_data.json"))?;
        let j = JsonValue::parse(&s)?;
        self.nr = j.get("nr").and_then(|v| v.as_usize()).unwrap_or(self.nr);
        self.ntheta = j
            .get("ntheta")
            .and_then(|v| v.as_usize())
            .unwrap_or(self.ntheta);
        self.nz = j.get("nz").and_then(|v| v.as_usize()).unwrap_or(self.nz);
        if let Some(rm) = j.get("radial_mode").and_then(|v| v.as_str()) {
            self.radial_mode = rm.to_string();
        }
        self.theta_min = j.get("theta_min").and_then(|v| v.as_f64());
        self.theta_max = j.get("theta_max").and_then(|v| v.as_f64());
        self.z_min_limit = j.get("z_min_limit").and_then(|v| v.as_f64());
        self.z_max_limit = j.get("z_max_limit").and_then(|v| v.as_f64());
        self.r_min_limit = j.get("r_min_limit").and_then(|v| v.as_f64());
        self.r_max_limit = j.get("r_max_limit").and_then(|v| v.as_f64());
        self.x_center = j.get("x_center").and_then(|v| v.as_f64());
        self.y_center = j.get("y_center").and_then(|v| v.as_f64());
        let read_vec = |k: &str| {
            j.get(k)
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_f64()).collect::<Vec<f64>>())
        };
        self.r_edges = read_vec("r_edges");
        self.theta_edges = read_vec("theta_edges");
        self.z_edges = read_vec("z_edges");
        Ok(())
    }
}

// ============================================================================
// 3. VORONOI (K-MEANS) PARTITIONER
// ============================================================================

#[derive(Debug, Clone)]
pub struct VoronoiPartitioner {
    pub n_cells_cfg: usize,
    pub random_state: u64,
    pub centroids: Option<Matrix>,
    tree: Option<KdTree>,
}

impl VoronoiPartitioner {
    pub fn new(n_cells: usize, random_state: u64) -> Self {
        Self {
            n_cells_cfg: n_cells,
            random_state,
            centroids: None,
            tree: None,
        }
    }
}

impl Partitioner for VoronoiPartitioner {
    fn method(&self) -> PartitioningMethod {
        PartitioningMethod::Voronoi
    }

    fn type_name(&self) -> &'static str {
        "VoronoiPartitioner"
    }

    fn n_cells(&self) -> usize {
        self.n_cells_cfg
    }

    fn label(&self) -> String {
        format!("voronoi_{}cells", self.n_cells_cfg)
    }

    fn fit(&mut self, coordinates: &[[f64; 3]]) -> Result<()> {
        if coordinates.is_empty() {
            return Err(DemMcmError::InvalidArgument(
                "VoronoiPartitioner::fit requires non-empty coordinates".to_string(),
            ));
        }
        let n = coordinates.len();
        let mut rng = Rng::seed_from_u64(self.random_state);
        let fit_mat = if n > 500_000 {
            let idx = rng.sample_indices(n, 500_000);
            let mut m = Matrix::zeros(idx.len(), 3);
            for (r, &i) in idx.iter().enumerate() {
                m.set(r, 0, coordinates[i][0]);
                m.set(r, 1, coordinates[i][1]);
                m.set(r, 2, coordinates[i][2]);
            }
            m
        } else {
            let mut m = Matrix::zeros(n, 3);
            for (r, c) in coordinates.iter().enumerate() {
                m.set(r, 0, c[0]);
                m.set(r, 1, c[1]);
                m.set(r, 2, c[2]);
            }
            m
        };

        let k = self.n_cells_cfg.min(fit_mat.rows).max(1);
        let km = KMeans::new(k)
            .with_random_state(self.random_state)
            .fit(&fit_mat)?;
        let centers = km.centroids;
        self.tree = Some(KdTree::build(&centers.data, centers.cols));
        self.centroids = Some(centers);
        Ok(())
    }

    fn compute_states(
        &self,
        coordinates: &[[f64; 3]],
        _velocities: Option<&[[f64; 3]]>,
    ) -> Result<Vec<i64>> {
        let tree = self.tree.as_ref().ok_or_else(|| {
            DemMcmError::NotFitted("VoronoiPartitioner must be fitted before use".to_string())
        })?;
        let mut out = Vec::with_capacity(coordinates.len());
        for c in coordinates {
            let (idx, _) = tree.query_nearest(c).unwrap();
            out.push(idx as i64);
        }
        Ok(out)
    }

    fn save_artifacts(&self) -> Result<PartitionerArtifacts> {
        let c = self.centroids.as_ref().ok_or_else(|| {
            DemMcmError::NotFitted("VoronoiPartitioner must be fitted before save()".to_string())
        })?;
        let mut art = PartitionerArtifacts::default();
        art.arrays
            .insert("centroids".to_string(), NpyArray::from_matrix(c));
        Ok(art)
    }

    fn geometry_hint(&self) -> PartitionerGeometry {
        if let Some(c) = &self.centroids {
            let seeds = (0..c.rows)
                .map(|r| [c.get(r, 0), c.get(r, 1), c.get(r, 2)])
                .collect();
            PartitionerGeometry::VoronoiSeeds(seeds)
        } else {
            PartitionerGeometry::PointClusters
        }
    }

    fn load(&mut self, dir: &Path) -> Result<()> {
        let c = NpyArray::load(&dir.join("centroids.npy"))?.to_matrix()?;
        self.tree = Some(KdTree::build(&c.data, c.cols));
        self.n_cells_cfg = c.rows;
        self.centroids = Some(c);
        Ok(())
    }
}

// ============================================================================
// 4. QUANTILE GRID PARTITIONER
// ============================================================================

#[derive(Debug, Clone)]
pub struct QuantileGridPartitioner {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub x_edges: Option<Vec<f64>>,
    pub y_edges: Option<Vec<f64>>,
    pub z_edges: Option<Vec<f64>>,
}

impl QuantileGridPartitioner {
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            nx,
            ny,
            nz,
            x_edges: None,
            y_edges: None,
            z_edges: None,
        }
    }
}

impl Partitioner for QuantileGridPartitioner {
    fn method(&self) -> PartitioningMethod {
        PartitioningMethod::Quantile
    }

    fn type_name(&self) -> &'static str {
        "QuantileGridPartitioner"
    }

    fn n_cells(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    fn label(&self) -> String {
        format!("quantile_nx{}_ny{}_nz{}", self.nx, self.ny, self.nz)
    }

    fn fit(&mut self, coordinates: &[[f64; 3]]) -> Result<()> {
        if coordinates.is_empty() {
            return Err(DemMcmError::InvalidArgument(
                "QuantileGridPartitioner::fit requires non-empty coordinates".to_string(),
            ));
        }
        let eps = 0.001;
        let compute_edges = |dim: usize, n_bins: usize| -> Vec<f64> {
            let mut vals: Vec<f64> = coordinates.iter().map(|c| c[dim]).collect();
            vals.sort_by(|a, b| a.total_cmp(b));
            let qs = linspace(0.0, 1.0, n_bins + 1);
            let mut edges: Vec<f64> = qs.into_iter().map(|q| quantile_sorted(&vals, q)).collect();
            edges[0] -= eps;
            *edges.last_mut().unwrap() += eps;
            edges
        };
        self.x_edges = Some(compute_edges(0, self.nx));
        self.y_edges = Some(compute_edges(1, self.ny));
        self.z_edges = Some(compute_edges(2, self.nz));
        Ok(())
    }

    fn compute_states(
        &self,
        coordinates: &[[f64; 3]],
        _velocities: Option<&[[f64; 3]]>,
    ) -> Result<Vec<i64>> {
        let (xe, ye, ze) = match (&self.x_edges, &self.y_edges, &self.z_edges) {
            (Some(x), Some(y), Some(z)) => (x, y, z),
            _ => {
                return Err(DemMcmError::NotFitted(
                    "QuantileGridPartitioner must be fitted before use".to_string(),
                ))
            }
        };
        let mut out = Vec::with_capacity(coordinates.len());
        for c in coordinates {
            let ix = searchsorted_right(xe, c[0])
                .saturating_sub(1)
                .min(self.nx - 1);
            let iy = searchsorted_right(ye, c[1])
                .saturating_sub(1)
                .min(self.ny - 1);
            let iz = searchsorted_right(ze, c[2])
                .saturating_sub(1)
                .min(self.nz - 1);
            out.push((ix + iy * self.nx + iz * self.nx * self.ny) as i64);
        }
        Ok(out)
    }

    fn save_artifacts(&self) -> Result<PartitionerArtifacts> {
        let (xe, ye, ze) = match (&self.x_edges, &self.y_edges, &self.z_edges) {
            (Some(x), Some(y), Some(z)) => (x, y, z),
            _ => {
                return Err(DemMcmError::NotFitted(
                    "QuantileGridPartitioner must be fitted before save()".to_string(),
                ))
            }
        };
        let mut art = PartitionerArtifacts::default();
        let mut obj = JsonValue::object();
        obj.insert(
            "x",
            JsonValue::Array(xe.iter().map(|&v| JsonValue::Number(v)).collect()),
        );
        obj.insert(
            "y",
            JsonValue::Array(ye.iter().map(|&v| JsonValue::Number(v)).collect()),
        );
        obj.insert(
            "z",
            JsonValue::Array(ze.iter().map(|&v| JsonValue::Number(v)).collect()),
        );
        art.json_files.insert("edges".to_string(), obj);
        Ok(art)
    }

    fn geometry_hint(&self) -> PartitionerGeometry {
        if let (Some(xe), Some(ye), Some(ze)) = (&self.x_edges, &self.y_edges, &self.z_edges) {
            PartitionerGeometry::TensorGrid {
                x_edges: xe.clone(),
                y_edges: ye.clone(),
                z_edges: ze.clone(),
            }
        } else {
            PartitionerGeometry::PointClusters
        }
    }

    fn load(&mut self, dir: &Path) -> Result<()> {
        let npz_path = dir.join("edges.npz");
        if npz_path.exists() {
            let map = crate::io::npy::load_npz(&npz_path)?;
            self.x_edges = Some(map["x"].to_f64_vec());
            self.y_edges = Some(map["y"].to_f64_vec());
            self.z_edges = Some(map["z"].to_f64_vec());
            return Ok(());
        }
        let s = fs::read_to_string(dir.join("edges.json"))?;
        let j = JsonValue::parse(&s)?;
        let read = |k: &str| -> Result<Vec<f64>> {
            j.get(k)
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_f64()).collect())
                .ok_or_else(|| DemMcmError::MissingData(format!("Missing {k} in edges.json")))
        };
        self.x_edges = Some(read("x")?);
        self.y_edges = Some(read("y")?);
        self.z_edges = Some(read("z")?);
        Ok(())
    }
}

// ============================================================================
// 5. ADAPTIVE OCTREE (AXIAL & OBLIQUE)
// ============================================================================

#[derive(Debug, Clone)]
pub struct HalfspaceConstraint {
    pub normal: [f64; 3],
    pub offset: f64,
    pub side_le: bool, // true = "le" (<=), false = "gt" (>)
}

#[derive(Debug, Clone)]
pub enum ObliqueNode {
    Internal {
        normal: [f64; 3],
        offset: f64,
        left: Box<ObliqueNode>,
        right: Box<ObliqueNode>,
    },
    Leaf {
        bounds: [f64; 6],
        centroid: [f64; 3],
        halfspaces: Vec<HalfspaceConstraint>,
    },
}

impl ObliqueNode {
    fn count_leaves(&self) -> usize {
        match self {
            Self::Leaf { .. } => 1,
            Self::Internal { left, right, .. } => left.count_leaves() + right.count_leaves(),
        }
    }

    fn flatten_leaves(&self, out: &mut Vec<([f64; 3], Vec<HalfspaceConstraint>)>) {
        match self {
            Self::Leaf {
                centroid,
                halfspaces,
                ..
            } => out.push((*centroid, halfspaces.clone())),
            Self::Internal { left, right, .. } => {
                left.flatten_leaves(out);
                right.flatten_leaves(out);
            }
        }
    }

    fn collect_leaves(&self, out: &mut Vec<([f64; 6], [f64; 3], Vec<HalfspaceConstraint>)>) {
        match self {
            Self::Leaf {
                bounds,
                centroid,
                halfspaces,
            } => out.push((*bounds, *centroid, halfspaces.clone())),
            Self::Internal { left, right, .. } => {
                left.collect_leaves(out);
                right.collect_leaves(out);
            }
        }
    }

    fn to_json(&self) -> JsonValue {
        let mut obj = JsonValue::object();
        match self {
            Self::Internal {
                normal,
                offset,
                left,
                right,
            } => {
                obj.insert("type", JsonValue::String("internal".to_string()));
                obj.insert(
                    "normal",
                    JsonValue::Array(normal.iter().map(|&v| JsonValue::Number(v)).collect()),
                );
                obj.insert("offset", JsonValue::Number(*offset));
                obj.insert("left", left.to_json());
                obj.insert("right", right.to_json());
            }
            Self::Leaf {
                bounds,
                centroid,
                halfspaces,
            } => {
                obj.insert("type", JsonValue::String("leaf".to_string()));
                obj.insert(
                    "bounds",
                    JsonValue::Array(bounds.iter().map(|&v| JsonValue::Number(v)).collect()),
                );
                obj.insert(
                    "centroid",
                    JsonValue::Array(centroid.iter().map(|&v| JsonValue::Number(v)).collect()),
                );
                let hs_arr = halfspaces
                    .iter()
                    .map(|h| {
                        let mut ho = JsonValue::object();
                        ho.insert(
                            "normal",
                            JsonValue::Array(
                                h.normal.iter().map(|&v| JsonValue::Number(v)).collect(),
                            ),
                        );
                        ho.insert("offset", JsonValue::Number(h.offset));
                        ho.insert(
                            "side",
                            JsonValue::String(if h.side_le { "le" } else { "gt" }.to_string()),
                        );
                        ho
                    })
                    .collect();
                obj.insert("halfspaces", JsonValue::Array(hs_arr));
            }
        }
        obj
    }

    fn from_json(j: &JsonValue) -> Result<Self> {
        let kind = j.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let arr3 = |val: Option<&JsonValue>| -> [f64; 3] {
            let a = val.and_then(|v| v.as_array());
            if let Some(v) = a {
                [
                    v.first().and_then(|x| x.as_f64()).unwrap_or(0.0),
                    v.get(1).and_then(|x| x.as_f64()).unwrap_or(0.0),
                    v.get(2).and_then(|x| x.as_f64()).unwrap_or(0.0),
                ]
            } else {
                [0.0; 3]
            }
        };
        match kind {
            "internal" => Ok(Self::Internal {
                normal: arr3(j.get("normal")),
                offset: j.get("offset").and_then(|v| v.as_f64()).unwrap_or(0.0),
                left: Box::new(Self::from_json(j.get("left").ok_or_else(|| {
                    DemMcmError::Serialization("Missing left child in oblique_tree".to_string())
                })?)?),
                right: Box::new(Self::from_json(j.get("right").ok_or_else(|| {
                    DemMcmError::Serialization("Missing right child in oblique_tree".to_string())
                })?)?),
            }),
            "leaf" => {
                let bounds_vec: Vec<f64> = j
                    .get("bounds")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|x| x.as_f64()).collect())
                    .unwrap_or_else(|| vec![0.0; 6]);
                let bounds = [
                    bounds_vec[0],
                    bounds_vec[1],
                    bounds_vec[2],
                    bounds_vec[3],
                    bounds_vec[4],
                    bounds_vec[5],
                ];
                let centroid = arr3(j.get("centroid"));
                let mut halfspaces = Vec::new();
                if let Some(arr) = j.get("halfspaces").and_then(|v| v.as_array()) {
                    for h in arr {
                        halfspaces.push(HalfspaceConstraint {
                            normal: arr3(h.get("normal")),
                            offset: h.get("offset").and_then(|v| v.as_f64()).unwrap_or(0.0),
                            side_le: h.get("side").and_then(|v| v.as_str()) != Some("gt"),
                        });
                    }
                }
                Ok(Self::Leaf {
                    bounds,
                    centroid,
                    halfspaces,
                })
            }
            other => Err(DemMcmError::Serialization(format!(
                "Unknown oblique node type {other:?}"
            ))),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OctreePartitioner {
    pub max_particles: usize,
    pub max_depth: usize,
    pub transform_type: Option<String>,
    pub oblique_method: Option<String>,

    pub leaves: Vec<[f64; 6]>,
    pub bounds: Option<[f64; 6]>,
    pub stats_min: Option<[f64; 3]>,
    pub stats_max: Option<[f64; 3]>,
    pub oblique_root: Option<ObliqueNode>,
}

impl OctreePartitioner {
    pub const OBLIQUE_METHODS: &'static [&'static str] =
        &["axis", "pca", "kmeans2", "2medians", "random", "svm"];

    pub fn new(
        max_particles: usize,
        max_depth: usize,
        transform_type: Option<String>,
        oblique_method: Option<String>,
    ) -> Self {
        Self {
            max_particles,
            max_depth,
            transform_type,
            oblique_method,
            leaves: Vec::new(),
            bounds: None,
            stats_min: None,
            stats_max: None,
            oblique_root: None,
        }
    }

    fn apply_transform(&self, coordinates: &[[f64; 3]]) -> Vec<[f64; 3]> {
        if self.transform_type.as_deref() == Some("normalize") {
            if let (Some(mins), Some(maxs)) = (self.stats_min, self.stats_max) {
                return coordinates
                    .iter()
                    .map(|c| {
                        [
                            (c[0] - mins[0]) / (maxs[0] - mins[0]),
                            (c[1] - mins[1]) / (maxs[1] - mins[1]),
                            (c[2] - mins[2]) / (maxs[2] - mins[2]),
                        ]
                    })
                    .collect();
            }
        }
        coordinates.to_vec()
    }

    fn subdivide(&mut self, coords: &[[f64; 3]], bounds: [f64; 6], depth: usize) {
        if coords.len() <= self.max_particles || depth >= self.max_depth {
            self.leaves.push(bounds);
            return;
        }
        let xs: Vec<f64> = coords.iter().map(|c| c[0]).collect();
        let ys: Vec<f64> = coords.iter().map(|c| c[1]).collect();
        let zs: Vec<f64> = coords.iter().map(|c| c[2]).collect();
        let xmid = median(&xs);
        let ymid = median(&ys);
        let zmid = median(&zs);

        let mut octants: [Vec<[f64; 3]>; 8] = Default::default();
        for &c in coords {
            let bit = (if c[0] >= xmid { 1 } else { 0 })
                + (if c[1] >= ymid { 2 } else { 0 })
                + (if c[2] >= zmid { 4 } else { 0 });
            octants[bit].push(c);
        }

        let [xmin, xmax, ymin, ymax, zmin, zmax] = bounds;
        for (idx, child_coords) in octants.iter().enumerate() {
            let ix = idx % 2 == 1;
            let iy = (idx / 2) % 2 == 1;
            let iz = idx / 4 == 1;
            let child_bounds = [
                if ix { xmid } else { xmin },
                if ix { xmax } else { xmid },
                if iy { ymid } else { ymin },
                if iy { ymax } else { ymid },
                if iz { zmid } else { zmin },
                if iz { zmax } else { zmid },
            ];
            self.subdivide(child_coords, child_bounds, depth + 1);
        }
    }

    fn find_splitting_plane(coords: &[[f64; 3]], method: &str) -> Result<([f64; 3], f64)> {
        if coords.len() < 2 {
            return Ok(([1.0, 0.0, 0.0], 0.0));
        }
        match method {
            "pca" => {
                let mut m = Matrix::zeros(coords.len(), 3);
                for (r, c) in coords.iter().enumerate() {
                    m.set(r, 0, c[0]);
                    m.set(r, 1, c[1]);
                    m.set(r, 2, c[2]);
                }
                let cov = covariance_matrix(&m)?;
                let (eigvals, eigvecs) = jacobi_eigh(&cov)?;
                let best_col = eigvals
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let normal = [
                    eigvecs.get(0, best_col),
                    eigvecs.get(1, best_col),
                    eigvecs.get(2, best_col),
                ];
                let proj: Vec<f64> = coords
                    .iter()
                    .map(|c| c[0] * normal[0] + c[1] * normal[1] + c[2] * normal[2])
                    .collect();
                Ok((normal, median(&proj)))
            }
            "kmeans2" => {
                let n = coords.len().min(10_000);
                let mut m = Matrix::zeros(n, 3);
                for (r, c) in coords.iter().take(n).enumerate() {
                    m.set(r, 0, c[0]);
                    m.set(r, 1, c[1]);
                    m.set(r, 2, c[2]);
                }
                let km = KMeans::new(2)
                    .with_n_init(3)
                    .with_random_state(42)
                    .fit(&m)?;
                let cc = km.centroids;
                let c1 = [cc.get(0, 0), cc.get(0, 1), cc.get(0, 2)];
                let c2 = [cc.get(1, 0), cc.get(1, 1), cc.get(1, 2)];
                let diff = [c2[0] - c1[0], c2[1] - c1[1], c2[2] - c1[2]];
                let norm = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]).sqrt();
                let normal = if norm > 1e-12 {
                    [diff[0] / norm, diff[1] / norm, diff[2] / norm]
                } else {
                    [1.0, 0.0, 0.0]
                };
                let mid = [
                    0.5 * (c1[0] + c2[0]),
                    0.5 * (c1[1] + c2[1]),
                    0.5 * (c1[2] + c2[2]),
                ];
                let offset = mid[0] * normal[0] + mid[1] * normal[1] + mid[2] * normal[2];
                Ok((normal, offset))
            }
            "2medians" => {
                let mut rng = Rng::seed_from_u64(42);
                let idx = rng.sample_indices(coords.len(), coords.len().min(2));
                let mut c1 = coords[idx[0]];
                let mut c2 = if idx.len() > 1 {
                    coords[idx[1]]
                } else {
                    coords[0]
                };
                for _ in 0..5 {
                    let mut g0 = Vec::new();
                    let mut g1 = Vec::new();
                    for &p in coords {
                        let d1 = ((p[0] - c1[0]).powi(2)
                            + (p[1] - c1[1]).powi(2)
                            + (p[2] - c1[2]).powi(2))
                        .sqrt();
                        let d2 = ((p[0] - c2[0]).powi(2)
                            + (p[1] - c2[1]).powi(2)
                            + (p[2] - c2[2]).powi(2))
                        .sqrt();
                        if d1 <= d2 {
                            g0.push(p);
                        } else {
                            g1.push(p);
                        }
                    }
                    if !g0.is_empty() {
                        c1 = [
                            median(&g0.iter().map(|p| p[0]).collect::<Vec<_>>()),
                            median(&g0.iter().map(|p| p[1]).collect::<Vec<_>>()),
                            median(&g0.iter().map(|p| p[2]).collect::<Vec<_>>()),
                        ];
                    }
                    if !g1.is_empty() {
                        c2 = [
                            median(&g1.iter().map(|p| p[0]).collect::<Vec<_>>()),
                            median(&g1.iter().map(|p| p[1]).collect::<Vec<_>>()),
                            median(&g1.iter().map(|p| p[2]).collect::<Vec<_>>()),
                        ];
                    }
                }
                let diff = [c2[0] - c1[0], c2[1] - c1[1], c2[2] - c1[2]];
                let norm = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]).sqrt();
                let normal = if norm > 1e-12 {
                    [diff[0] / norm, diff[1] / norm, diff[2] / norm]
                } else {
                    [1.0, 0.0, 0.0]
                };
                let mid = [
                    0.5 * (c1[0] + c2[0]),
                    0.5 * (c1[1] + c2[1]),
                    0.5 * (c1[2] + c2[2]),
                ];
                let offset = mid[0] * normal[0] + mid[1] * normal[1] + mid[2] * normal[2];
                Ok((normal, offset))
            }
            "random" => {
                let mut rng = Rng::seed_from_u64(4);
                let mut normal = [rng.next_normal(), rng.next_normal(), rng.next_normal()];
                let norm =
                    (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
                if norm > 1e-12 {
                    normal[0] /= norm;
                    normal[1] /= norm;
                    normal[2] /= norm;
                } else {
                    normal = [1.0, 0.0, 0.0];
                }
                let proj: Vec<f64> = coords
                    .iter()
                    .map(|c| c[0] * normal[0] + c[1] * normal[1] + c[2] * normal[2])
                    .collect();
                Ok((normal, median(&proj)))
            }
            "svm" => {
                let mut m = Matrix::zeros(coords.len(), 3);
                for (r, c) in coords.iter().enumerate() {
                    m.set(r, 0, c[0]);
                    m.set(r, 1, c[1]);
                    m.set(r, 2, c[2]);
                }
                let cov = covariance_matrix(&m)?;
                let (eigvals, eigvecs) = jacobi_eigh(&cov)?;
                let best_col = eigvals
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let pca_normal = [
                    eigvecs.get(0, best_col),
                    eigvecs.get(1, best_col),
                    eigvecs.get(2, best_col),
                ];
                let proj_pca: Vec<f64> = coords
                    .iter()
                    .map(|c| c[0] * pca_normal[0] + c[1] * pca_normal[1] + c[2] * pca_normal[2])
                    .collect();
                let med = median(&proj_pca);
                let labels: Vec<i32> = proj_pca
                    .iter()
                    .map(|&p| if p > med { 1 } else { 0 })
                    .collect();
                let has_0 = labels.contains(&0);
                let has_1 = labels.contains(&1);
                if !(has_0 && has_1) {
                    return Ok(([1.0, 0.0, 0.0], 0.0));
                }
                let svc = LinearSvc::new(1.0)
                    .with_max_iter(1000)
                    .with_random_state(42);
                let (coef, intercept) = svc.fit(&m, &labels)?;
                let wn = (coef[0].powi(2) + coef[1].powi(2) + coef[2].powi(2)).sqrt();
                if wn < 1e-12 {
                    Ok(([1.0, 0.0, 0.0], 0.0))
                } else {
                    Ok(([coef[0] / wn, coef[1] / wn, coef[2] / wn], -intercept / wn))
                }
            }
            other => Err(DemMcmError::InvalidArgument(format!(
                "Unknown oblique method: {other}"
            ))),
        }
    }

    fn build_oblique_tree(
        &self,
        coords: &[[f64; 3]],
        bounds: [f64; 6],
        depth: usize,
        halfspaces: Vec<HalfspaceConstraint>,
    ) -> Result<ObliqueNode> {
        let make_leaf = || {
            let centroid = if coords.is_empty() {
                [0.0; 3]
            } else {
                let n = coords.len() as f64;
                let s = coords.iter().fold([0.0; 3], |acc, c| {
                    [acc[0] + c[0], acc[1] + c[1], acc[2] + c[2]]
                });
                [s[0] / n, s[1] / n, s[2] / n]
            };
            ObliqueNode::Leaf {
                bounds,
                centroid,
                halfspaces: halfspaces.clone(),
            }
        };

        if coords.len() <= self.max_particles || depth >= self.max_depth {
            return Ok(make_leaf());
        }

        let method = self.oblique_method.as_deref().unwrap_or("pca");
        let (normal, offset) = Self::find_splitting_plane(coords, method)?;
        let mut left_coords = Vec::new();
        let mut right_coords = Vec::new();
        for &c in coords {
            let proj = c[0] * normal[0] + c[1] * normal[1] + c[2] * normal[2];
            if proj <= offset {
                left_coords.push(c);
            } else {
                right_coords.push(c);
            }
        }
        if left_coords.is_empty() || right_coords.is_empty() {
            return Ok(make_leaf());
        }

        let mut hs_left = halfspaces.clone();
        hs_left.push(HalfspaceConstraint {
            normal,
            offset,
            side_le: true,
        });
        let mut hs_right = halfspaces;
        hs_right.push(HalfspaceConstraint {
            normal,
            offset,
            side_le: false,
        });

        Ok(ObliqueNode::Internal {
            normal,
            offset,
            left: Box::new(self.build_oblique_tree(&left_coords, bounds, depth + 1, hs_left)?),
            right: Box::new(self.build_oblique_tree(&right_coords, bounds, depth + 1, hs_right)?),
        })
    }
}

impl Partitioner for OctreePartitioner {
    fn method(&self) -> PartitioningMethod {
        PartitioningMethod::Octree
    }

    fn type_name(&self) -> &'static str {
        "OctreePartitioner"
    }

    fn splitting_method(&self) -> String {
        format!(
            "octree_{}",
            self.oblique_method.as_deref().unwrap_or("axis")
        )
    }

    fn n_cells(&self) -> usize {
        if let Some(root) = &self.oblique_root {
            root.count_leaves()
        } else {
            self.leaves.len()
        }
    }

    fn label(&self) -> String {
        let om = self.oblique_method.as_deref().unwrap_or("axis");
        format!("octree_mp{}_md{}_{om}", self.max_particles, self.max_depth)
    }

    fn fit(&mut self, coordinates: &[[f64; 3]]) -> Result<()> {
        if coordinates.is_empty() {
            return Err(DemMcmError::InvalidArgument(
                "OctreePartitioner::fit requires non-empty coordinates".to_string(),
            ));
        }
        let eps = 0.001;
        let mut mins = [f64::INFINITY; 3];
        let mut maxs = [f64::NEG_INFINITY; 3];
        for c in coordinates {
            for d in 0..3 {
                mins[d] = mins[d].min(c[d]);
                maxs[d] = maxs[d].max(c[d]);
            }
        }
        self.stats_min = Some([mins[0] - eps, mins[1] - eps, mins[2] - eps]);
        self.stats_max = Some([maxs[0] + eps, maxs[1] + eps, maxs[2] + eps]);

        let transformed = self.apply_transform(coordinates);
        let mut tmin = [f64::INFINITY; 3];
        let mut tmax = [f64::NEG_INFINITY; 3];
        for c in &transformed {
            for d in 0..3 {
                tmin[d] = tmin[d].min(c[d]);
                tmax[d] = tmax[d].max(c[d]);
            }
        }
        let bounds = [
            tmin[0] - eps,
            tmax[0] + eps,
            tmin[1] - eps,
            tmax[1] + eps,
            tmin[2] - eps,
            tmax[2] + eps,
        ];
        self.bounds = Some(bounds);
        self.leaves.clear();

        let is_oblique = self.oblique_method.as_deref().is_some_and(|m| m != "axis");
        if is_oblique {
            self.oblique_root =
                Some(self.build_oblique_tree(&transformed, bounds, 0, Vec::new())?);
        } else {
            self.oblique_root = None;
            self.subdivide(&transformed, bounds, 0);
        }
        Ok(())
    }

    fn compute_states(
        &self,
        coordinates: &[[f64; 3]],
        _velocities: Option<&[[f64; 3]]>,
    ) -> Result<Vec<i64>> {
        if self.oblique_root.is_none() && self.leaves.is_empty() {
            return Err(DemMcmError::NotFitted(
                "OctreePartitioner must be fitted before use".to_string(),
            ));
        }

        let transformed_buf = self.apply_transform(coordinates);
        let qcoords: &[[f64; 3]] = &transformed_buf;

        if let Some(root) = &self.oblique_root {
            let mut leaves = Vec::new();
            root.flatten_leaves(&mut leaves);
            let mut states = vec![-1i64; qcoords.len()];
            for (cell_id, (_, hs)) in leaves.iter().enumerate() {
                for (i, c) in qcoords.iter().enumerate() {
                    let ok = hs.iter().all(|h| {
                        let proj = c[0] * h.normal[0] + c[1] * h.normal[1] + c[2] * h.normal[2];
                        if h.side_le {
                            proj <= h.offset
                        } else {
                            proj > h.offset
                        }
                    });
                    if ok {
                        states[i] = cell_id as i64;
                    }
                }
            }
            if states.contains(&-1) {
                let centers: Vec<Vec<f64>> = leaves.iter().map(|(c, _)| c.to_vec()).collect();
                let tree = KdTree::from_rows(&centers);
                for (i, s) in states.iter_mut().enumerate() {
                    if *s == -1 {
                        let (idx, _) = tree.query_nearest(&qcoords[i]).unwrap();
                        *s = idx as i64;
                    }
                }
            }
            return Ok(states);
        }

        // Axial octree bounding boxes
        let mut states = vec![-1i64; qcoords.len()];
        for (cell_id, b) in self.leaves.iter().enumerate() {
            for (i, c) in qcoords.iter().enumerate() {
                if c[0] >= b[0]
                    && c[0] < b[1]
                    && c[1] >= b[2]
                    && c[1] < b[3]
                    && c[2] >= b[4]
                    && c[2] < b[5]
                {
                    states[i] = cell_id as i64;
                }
            }
        }
        if states.contains(&-1) {
            let centers: Vec<Vec<f64>> = self
                .leaves
                .iter()
                .map(|b| {
                    vec![
                        0.5 * (b[0] + b[1]),
                        0.5 * (b[2] + b[3]),
                        0.5 * (b[4] + b[5]),
                    ]
                })
                .collect();
            let tree = KdTree::from_rows(&centers);
            for (i, s) in states.iter_mut().enumerate() {
                if *s == -1 {
                    let (idx, _) = tree.query_nearest(&qcoords[i]).unwrap();
                    *s = idx as i64;
                }
            }
        }
        Ok(states)
    }

    fn save_artifacts(&self) -> Result<PartitionerArtifacts> {
        let mut art = PartitionerArtifacts::default();
        if let Some(root) = &self.oblique_root {
            art.json_files
                .insert("oblique_tree".to_string(), root.to_json());
            art.text_files.insert(
                "octree_mode.txt".to_string(),
                format!(
                    "oblique\n{}\n",
                    self.oblique_method.as_deref().unwrap_or("axis")
                ),
            );
        } else {
            let mut m = Matrix::zeros(self.leaves.len(), 6);
            for (r, b) in self.leaves.iter().enumerate() {
                for (c, &val) in b.iter().enumerate() {
                    m.set(r, c, val);
                }
            }
            art.arrays
                .insert("leaves".to_string(), NpyArray::from_matrix(&m));
            art.text_files
                .insert("octree_mode.txt".to_string(), "axis\n".to_string());
        }
        if let Some(b) = self.bounds {
            art.arrays
                .insert("bounds".to_string(), NpyArray::from_f64_vec(b.to_vec()));
        }
        Ok(art)
    }

    fn geometry_hint(&self) -> PartitionerGeometry {
        let inv_bounds = |b: [f64; 6]| -> [f64; 6] {
            if self.transform_type.as_deref() == Some("normalize") {
                if let (Some(mins), Some(maxs)) = (self.stats_min, self.stats_max) {
                    let s0 = maxs[0] - mins[0];
                    let s1 = maxs[1] - mins[1];
                    let s2 = maxs[2] - mins[2];
                    return [
                        mins[0] + b[0] * s0,
                        mins[0] + b[1] * s0,
                        mins[1] + b[2] * s1,
                        mins[1] + b[3] * s1,
                        mins[2] + b[4] * s2,
                        mins[2] + b[5] * s2,
                    ];
                }
            }
            b
        };
        if let Some(root) = &self.oblique_root {
            let mut leaves = Vec::new();
            root.collect_leaves(&mut leaves);
            let mut domain_bounds = inv_bounds(self.bounds.unwrap_or([0.0; 6]));
            if let (Some(mins), Some(maxs)) = (self.stats_min, self.stats_max) {
                domain_bounds = [mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]];
            }
            let cells = leaves
                .into_iter()
                .map(|(b, cen, hs)| {
                    let b_orig = inv_bounds(b);
                    let mut c_orig = cen;
                    let mut hs_orig = Vec::with_capacity(hs.len());
                    if self.transform_type.as_deref() == Some("normalize") {
                        if let (Some(mins), Some(maxs)) = (self.stats_min, self.stats_max) {
                            let sp = [maxs[0] - mins[0], maxs[1] - mins[1], maxs[2] - mins[2]];
                            c_orig = [
                                mins[0] + cen[0] * sp[0],
                                mins[1] + cen[1] * sp[1],
                                mins[2] + cen[2] * sp[2],
                            ];
                            for h in hs {
                                // n_t . ((x - min)/sp) <= off  =>  sum (n_t/sp)*x <= off + sum (n_t*min/sp)
                                let nn = [
                                    h.normal[0] / sp[0],
                                    h.normal[1] / sp[1],
                                    h.normal[2] / sp[2],
                                ];
                                let off = h.offset
                                    + h.normal[0] * mins[0] / sp[0]
                                    + h.normal[1] * mins[1] / sp[1]
                                    + h.normal[2] * mins[2] / sp[2];
                                hs_orig.push((nn, off, h.side_le));
                            }
                        }
                    } else {
                        for h in hs {
                            hs_orig.push((h.normal, h.offset, h.side_le));
                        }
                    }
                    ObliqueCellGeometry {
                        bounds: b_orig,
                        centroid: c_orig,
                        halfspaces: hs_orig,
                    }
                })
                .collect();
            PartitionerGeometry::ObliquePolyhedra {
                domain_bounds,
                cells,
            }
        } else if !self.leaves.is_empty() {
            PartitionerGeometry::AxisBoxes(self.leaves.iter().copied().map(inv_bounds).collect())
        } else {
            PartitionerGeometry::PointClusters
        }
    }

    fn load(&mut self, dir: &Path) -> Result<()> {
        let mode_path = dir.join("octree_mode.txt");
        let mode = if mode_path.exists() {
            let txt = fs::read_to_string(&mode_path)?;
            let mut lines = txt.lines();
            let m = lines.next().unwrap_or("axis").trim().to_string();
            if m == "oblique" {
                if let Some(om) = lines.next() {
                    self.oblique_method = Some(om.trim().to_string());
                }
            }
            m
        } else {
            "axis".to_string()
        };

        if mode == "oblique" {
            let json_path = dir.join("oblique_tree.json");
            if json_path.exists() {
                let s = fs::read_to_string(&json_path)?;
                let j = JsonValue::parse(&s)?;
                self.oblique_root = Some(ObliqueNode::from_json(&j)?);
                self.leaves.clear();
            } else {
                return Err(DemMcmError::DataSource(
                    "Oblique octree JSON not found (Python pickle cannot be loaded in pure Rust)"
                        .to_string(),
                ));
            }
        } else {
            let m = NpyArray::load(&dir.join("leaves.npy"))?.to_matrix()?;
            self.leaves = (0..m.rows)
                .map(|r| {
                    let row = m.row(r);
                    [row[0], row[1], row[2], row[3], row[4], row[5]]
                })
                .collect();
            self.oblique_root = None;
        }

        let b_path = dir.join("bounds.npy");
        if b_path.exists() {
            let b = NpyArray::load(&b_path)?.to_f64_vec();
            if b.len() == 6 {
                self.bounds = Some([b[0], b[1], b[2], b[3], b[4], b[5]]);
            }
        }
        Ok(())
    }
}

#[allow(dead_code)]
fn _unused_hm(_h: HashMap<String, JsonValue>) {}
