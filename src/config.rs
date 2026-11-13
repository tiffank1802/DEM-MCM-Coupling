//! Configuration, constants, and shared types for `dem_mcm_coupling`.
//!
//! Mirrors `dem_mcm_coupling/_config.py` and enforces the single-source
//! row-stochastic matrix convention:
//! - `P[i, j]` is the probability to jump from state `i` to state `j`;
//! - rows sum to 1 (`P.sum(axis=1) == 1`);
//! - a state vector evolves via right multiplication: `phi_next = phi @ P`.

use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

use crate::error::{DemMcmError, Result};
use crate::io::json::JsonValue;
use crate::linalg::Matrix;

/// Identifier of the Hugging Face dataset repository holding DEM simulations
/// and pre-computed Markov results.
pub const BUCKET_ID: &str = "ktongue/DEM_MCM";

/// Default bucket prefix (all particles, both diameters).
pub const DEFAULT_BUCKET_PREFIX: &str = "_Good/Experiment";

/// Conversion factor between a DEM timestep index and seconds.
pub const TIMESTEP_TO_SECONDS: f64 = 0.01;

/// Default parquet file name of the complete DEM simulation inside a directory.
pub const SIMULATION_PARQUET_NAME: &str = "simulation_complete.parquet";

/// Timestep from which the DEM simulation is considered stationary.
pub const PERMANENT_START: usize = 250;

/// Typical particle count per DEM timestep in the reference simulation.
pub const N_PARTICLES_PER_TIMESTEP: usize = 1030;

/// Return the bucket prefix associated with an optional particle diameter (metres).
pub fn get_bucket_prefix(particle_diameter: Option<f64>) -> &'static str {
    match particle_diameter {
        Some(d) if (d - 0.004).abs() < 1e-9 => "_Good/SMALL",
        Some(d) if (d - 0.008).abs() < 1e-9 => "_Good/BIG",
        _ => DEFAULT_BUCKET_PREFIX,
    }
}

/// All 14 supported spatial partitioning methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PartitioningMethod {
    Cartesian,
    Cylindrical,
    Voronoi,
    Quantile,
    Octree,
    Physics,
    PhysicsFullVel,
    Gmm,
    Spectral,
    SpectralBiclustering,
    Adaptive,
    Multizone,
    Single,
    Dbscan,
}

impl PartitioningMethod {
    /// Return all 14 partitioning methods in registry order.
    pub fn all() -> &'static [PartitioningMethod] {
        &[
            Self::Cartesian,
            Self::Cylindrical,
            Self::Voronoi,
            Self::Quantile,
            Self::Octree,
            Self::Physics,
            Self::PhysicsFullVel,
            Self::Spectral,
            Self::Gmm,
            Self::SpectralBiclustering,
            Self::Adaptive,
            Self::Multizone,
            Self::Single,
            Self::Dbscan,
        ]
    }

    /// Canonical lowercase string identifier matching Python's `REGISTRY`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cartesian => "cartesian",
            Self::Cylindrical => "cylindrical",
            Self::Voronoi => "voronoi",
            Self::Quantile => "quantile",
            Self::Octree => "octree",
            Self::Physics => "physics",
            Self::PhysicsFullVel => "physics_full_vel",
            Self::Gmm => "gmm",
            Self::Spectral => "spectral",
            Self::SpectralBiclustering => "spectral_biclustering",
            Self::Adaptive => "adaptive",
            Self::Multizone => "multizone",
            Self::Single => "single",
            Self::Dbscan => "dbscan",
        }
    }
}

impl fmt::Display for PartitioningMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PartitioningMethod {
    type Err = DemMcmError;

    fn from_str(s: &str) -> Result<Self> {
        validate_partitioning_method(s)
    }
}

/// Validate a string identifier into a [`PartitioningMethod`].
pub fn validate_partitioning_method(method: &str) -> Result<PartitioningMethod> {
    match method {
        "cartesian" => Ok(PartitioningMethod::Cartesian),
        "cylindrical" => Ok(PartitioningMethod::Cylindrical),
        "voronoi" => Ok(PartitioningMethod::Voronoi),
        "quantile" => Ok(PartitioningMethod::Quantile),
        "octree" => Ok(PartitioningMethod::Octree),
        "physics" | "physics_aware" => Ok(PartitioningMethod::Physics),
        "physics_full_vel" | "full_velocity_kmeans" => Ok(PartitioningMethod::PhysicsFullVel),
        "gmm" => Ok(PartitioningMethod::Gmm),
        "spectral" => Ok(PartitioningMethod::Spectral),
        "spectral_biclustering" | "biclustering" => Ok(PartitioningMethod::SpectralBiclustering),
        "adaptive" => Ok(PartitioningMethod::Adaptive),
        "multizone" => Ok(PartitioningMethod::Multizone),
        "single" | "single_cell" => Ok(PartitioningMethod::Single),
        "dbscan" => Ok(PartitioningMethod::Dbscan),
        _ => {
            let valid: Vec<&str> = PartitioningMethod::all()
                .iter()
                .map(|m| m.as_str())
                .collect();
            Err(DemMcmError::InvalidMethod(format!(
                "Unknown partitioning method: {method:?}. Valid methods: {}",
                valid.join(", ")
            )))
        }
    }
}

/// Configuration metadata of a single partitioner instance.
#[derive(Debug, Clone)]
pub struct PartitionerConfig {
    pub method: PartitioningMethod,
    pub method_kwargs: HashMap<String, JsonValue>,
    pub n_cells: usize,
    pub label: String,
}

/// Initial Markov state vector `phi(0)` and its metadata.
#[derive(Debug, Clone)]
pub struct StateVectorData {
    /// Particle counts per partition cell, length `n_states`.
    pub phi: Vec<f64>,
    /// Index of the DEM timestep the vector was built from.
    pub timestamp: usize,
    /// Total number of particles represented by `phi`.
    pub total_particles: usize,
    /// Human-readable description of the vector.
    pub description: String,
}

impl StateVectorData {
    /// Check that `phi.iter().sum()` matches `total_particles` within `rtol`.
    pub fn validate_normalization(&self, rtol: f64) -> bool {
        let sum: f64 = self.phi.iter().sum();
        let expected = self.total_particles as f64;
        let tol = rtol * expected.abs();
        (sum - expected).abs() <= tol + 1e-8
    }
}

/// Trajectory produced by a Markov propagation.
#[derive(Debug, Clone)]
pub struct MarkovTrajectory {
    /// State vectors over time, shape `(n_steps + 1, n_states)`.
    pub states: Matrix,
    /// Timestep indices, length `n_steps + 1`.
    pub times: Vec<usize>,
    /// Times in seconds (`times * TIMESTEP_TO_SECONDS`).
    pub times_seconds: Vec<f64>,
    /// Partitioning method used for the simulation.
    pub method: String,
    /// Human-readable description of the trajectory.
    pub description: String,
}

/// Per-species data of a homogeneous or inhomogeneous experiment.
#[derive(Debug, Clone)]
pub struct SpeciesExperimentData {
    /// Homogeneous row-stochastic transition matrix `(n_states, n_states)` if homogeneous.
    pub p: Option<Matrix>,
    /// Inhomogeneous block transition matrices `(n_blocks, n_states, n_states)` if inhomogeneous.
    pub p_blocks: Option<Vec<Matrix>>,
    /// DEM state trajectory counts `(n_timesteps, n_states)`.
    pub s_matrix: Matrix,
    /// DEM timestep indices `(n_timesteps,)`.
    pub times: Vec<i64>,
}

/// Complete experiment data loaded from or written to a data source.
#[derive(Debug, Clone)]
pub struct ExperimentData {
    pub config: JsonValue,
    pub stats: JsonValue,
    pub species: HashMap<String, SpeciesExperimentData>,
    /// Optional raw particle-state matrix `(n_timesteps, n_particles)`.
    pub states_matrix: Option<Vec<Vec<i64>>>,
    pub inhomogeneous: bool,
    pub inhomogeneous_metadata: Option<JsonValue>,
}
