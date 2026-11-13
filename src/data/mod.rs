//! Pluggable DEM & Markov data-source abstraction (`DataSource`, `DemFrame`, `DemSnapshot`).
//!
//! Mirrors `dem_mcm_coupling/data/` so the Markov engine and sweep pipelines
//! work uniformly across local directories (`.parquet`, `.parquet.gz`, `.csv`,
//! `.npy`/`.json`) and in-memory stores.

pub mod local;
pub mod memory;

use std::collections::{BTreeMap, HashMap};

use crate::config::ExperimentData;
use crate::error::{DemMcmError, Result};
use crate::io::json::JsonValue;
use crate::io::npy::NpyArray;

pub use local::LocalDataSource;
pub use memory::InMemoryDataSource;

/// Row-per-particle DEM data at a single timestep (Rust counterpart of a
/// single-timestep `pandas.DataFrame`).
#[derive(Debug, Clone, PartialEq)]
pub struct DemFrame {
    pub timestep: usize,
    pub coordinates: Vec<[f64; 3]>,
    pub velocities: Vec<[f64; 3]>,
    pub diameters: Vec<f64>,
    pub particle_ids: Vec<i64>,
    pub residence_times: Vec<f64>,
    pub time: Option<f64>,
}

impl DemFrame {
    pub fn new(timestep: usize) -> Self {
        Self {
            timestep,
            coordinates: Vec::new(),
            velocities: Vec::new(),
            diameters: Vec::new(),
            particle_ids: Vec::new(),
            residence_times: Vec::new(),
            time: None,
        }
    }

    pub fn from_coords(timestep: usize, coordinates: Vec<[f64; 3]>) -> Self {
        Self {
            timestep,
            coordinates,
            velocities: Vec::new(),
            diameters: Vec::new(),
            particle_ids: Vec::new(),
            residence_times: Vec::new(),
            time: None,
        }
    }

    pub fn with_kinematics(
        timestep: usize,
        coordinates: Vec<[f64; 3]>,
        velocities: Vec<[f64; 3]>,
        diameters: Vec<f64>,
    ) -> Self {
        Self {
            timestep,
            coordinates,
            velocities,
            diameters,
            particle_ids: Vec::new(),
            residence_times: Vec::new(),
            time: None,
        }
    }

    pub fn n_particles(&self) -> usize {
        self.coordinates.len()
    }

    pub fn to_snapshot(&self) -> DemSnapshot {
        DemSnapshot {
            timestep: self.timestep,
            coordinates: self.coordinates.clone(),
            velocities: self.velocities.clone(),
            diameters: self.diameters.clone(),
            particle_ids: self.particle_ids.clone(),
        }
    }
}

/// Particle data at a single DEM timestep (`DemSnapshot` mirror).
#[derive(Debug, Clone, PartialEq)]
pub struct DemSnapshot {
    pub timestep: usize,
    pub coordinates: Vec<[f64; 3]>,
    pub velocities: Vec<[f64; 3]>,
    pub diameters: Vec<f64>,
    pub particle_ids: Vec<i64>,
}

impl DemSnapshot {
    pub fn n_particles(&self) -> usize {
        self.coordinates.len()
    }
}

/// Stacked `(coordinates, velocities, diameters)` returned by `DataSource::sample_coordinates`.
#[derive(Debug, Clone)]
pub struct SampledCoordinates {
    pub coordinates: Vec<[f64; 3]>,
    pub velocities: Vec<[f64; 3]>,
    pub diameters: Vec<f64>,
}

/// Payload passed to [`DataSource::write_experiment`].
#[derive(Debug, Clone)]
pub struct ExperimentWritePayload<'a> {
    pub folder_name: &'a str,
    pub stats: &'a JsonValue,
    pub config: &'a JsonValue,
    pub species_arrays: HashMap<String, NpyArray>,
    pub partitioner_arrays: HashMap<String, NpyArray>,
    pub partitioner_json: HashMap<String, JsonValue>,
    pub image_data: HashMap<String, Vec<u8>>,
    pub prefix: Option<&'a str>,
    pub inhomogeneous_metadata: Option<&'a JsonValue>,
}

/// Abstract interface of a DEM/Markov data source.
pub trait DataSource {
    /// Load DEM particle data keyed by sorted timestep index.
    fn read_timesteps(
        &self,
        timestep_indices: Option<&[usize]>,
    ) -> Result<BTreeMap<usize, DemFrame>>;

    /// List sorted names of stored Markov experiment folders.
    fn list_experiments(&self, prefix: Option<&str>) -> Result<Vec<String>>;

    /// Load a pre-computed Markov experiment.
    fn read_experiment(&self, folder_name: &str, prefix: Option<&str>) -> Result<ExperimentData>;

    /// Persist a Markov experiment.
    fn write_experiment(&mut self, payload: ExperimentWritePayload<'_>) -> Result<()>;

    /// Return sorted list of available timestep indices.
    fn list_timesteps(&self) -> Result<Vec<usize>> {
        Ok(self.read_timesteps(None)?.keys().copied().collect())
    }

    /// Return a single timestep as a [`DemSnapshot`].
    fn snapshot(&self, timestep: usize) -> Result<DemSnapshot> {
        let map = self.read_timesteps(Some(&[timestep]))?;
        match map.get(&timestep) {
            Some(frame) => Ok(frame.to_snapshot()),
            None => Err(DemMcmError::DataSource(format!(
                "Timestep {timestep} not found in data source"
            ))),
        }
    }

    /// Stack coordinates, velocities, and diameters over timesteps.
    fn sample_coordinates(&self, timestep_indices: Option<&[usize]>) -> Result<SampledCoordinates> {
        let map = self.read_timesteps(timestep_indices)?;
        let mut coordinates = Vec::new();
        let mut velocities = Vec::new();
        let mut diameters = Vec::new();
        for frame in map.values() {
            coordinates.extend_from_slice(&frame.coordinates);
            if frame.velocities.len() == frame.coordinates.len() {
                velocities.extend_from_slice(&frame.velocities);
            } else {
                velocities.resize(coordinates.len(), [0.0, 0.0, 0.0]);
            }
            diameters.extend_from_slice(&frame.diameters);
        }
        Ok(SampledCoordinates {
            coordinates,
            velocities,
            diameters,
        })
    }
}
