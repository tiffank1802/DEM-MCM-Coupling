//! Core Markov chain model for DEM/MCM coupling (`dem_mcm_coupling/markov_core.py`).
//!
//! [`Markov`] orchestrates a single partitioning configuration:
//! 1. creation of a [`Partitioner`](crate::partitioners::Partitioner);
//! 2. loading of DEM particle frames from any [`DataSource`];
//! 3. fitting the partitioner on stacked timestep coordinates;
//! 4. building the initial state vector `phi(0)`;
//! 5. propagating `phi(t + 1) = phi(t) @ M`.

use std::collections::BTreeMap;
use std::collections::HashMap;

use crate::config::{
    validate_partitioning_method, MarkovTrajectory, PartitioningMethod, StateVectorData,
    TIMESTEP_TO_SECONDS,
};
use crate::data::{DataSource, DemFrame};
use crate::error::{DemMcmError, Result};
use crate::io::json::JsonValue;
use crate::linalg::Matrix;
use crate::partitioners::{create_partitioner_by_method, Partitioner};
use crate::utils::apply_species_mask;

/// Manager of one Markovian partitioning configuration.
pub struct Markov {
    pub method: PartitioningMethod,
    pub partitioner: Box<dyn Partitioner>,
    pub datas: BTreeMap<usize, DemFrame>,
    pub coords: Vec<[f64; 3]>,
    pub velocities: Vec<[f64; 3]>,
    pub states: Vec<i64>,
    pub initial_state: Option<StateVectorData>,
    pub transition_matrix: Option<Matrix>,
}

impl std::fmt::Debug for Markov {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Markov(method={:?}, n_cells={}, datas={}, coords_len={})",
            self.method.as_str(),
            self.partitioner.n_cells(),
            self.datas.len(),
            self.coords.len()
        )
    }
}

impl Markov {
    /// Initialise a [`Markov`] instance for a partitioning method and kwargs.
    pub fn new(method: &str, method_kwargs: Option<&HashMap<String, JsonValue>>) -> Result<Self> {
        let pm = validate_partitioning_method(method)?;
        let empty = HashMap::new();
        let kw = method_kwargs.unwrap_or(&empty);
        let partitioner = create_partitioner_by_method(pm, kw)?;
        Ok(Self {
            method: pm,
            partitioner,
            datas: BTreeMap::new(),
            coords: Vec::new(),
            velocities: Vec::new(),
            states: Vec::new(),
            initial_state: None,
            transition_matrix: None,
        })
    }

    /// Load DEM particle data into memory from a [`DataSource`].
    pub fn load_dem_data(&mut self, source: &dyn DataSource) -> Result<&BTreeMap<usize, DemFrame>> {
        self.datas = source.read_timesteps(None)?;
        Ok(&self.datas)
    }

    /// Directly populate DEM frames (for in-memory construction).
    pub fn set_dem_data(&mut self, datas: BTreeMap<usize, DemFrame>) {
        self.datas = datas;
    }

    /// Stack coordinates `(n_particles, 3)` across the requested timestep indices.
    pub fn get_coords(&mut self, timestep_indices: &[usize]) -> Result<&[[f64; 3]]> {
        if self.datas.is_empty() {
            return Err(DemMcmError::MissingData(
                "No DEM data loaded; call load_dem_data() first".to_string(),
            ));
        }
        let mut out = Vec::new();
        for &idx in timestep_indices {
            let frame = self.datas.get(&idx).ok_or_else(|| {
                let avail: Vec<usize> = self.datas.keys().take(10).copied().collect();
                DemMcmError::MissingData(format!(
                    "Timestep {idx} unavailable. Available sample: {avail:?}"
                ))
            })?;
            out.extend_from_slice(&frame.coordinates);
        }
        self.coords = out;
        Ok(&self.coords)
    }

    /// Stack velocities `(n_particles, 3)` across the requested timestep indices.
    pub fn get_velocities(&mut self, timestep_indices: &[usize]) -> Result<&[[f64; 3]]> {
        if self.datas.is_empty() {
            return Err(DemMcmError::MissingData(
                "No DEM data loaded; call load_dem_data() first".to_string(),
            ));
        }
        let mut out = Vec::new();
        for &idx in timestep_indices {
            let frame = self
                .datas
                .get(&idx)
                .ok_or_else(|| DemMcmError::MissingData(format!("Timestep {idx} unavailable")))?;
            if frame.velocities.is_empty() {
                out.resize(out.len() + frame.coordinates.len(), [0.0, 0.0, 0.0]);
            } else {
                out.extend_from_slice(&frame.velocities);
            }
        }
        self.velocities = out;
        Ok(&self.velocities)
    }

    /// Fit the partitioner on particle coordinates.
    pub fn fit_partitioner(&mut self, coordinates: &[[f64; 3]]) -> Result<()> {
        self.partitioner.fit(coordinates)
    }

    /// Assign each particle to its partition state index.
    pub fn compute_states(&mut self, coordinates: Option<&[[f64; 3]]>) -> Result<&[i64]> {
        let coords = coordinates.unwrap_or(&self.coords);
        if coords.is_empty() {
            return Err(DemMcmError::InvalidArgument(
                "Empty coordinates".to_string(),
            ));
        }
        let vel = if !self.velocities.is_empty() && self.velocities.len() == coords.len() {
            Some(self.velocities.as_slice())
        } else {
            None
        };
        self.states = self.partitioner.compute_states(coords, vel)?;
        Ok(&self.states)
    }

    /// Build the initial state vector `phi(0)` at `timestep_index`.
    pub fn build_initial_state_vector(
        &mut self,
        timestep_index: usize,
        species_mask: Option<&[bool]>,
        normalize: bool,
    ) -> Result<StateVectorData> {
        let n_cells = self.partitioner.n_cells();
        if n_cells == 0 {
            return Err(DemMcmError::NotFitted(
                "Partitioner not fitted. Call fit_partitioner() first.".to_string(),
            ));
        }
        let coords = self.get_coords(&[timestep_index])?.to_vec();
        let states_full = self.compute_states(Some(&coords))?.to_vec();

        let (states, total) = if let Some(mask) = species_mask {
            let filtered = apply_species_mask(&states_full, Some(mask));
            let count = mask.iter().filter(|&&b| b).count();
            (filtered, count)
        } else {
            let count = states_full.len();
            (states_full, count)
        };

        let mut phi = vec![0.0f64; n_cells];
        for &s in &states {
            if s >= 0 && (s as usize) < n_cells {
                phi[s as usize] += 1.0;
            }
        }

        let state = StateVectorData {
            phi,
            timestamp: timestep_index,
            total_particles: total,
            description: format!("Initial state at t={timestep_index}"),
        };

        if normalize && !state.validate_normalization(1e-5) {
            let sum: f64 = state.phi.iter().sum();
            return Err(DemMcmError::InvalidArgument(format!(
                "State not normalised: sum(phi)={sum}, expected={total}"
            )));
        }

        self.initial_state = Some(state.clone());
        Ok(state)
    }

    /// Propagate a state vector through a row-stochastic transition matrix `M`.
    ///
    /// Computes `phi(t + 1) = phi(t) @ M`.
    pub fn propagate_markov(
        &self,
        initial_state: &[f64],
        transition_matrix: &Matrix,
        n_steps: usize,
    ) -> Result<MarkovTrajectory> {
        let n_states = transition_matrix.rows;
        if transition_matrix.rows != transition_matrix.cols {
            return Err(DemMcmError::ShapeMismatch(
                "Transition matrix must be square".to_string(),
            ));
        }
        if initial_state.len() != n_states {
            return Err(DemMcmError::ShapeMismatch(format!(
                "Dimension mismatch: initial_state={}, M={}x{}",
                initial_state.len(),
                n_states,
                n_states
            )));
        }

        let mut traj = Matrix::zeros(n_steps + 1, n_states);
        traj.set_row(0, initial_state);
        let mut curr = initial_state.to_vec();
        for t in 1..=n_steps {
            curr = transition_matrix.vec_mul(&curr)?;
            traj.set_row(t, &curr);
        }

        let times: Vec<usize> = (0..=n_steps).collect();
        let times_seconds: Vec<f64> = times
            .iter()
            .map(|&t| t as f64 * TIMESTEP_TO_SECONDS)
            .collect();

        Ok(MarkovTrajectory {
            states: traj,
            times,
            times_seconds,
            method: self.method.as_str().to_string(),
            description: format!("Markov propagation, {n_steps} steps"),
        })
    }
}
