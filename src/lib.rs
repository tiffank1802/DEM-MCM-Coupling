//! Pure-Rust implementation of the `dem_mcm_coupling` Discrete Element Method
//! (DEM) + Markov Chain Model (MCM) coupling engine.
//!
//! # Modules
//! - [`config`] — Core constants, method enums, trajectory & experiment types
//! - [`data`] — `DataSource` trait, `DemFrame`, `InMemoryDataSource`, `LocalDataSource`
//! - [`io`] — Zero-dependency JSON, NumPy `.npy`/`.npz`, DEFLATE/GZIP/Snappy/ZIP, CSV, and Apache Parquet codecs
//! - [`linalg`] — Dense matrix operations, Cholesky, Jacobi eigendecomposition, SVD, Hessenberg/Francis QR eigenvalues, polyfit
//! - [`spatial`] — Multi-dimensional k-d tree for exact 1-NN and radius queries
//! - [`ml`] — KMeans, MiniBatchKMeans, DBSCAN, GMM, SpectralClustering, SpectralBiclustering, LinearSVC
//! - [`partitioners`] — All 14 geometric, physics-aware, clustering, and hybrid spatial partitioners
//! - [`markov_core`] — Single-configuration `Markov` runner and propagation
//! - [`run_sweep`] — Homogeneous, no-species, and inhomogeneous sweep pipelines
//! - [`metrics`] — Mixing physics (RSD, entropy, Danckwerts segregation intensity, mixing times, stationary distribution, probability law fitting, physical validation)
//! - [`analyze_results`] — Experiment loading and DEM-vs-Markov evaluation

pub mod analyze_results;
pub mod config;
pub mod data;
pub mod error;
pub mod gui;
pub mod io;
pub mod linalg;
pub mod markov_core;
pub mod metrics;
pub mod ml;
pub mod partitioners;
pub mod research;
pub mod rng;
pub mod run_sweep;
pub mod spatial;
pub mod utils;

pub use config::{
    get_bucket_prefix, validate_partitioning_method, ExperimentData, MarkovTrajectory,
    PartitionerConfig, PartitioningMethod, SpeciesExperimentData, StateVectorData,
    N_PARTICLES_PER_TIMESTEP, PERMANENT_START, SIMULATION_PARQUET_NAME, TIMESTEP_TO_SECONDS,
};
pub use data::{DataSource, DemFrame, DemSnapshot, InMemoryDataSource, LocalDataSource};
pub use error::{DemMcmError, Result};
pub use linalg::Matrix;
pub use markov_core::Markov;
pub use partitioners::{create_partitioner, Partitioner};
pub use run_sweep::{
    get_configs, run_experiment, run_inhomogeneous_experiment, run_no_species_experiment,
    run_sweep, ExperimentConfig, SweepMode,
};
