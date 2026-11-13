//! Error types for the `dem_mcm_coupling` crate.

use std::fmt;

/// Convenience alias for `std::result::Result<T, DemMcmError>`.
pub type Result<T> = std::result::Result<T, DemMcmError>;

/// Unified error type returned across the library.
#[derive(Debug)]
pub enum DemMcmError {
    /// Unknown or unsupported partitioning method identifier.
    InvalidMethod(String),
    /// Invalid parameter or configuration value.
    InvalidArgument(String),
    /// Operation attempted before a partitioner or model was fitted.
    NotFitted(String),
    /// Requested timestep, column, or species was not present.
    MissingData(String),
    /// Incompatible array dimensions or particle counts.
    ShapeMismatch(String),
    /// Failure while reading from or writing to a [`crate::data::DataSource`].
    DataSource(String),
    /// Failure while parsing or serialising `.npy`, `.npz`, `.json`, `.csv`, or `.parquet`.
    Serialization(String),
    /// Underlying filesystem or stream I/O error.
    Io(std::io::Error),
}

impl fmt::Display for DemMcmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMethod(msg) => write!(f, "Invalid method: {msg}"),
            Self::InvalidArgument(msg) => write!(f, "Invalid argument: {msg}"),
            Self::NotFitted(msg) => write!(f, "Not fitted: {msg}"),
            Self::MissingData(msg) => write!(f, "Missing data: {msg}"),
            Self::ShapeMismatch(msg) => write!(f, "Shape mismatch: {msg}"),
            Self::DataSource(msg) => write!(f, "Data source error: {msg}"),
            Self::Serialization(msg) => write!(f, "Serialization error: {msg}"),
            Self::Io(err) => write!(f, "I/O error: {err}"),
        }
    }
}

impl std::error::Error for DemMcmError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for DemMcmError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}
