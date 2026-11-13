//! Pure-Rust I/O codecs for DEM/Markov persistence and data ingestion.
//!
//! Includes zero-dependency readers and writers for:
//! - JSON (`config.json`, `stats.json`, `inhomogeneous_metadata.json`, partitioner metadata)
//! - NumPy `.npy` & `.npz` (binary compatible with `numpy.save` / `numpy.load`)
//! - RFC 1951 DEFLATE / RFC 1952 GZIP & Snappy decompression
//! - DEM CSV files (`data_<idx>.csv` / combined CSV)
//! - Apache Parquet (`.parquet` and `.parquet.gz` simulation chunks)

pub mod csv;
pub mod deflate;
pub mod json;
pub mod npy;
pub mod parquet;
