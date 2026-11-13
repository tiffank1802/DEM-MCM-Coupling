//! General-purpose helpers for `dem_mcm_coupling`.

use std::collections::BTreeMap;
use std::path::Path;

use crate::data::{DataSource, DemFrame, InMemoryDataSource, LocalDataSource};
use crate::error::{DemMcmError, Result};
use crate::io::parquet::read_dem_parquet;

/// Load a `.parquet` or `.parquet.gz` file and index its rows by DEM timestep.
pub fn load_parquet_as_timestep_dict(path: impl AsRef<Path>) -> Result<BTreeMap<usize, DemFrame>> {
    let p = path.as_ref();
    if p.is_dir() {
        return LocalDataSource::new(p).read_timesteps(None);
    }
    read_dem_parquet(p, None)
}

/// Filter a flat state array to keep only particles matching `species_mask`.
///
/// When `states` contains multiple stacked timesteps (`k * n_particles`), the
/// boolean mask is tiled cleanly across the full length.
pub fn apply_species_mask(states: &[i64], species_mask: Option<&[bool]>) -> Vec<i64> {
    let Some(mask) = species_mask else {
        return states.to_vec();
    };
    if mask.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(states.len());
    for (i, &s) in states.iter().enumerate() {
        if mask[i % mask.len()] {
            out.push(s);
        }
    }
    out
}

/// Construct a [`DataSource`] from a URI (`memory://` or local filesystem path).
pub fn data_source_from_uri(uri: &str) -> Result<Box<dyn DataSource>> {
    if uri.starts_with("memory://") {
        Ok(Box::new(InMemoryDataSource::new()))
    } else if uri.starts_with("hf://") {
        Err(DemMcmError::DataSource(
            "Remote hf:// streaming is not built in pure-Rust offline mode; use a local path or memory://".to_string(),
        ))
    } else {
        Ok(Box::new(LocalDataSource::new(uri)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn species_mask_tiling() {
        let states = vec![0, 1, 2, 3, 4, 5, 6];
        let mask = vec![true, false, true];
        let filtered = apply_species_mask(&states, Some(&mask));
        assert_eq!(filtered, vec![0, 2, 3, 5, 6]);
    }
}
