//! Local-filesystem implementation of [`DataSource`].
//!
//! Reads DEM simulation data from:
//! - `<root>/<parquet_name>` (e.g. `simulation_complete.parquet` or `.parquet.gz`)
//! - `<root>/*.parquet.gz` / `<root>/*.parquet` (chunked directories like `data/chunks/`)
//! - `<root>/data_*.csv` / `<root>/file_*.csv` (CSV snapshots)
//!
//! And persists/restores Markov experiments using standard `stats.json`,
//! `config.json`, `inhomogeneous_metadata.json`, `.npy` arrays, and `partitioner/`
//! sub-directories (binary-compatible with Python `LocalDataSource`).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::config::{ExperimentData, SpeciesExperimentData, SIMULATION_PARQUET_NAME};
use crate::data::{DataSource, DemFrame, ExperimentWritePayload};
use crate::error::{DemMcmError, Result};
use crate::io::csv::read_dem_csv;
use crate::io::json::JsonValue;
use crate::io::npy::NpyArray;
use crate::io::parquet::read_dem_parquet;

/// Local-directory data source.
#[derive(Debug, Clone)]
pub struct LocalDataSource {
    pub root: PathBuf,
    pub parquet_name: String,
}

impl LocalDataSource {
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
            parquet_name: SIMULATION_PARQUET_NAME.to_string(),
        }
    }

    pub fn with_parquet_name(mut self, name: impl Into<String>) -> Self {
        self.parquet_name = name.into();
        self
    }

    pub fn parquet_path(&self) -> PathBuf {
        self.root.join(&self.parquet_name)
    }

    fn read_json(path: &Path) -> Result<JsonValue> {
        let s = fs::read_to_string(path)?;
        JsonValue::parse(&s)
    }

    fn write_json(path: &Path, val: &JsonValue) -> Result<()> {
        fs::write(path, val.to_string_pretty())?;
        Ok(())
    }
}

impl DataSource for LocalDataSource {
    fn read_timesteps(
        &self,
        timestep_indices: Option<&[usize]>,
    ) -> Result<BTreeMap<usize, DemFrame>> {
        let filter: Option<HashSet<usize>> = timestep_indices.map(|s| s.iter().copied().collect());

        let pq_path = self.parquet_path();
        if pq_path.is_file() {
            return read_dem_parquet(&pq_path, filter.as_ref());
        }

        // Check if root itself is a parquet file path
        if self.root.is_file() {
            let ext = self.root.extension().and_then(|e| e.to_str()).unwrap_or("");
            if ext.eq_ignore_ascii_case("csv") {
                let mut all = read_dem_csv(&self.root, None)?;
                if let Some(f) = filter {
                    all.retain(|k, _| f.contains(k));
                }
                return Ok(all);
            }
            return read_dem_parquet(&self.root, filter.as_ref());
        }

        if !self.root.is_dir() {
            return Err(DemMcmError::DataSource(format!(
                "Parquet file or directory not found: {}",
                pq_path.display()
            )));
        }

        // Look for parquet / parquet.gz chunks inside root
        let mut parquet_files: Vec<PathBuf> = Vec::new();
        let mut csv_files: Vec<PathBuf> = Vec::new();
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            let p = entry.path();
            if p.is_file() {
                let fname = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if fname.ends_with(".parquet") || fname.ends_with(".parquet.gz") {
                    parquet_files.push(p);
                } else if fname.ends_with(".csv") {
                    csv_files.push(p);
                }
            }
        }
        parquet_files.sort();
        csv_files.sort();

        if !parquet_files.is_empty() {
            let n_workers = std::thread::available_parallelism()
                .map(|n| n.get().min(parquet_files.len()))
                .unwrap_or(1)
                .max(1);
            if n_workers == 1 || parquet_files.len() == 1 {
                let mut combined = BTreeMap::new();
                for p in parquet_files {
                    let chunk_map = read_dem_parquet(&p, filter.as_ref())?;
                    for (ts, frame) in chunk_map {
                        combined.insert(ts, frame);
                    }
                }
                return Ok(combined);
            }
            let next_idx = std::sync::atomic::AtomicUsize::new(0);
            let files_ref = &parquet_files;
            let filter_ref = filter.as_ref();
            let results: std::sync::Mutex<Result<Vec<BTreeMap<usize, DemFrame>>>> =
                std::sync::Mutex::new(Ok(Vec::with_capacity(parquet_files.len())));
            std::thread::scope(|s| {
                for _ in 0..n_workers {
                    s.spawn(|| loop {
                        let idx = next_idx.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if idx >= files_ref.len() {
                            break;
                        }
                        let res = read_dem_parquet(&files_ref[idx], filter_ref);
                        let mut guard = results.lock().unwrap();
                        if let Ok(ref mut vec) = *guard {
                            match res {
                                Ok(chunk_map) => vec.push(chunk_map),
                                Err(e) => {
                                    *guard = Err(e);
                                    break;
                                }
                            }
                        } else {
                            break;
                        }
                    });
                }
            });
            let chunks = results.into_inner().unwrap()?;
            let mut combined = BTreeMap::new();
            for chunk_map in chunks {
                for (ts, frame) in chunk_map {
                    combined.insert(ts, frame);
                }
            }
            return Ok(combined);
        }

        if !csv_files.is_empty() {
            let mut combined = BTreeMap::new();
            for p in csv_files {
                let part = read_dem_csv(&p, None)?;
                for (ts, frame) in part {
                    if filter.as_ref().map_or(true, |f| f.contains(&ts)) {
                        combined.insert(ts, frame);
                    }
                }
            }
            return Ok(combined);
        }

        Err(DemMcmError::DataSource(format!(
            "No parquet or CSV DEM data found in {}",
            self.root.display()
        )))
    }

    fn list_experiments(&self, _prefix: Option<&str>) -> Result<Vec<String>> {
        if !self.root.is_dir() {
            return Ok(Vec::new());
        }
        let mut names = Vec::new();
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            if entry.path().is_dir() {
                if let Some(name) = entry.file_name().to_str() {
                    names.push(name.to_string());
                }
            }
        }
        names.sort();
        Ok(names)
    }

    fn read_experiment(&self, folder_name: &str, _prefix: Option<&str>) -> Result<ExperimentData> {
        let folder = self.root.join(folder_name);
        if !folder.is_dir() {
            return Err(DemMcmError::DataSource(format!(
                "Experiment folder not found: {}",
                folder.display()
            )));
        }

        let stats = Self::read_json(&folder.join("stats.json"))?;
        let config = Self::read_json(&folder.join("config.json"))?;
        let inh_path = folder.join("inhomogeneous_metadata.json");
        let inhomogeneous = inh_path.exists();
        let inhomogeneous_metadata = if inhomogeneous {
            Some(Self::read_json(&inh_path)?)
        } else {
            None
        };

        let species_list: Vec<String> = stats
            .get("species_list")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_else(|| vec!["small".to_string(), "large".to_string()]);

        let mut species = HashMap::new();
        for sp in species_list {
            let s_matrix =
                NpyArray::load(&folder.join(format!("S_matrix_{sp}.npy")))?.to_matrix()?;
            let times = NpyArray::load(&folder.join(format!("times_{sp}.npy")))?.to_i64_vec()?;
            if inhomogeneous {
                let p_blocks =
                    NpyArray::load(&folder.join(format!("P_blocks_{sp}.npy")))?.to_matrix_3d()?;
                species.insert(
                    sp,
                    SpeciesExperimentData {
                        p: None,
                        p_blocks: Some(p_blocks),
                        s_matrix,
                        times,
                    },
                );
            } else {
                let p = NpyArray::load(&folder.join(format!("transitionmatrix_{sp}.npy")))?
                    .to_matrix()?;
                species.insert(
                    sp,
                    SpeciesExperimentData {
                        p: Some(p),
                        p_blocks: None,
                        s_matrix,
                        times,
                    },
                );
            }
        }

        let matrix_path = folder.join("states_matrix.npy");
        let states_matrix = if matrix_path.exists() {
            Some(NpyArray::load(&matrix_path)?.to_i64_2d()?)
        } else {
            None
        };

        Ok(ExperimentData {
            config,
            stats,
            species,
            states_matrix,
            inhomogeneous,
            inhomogeneous_metadata,
        })
    }

    fn write_experiment(&mut self, payload: ExperimentWritePayload<'_>) -> Result<()> {
        let folder = self.root.join(payload.folder_name);
        fs::create_dir_all(&folder)?;

        Self::write_json(&folder.join("stats.json"), payload.stats)?;
        Self::write_json(&folder.join("config.json"), payload.config)?;

        for (name, arr) in payload.species_arrays {
            arr.save(&folder.join(format!("{name}.npy")))?;
        }

        if !payload.partitioner_arrays.is_empty() || !payload.partitioner_json.is_empty() {
            let part_dir = folder.join("partitioner");
            fs::create_dir_all(&part_dir)?;
            for (k, arr) in payload.partitioner_arrays {
                arr.save(&part_dir.join(format!("{k}.npy")))?;
            }
            for (k, val) in payload.partitioner_json {
                Self::write_json(&part_dir.join(format!("{k}.json")), &val)?;
            }
        }

        if !payload.image_data.is_empty() {
            let img_dir = folder.join("images");
            fs::create_dir_all(&img_dir)?;
            for (k, bytes) in payload.image_data {
                fs::write(img_dir.join(k), bytes)?;
            }
        }

        if let Some(meta) = payload.inhomogeneous_metadata {
            Self::write_json(&folder.join("inhomogeneous_metadata.json"), meta)?;
        }

        Ok(())
    }
}
