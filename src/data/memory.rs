//! In-memory implementation of [`DataSource`].

use std::collections::{BTreeMap, HashMap};

use crate::config::{ExperimentData, SpeciesExperimentData};
use crate::data::{DataSource, DemFrame, ExperimentWritePayload};
use crate::error::{DemMcmError, Result};
use crate::io::json::JsonValue;

type PartitionerBlob = (HashMap<String, Vec<u8>>, HashMap<String, JsonValue>);

/// DEM/Markov data held in memory for tests, embedding, and prototyping.
#[derive(Debug, Default, Clone)]
pub struct InMemoryDataSource {
    timesteps: BTreeMap<usize, DemFrame>,
    experiments: BTreeMap<String, ExperimentData>,
    partitioner_blobs: BTreeMap<String, PartitionerBlob>,
    image_blobs: BTreeMap<String, HashMap<String, Vec<u8>>>,
}

impl InMemoryDataSource {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_timesteps(timesteps: BTreeMap<usize, DemFrame>) -> Self {
        Self {
            timesteps,
            ..Self::default()
        }
    }

    pub fn add_timestep(&mut self, frame: DemFrame) {
        self.timesteps.insert(frame.timestep, frame);
    }

    pub fn add_experiment(&mut self, folder_name: impl Into<String>, exp: ExperimentData) {
        self.experiments.insert(folder_name.into(), exp);
    }

    pub fn get_images(&self, folder_name: &str) -> Option<&HashMap<String, Vec<u8>>> {
        self.image_blobs.get(folder_name)
    }
}

impl DataSource for InMemoryDataSource {
    fn read_timesteps(
        &self,
        timestep_indices: Option<&[usize]>,
    ) -> Result<BTreeMap<usize, DemFrame>> {
        match timestep_indices {
            None => Ok(self.timesteps.clone()),
            Some(indices) => {
                let mut out = BTreeMap::new();
                for &idx in indices {
                    if let Some(frame) = self.timesteps.get(&idx) {
                        out.insert(idx, frame.clone());
                    }
                }
                Ok(out)
            }
        }
    }

    fn list_experiments(&self, _prefix: Option<&str>) -> Result<Vec<String>> {
        Ok(self.experiments.keys().cloned().collect())
    }

    fn read_experiment(&self, folder_name: &str, _prefix: Option<&str>) -> Result<ExperimentData> {
        self.experiments
            .get(folder_name)
            .cloned()
            .ok_or_else(|| DemMcmError::DataSource(format!("Unknown experiment: {folder_name:?}")))
    }

    fn write_experiment(&mut self, payload: ExperimentWritePayload<'_>) -> Result<()> {
        let inhomogeneous = payload.inhomogeneous_metadata.is_some();
        let species_list: Vec<String> = payload
            .stats
            .get("species_list")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();

        let mut species_map = HashMap::new();
        for sp in species_list {
            let s_key = format!("S_matrix_{sp}");
            let t_key = format!("times_{sp}");
            let s_matrix = payload
                .species_arrays
                .get(&s_key)
                .map(|a| a.to_matrix())
                .transpose()?
                .unwrap_or_else(|| crate::linalg::Matrix::zeros(0, 0));
            let times = payload
                .species_arrays
                .get(&t_key)
                .map(|a| a.to_i64_vec())
                .transpose()?
                .unwrap_or_default();

            if inhomogeneous {
                let p_blocks = payload
                    .species_arrays
                    .get(&format!("P_blocks_{sp}"))
                    .map(|a| a.to_matrix_3d())
                    .transpose()?;
                species_map.insert(
                    sp,
                    SpeciesExperimentData {
                        p: None,
                        p_blocks,
                        s_matrix,
                        times,
                    },
                );
            } else {
                let p = payload
                    .species_arrays
                    .get(&format!("transitionmatrix_{sp}"))
                    .map(|a| a.to_matrix())
                    .transpose()?;
                species_map.insert(
                    sp,
                    SpeciesExperimentData {
                        p,
                        p_blocks: None,
                        s_matrix,
                        times,
                    },
                );
            }
        }

        let states_matrix = payload
            .species_arrays
            .get("states_matrix")
            .map(|a| a.to_i64_2d())
            .transpose()?;

        let exp = ExperimentData {
            config: payload.config.clone(),
            stats: payload.stats.clone(),
            species: species_map,
            states_matrix,
            inhomogeneous,
            inhomogeneous_metadata: payload.inhomogeneous_metadata.cloned(),
        };

        if !payload.partitioner_arrays.is_empty() || !payload.partitioner_json.is_empty() {
            let mut arr_bytes = HashMap::new();
            for (k, v) in payload.partitioner_arrays {
                arr_bytes.insert(k, v.to_bytes());
            }
            self.partitioner_blobs.insert(
                payload.folder_name.to_string(),
                (arr_bytes, payload.partitioner_json),
            );
        }
        if !payload.image_data.is_empty() {
            self.image_blobs
                .insert(payload.folder_name.to_string(), payload.image_data);
        }
        self.experiments
            .insert(payload.folder_name.to_string(), exp);
        Ok(())
    }
}
