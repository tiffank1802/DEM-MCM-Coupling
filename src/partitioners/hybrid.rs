//! Hybrid and baseline spatial partitioners:
//! - [`AdaptivePartitioner`]
//! - [`MultiZonePartitioner`]
//! - [`SingleCellPartitioner`]

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::config::PartitioningMethod;
use crate::error::{DemMcmError, Result};
use crate::io::json::JsonValue;
use crate::partitioners::geometric::quantile_sorted;
use crate::partitioners::{
    create_partitioner, Partitioner, PartitionerArtifacts, PartitionerGeometry,
};

// ============================================================================
// 12. ADAPTIVE TOP/BOTTOM PARTITIONER
// ============================================================================

pub struct AdaptivePartitioner {
    pub y_split_input: Option<f64>,
    pub y_split_mode: String,
    pub top_method: String,
    pub top_kwargs: HashMap<String, JsonValue>,
    pub bottom_method: String,
    pub bottom_kwargs: HashMap<String, JsonValue>,

    pub y_split: Option<f64>,
    pub y_min: Option<f64>,
    pub y_max: Option<f64>,
    pub top_partitioner: Option<Box<dyn Partitioner>>,
    pub bottom_partitioner: Option<Box<dyn Partitioner>>,
    pub n_cells_top: Option<usize>,
    pub n_cells_bottom: Option<usize>,
}

impl AdaptivePartitioner {
    pub fn new(
        y_split: Option<f64>,
        y_split_mode: impl Into<String>,
        top_method: impl Into<String>,
        top_kwargs: HashMap<String, JsonValue>,
        bottom_method: impl Into<String>,
        bottom_kwargs: HashMap<String, JsonValue>,
    ) -> Self {
        Self {
            y_split_input: y_split,
            y_split_mode: y_split_mode.into(),
            top_method: top_method.into(),
            top_kwargs,
            bottom_method: bottom_method.into(),
            bottom_kwargs,
            y_split: None,
            y_min: None,
            y_max: None,
            top_partitioner: None,
            bottom_partitioner: None,
            n_cells_top: None,
            n_cells_bottom: None,
        }
    }
}

impl Partitioner for AdaptivePartitioner {
    fn method(&self) -> PartitioningMethod {
        PartitioningMethod::Adaptive
    }

    fn type_name(&self) -> &'static str {
        "AdaptivePartitioner"
    }

    fn n_cells(&self) -> usize {
        match (self.n_cells_top, self.n_cells_bottom) {
            (Some(t), Some(b)) => t + b,
            _ => 0,
        }
    }

    fn label(&self) -> String {
        let top_str = self
            .n_cells_top
            .map_or_else(|| "None".to_string(), |v| v.to_string());
        let bot_str = self
            .n_cells_bottom
            .map_or_else(|| "None".to_string(), |v| v.to_string());
        let split_str = self
            .y_split_input
            .map_or_else(|| "None".to_string(), |v| format!("{v}"));
        format!(
            "adaptive_y_{}_top{top_str}_bot{bot_str}_split{split_str}_mode{}",
            self.bottom_method, self.y_split_mode
        )
    }

    fn fit(&mut self, coordinates: &[[f64; 3]]) -> Result<()> {
        if coordinates.is_empty() {
            return Err(DemMcmError::InvalidArgument(
                "AdaptivePartitioner::fit requires non-empty coordinates".to_string(),
            ));
        }
        let mut ys: Vec<f64> = coordinates.iter().map(|c| c[1]).collect();
        ys.sort_by(|a, b| a.total_cmp(b));
        let y_min = ys[0];
        let y_max = *ys.last().unwrap();
        self.y_min = Some(y_min);
        self.y_max = Some(y_max);

        let split = match self.y_split_mode.as_str() {
            "quantile" => {
                let q = self.y_split_input.unwrap_or(0.9);
                quantile_sorted(&ys, q)
            }
            "absolute" => self.y_split_input.unwrap_or(0.5 * (y_min + y_max)),
            other => {
                return Err(DemMcmError::InvalidArgument(format!(
                    "Unknown y_split_mode: {other}"
                )))
            }
        };
        self.y_split = Some(split);

        let mut bottom_coords = Vec::new();
        let mut top_coords = Vec::new();
        for &c in coordinates {
            if c[1] <= split {
                bottom_coords.push(c);
            } else {
                top_coords.push(c);
            }
        }

        let mut bp = create_partitioner(&self.bottom_method, &self.bottom_kwargs)?;
        if !bottom_coords.is_empty() {
            bp.fit(&bottom_coords)?;
        }
        self.n_cells_bottom = Some(bp.n_cells());
        self.bottom_partitioner = Some(bp);

        if self.top_method == "single" {
            self.top_partitioner = None;
            self.n_cells_top = Some(1);
        } else {
            let mut tp = create_partitioner(&self.top_method, &self.top_kwargs)?;
            if !top_coords.is_empty() {
                tp.fit(&top_coords)?;
            }
            self.n_cells_top = Some(tp.n_cells());
            self.top_partitioner = Some(tp);
        }
        Ok(())
    }

    fn compute_states(
        &self,
        coordinates: &[[f64; 3]],
        _velocities: Option<&[[f64; 3]]>,
    ) -> Result<Vec<i64>> {
        let (split, bp, n_bottom) =
            match (self.y_split, &self.bottom_partitioner, self.n_cells_bottom) {
                (Some(s), Some(b), Some(nb)) => (s, b, nb),
                _ => {
                    return Err(DemMcmError::NotFitted(
                        "AdaptivePartitioner must be fitted before use".to_string(),
                    ))
                }
            };

        let mut bottom_idx = Vec::new();
        let mut bottom_coords = Vec::new();
        let mut top_idx = Vec::new();
        let mut top_coords = Vec::new();
        for (i, &c) in coordinates.iter().enumerate() {
            if c[1] <= split {
                bottom_idx.push(i);
                bottom_coords.push(c);
            } else {
                top_idx.push(i);
                top_coords.push(c);
            }
        }

        let mut states = vec![0i64; coordinates.len()];
        if !bottom_coords.is_empty() {
            let bs = bp.compute_states(&bottom_coords, None)?;
            for (k, &i) in bottom_idx.iter().enumerate() {
                states[i] = bs[k];
            }
        }
        if !top_coords.is_empty() {
            if let Some(tp) = &self.top_partitioner {
                let ts = tp.compute_states(&top_coords, None)?;
                for (k, &i) in top_idx.iter().enumerate() {
                    states[i] = ts[k] + n_bottom as i64;
                }
            } else {
                for &i in &top_idx {
                    states[i] = n_bottom as i64;
                }
            }
        }
        Ok(states)
    }

    fn save_artifacts(&self) -> Result<PartitionerArtifacts> {
        let mut art = PartitionerArtifacts::default();
        let mut obj = JsonValue::object();
        if let Some(s) = self.y_split {
            obj.insert("y_split", JsonValue::Number(s));
        }
        obj.insert("y_split_mode", JsonValue::String(self.y_split_mode.clone()));
        obj.insert("top_method", JsonValue::String(self.top_method.clone()));
        obj.insert(
            "bottom_method",
            JsonValue::String(self.bottom_method.clone()),
        );
        if let Some(nt) = self.n_cells_top {
            obj.insert("n_cells_top", JsonValue::Number(nt as f64));
        }
        if let Some(nb) = self.n_cells_bottom {
            obj.insert("n_cells_bottom", JsonValue::Number(nb as f64));
        }
        art.json_files.insert("adaptive_config".to_string(), obj);
        Ok(art)
    }

    fn geometry_hint(&self) -> PartitionerGeometry {
        if let (Some(ys), Some(bp)) = (self.y_split, &self.bottom_partitioner) {
            PartitionerGeometry::AdaptiveZones {
                y_split: ys,
                y_min: self.y_min.unwrap_or(ys - 0.1),
                y_max: self.y_max.unwrap_or(ys + 0.1),
                n_bottom: self.n_cells_bottom.unwrap_or(bp.n_cells()),
                bottom: Box::new(bp.geometry_hint()),
                top: self
                    .top_partitioner
                    .as_ref()
                    .map(|tp| Box::new(tp.geometry_hint())),
            }
        } else {
            PartitionerGeometry::PointClusters
        }
    }

    fn save(&self, dir: &Path) -> Result<()> {
        fs::create_dir_all(dir)?;
        let mut meta = JsonValue::object();
        meta.insert("type", JsonValue::String(self.type_name().to_string()));
        meta.insert("label", JsonValue::String(self.label()));
        meta.insert("n_cells", JsonValue::Number(self.n_cells() as f64));
        fs::write(dir.join("partitioner_meta.json"), meta.to_string_pretty())?;

        let art = self.save_artifacts()?;
        for (k, val) in art.json_files {
            fs::write(dir.join(format!("{k}.json")), val.to_string_pretty())?;
        }
        if let Some(bp) = &self.bottom_partitioner {
            bp.save(&dir.join("bottom"))?;
        }
        if let Some(tp) = &self.top_partitioner {
            tp.save(&dir.join("top"))?;
        }
        Ok(())
    }

    fn load(&mut self, dir: &Path) -> Result<()> {
        let s = fs::read_to_string(dir.join("adaptive_config.json"))?;
        let j = JsonValue::parse(&s)?;
        self.y_split = j.get("y_split").and_then(|v| v.as_f64());
        if let Some(m) = j.get("y_split_mode").and_then(|v| v.as_str()) {
            self.y_split_mode = m.to_string();
        }
        if let Some(tm) = j.get("top_method").and_then(|v| v.as_str()) {
            self.top_method = tm.to_string();
        }
        if let Some(bm) = j.get("bottom_method").and_then(|v| v.as_str()) {
            self.bottom_method = bm.to_string();
        }
        self.n_cells_top = j.get("n_cells_top").and_then(|v| v.as_usize());
        self.n_cells_bottom = j.get("n_cells_bottom").and_then(|v| v.as_usize());
        if dir.join("bottom").is_dir() {
            let mut bp = create_partitioner(&self.bottom_method, &self.bottom_kwargs)?;
            bp.load(&dir.join("bottom"))?;
            self.bottom_partitioner = Some(bp);
        }
        if dir.join("top").is_dir() && self.top_method != "single" {
            let mut tp = create_partitioner(&self.top_method, &self.top_kwargs)?;
            tp.load(&dir.join("top"))?;
            self.top_partitioner = Some(tp);
        }
        Ok(())
    }
}

// ============================================================================
// 13. MULTI-ZONE PARTITIONER
// ============================================================================

#[derive(Debug, Clone)]
pub struct ZoneSpec {
    pub y_min: Option<f64>,
    pub y_max: Option<f64>,
    pub method: String,
    pub kwargs: HashMap<String, JsonValue>,
}

impl ZoneSpec {
    pub fn from_json(val: &JsonValue) -> Option<Self> {
        let method = val
            .get("method")
            .and_then(|v| v.as_str())
            .unwrap_or("single")
            .to_string();
        let y_min = val.get("y_min").and_then(|v| v.as_f64());
        let y_max = val.get("y_max").and_then(|v| v.as_f64());
        let kwargs = val
            .get("kwargs")
            .and_then(|v| v.as_object())
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();
        Some(Self {
            y_min,
            y_max,
            method,
            kwargs,
        })
    }

    pub fn to_json(&self) -> JsonValue {
        let mut obj = JsonValue::object();
        if let Some(ymin) = self.y_min {
            obj.insert("y_min", JsonValue::Number(ymin));
        }
        if let Some(ymax) = self.y_max {
            obj.insert("y_max", JsonValue::Number(ymax));
        }
        obj.insert("method", JsonValue::String(self.method.clone()));
        let mut kw = JsonValue::object();
        for (k, v) in &self.kwargs {
            kw.insert(k, v.clone());
        }
        obj.insert("kwargs", kw);
        obj
    }
}

pub struct MultiZonePartitioner {
    pub zones_config: Vec<ZoneSpec>,
    pub y_mode: String,

    pub zones: Vec<(f64, f64, Box<dyn Partitioner>)>,
    pub cell_offsets: Vec<usize>,
    pub total_cells: usize,
}

impl MultiZonePartitioner {
    pub fn new(zones: Vec<ZoneSpec>, y_mode: impl Into<String>) -> Self {
        Self {
            zones_config: zones,
            y_mode: y_mode.into(),
            zones: Vec::new(),
            cell_offsets: Vec::new(),
            total_cells: 0,
        }
    }
}

impl Partitioner for MultiZonePartitioner {
    fn method(&self) -> PartitioningMethod {
        PartitioningMethod::Multizone
    }

    fn type_name(&self) -> &'static str {
        "MultiZonePartitioner"
    }

    fn n_cells(&self) -> usize {
        self.total_cells
    }

    fn label(&self) -> String {
        let methods: Vec<&str> = self
            .zones_config
            .iter()
            .map(|z| z.method.as_str())
            .collect();
        format!(
            "multizone_{}zones_{}",
            self.zones_config.len(),
            methods.join("_")
        )
    }

    fn fit(&mut self, coordinates: &[[f64; 3]]) -> Result<()> {
        if coordinates.is_empty() {
            return Err(DemMcmError::InvalidArgument(
                "MultiZonePartitioner::fit requires non-empty coordinates".to_string(),
            ));
        }
        if self.zones_config.is_empty() {
            return Err(DemMcmError::InvalidArgument(
                "MultiZonePartitioner requires at least one zone".to_string(),
            ));
        }

        let mut ys: Vec<f64> = coordinates.iter().map(|c| c[1]).collect();
        ys.sort_by(|a, b| a.total_cmp(b));
        let y_data_min = ys[0];
        let y_data_max = *ys.last().unwrap();

        self.zones.clear();
        self.cell_offsets = vec![0];
        let n_zones = self.zones_config.len();

        for (i, cfg) in self.zones_config.iter().enumerate() {
            let (y_min, y_max) = if self.y_mode == "quantile" {
                (
                    quantile_sorted(&ys, cfg.y_min.unwrap_or(0.0)),
                    quantile_sorted(&ys, cfg.y_max.unwrap_or(1.0)),
                )
            } else {
                (
                    cfg.y_min.unwrap_or(y_data_min),
                    cfg.y_max.unwrap_or(y_data_max),
                )
            };

            let zone_coords: Vec<[f64; 3]> = coordinates
                .iter()
                .copied()
                .filter(|c| {
                    if i == n_zones - 1 {
                        c[1] >= y_min && c[1] <= y_max
                    } else {
                        c[1] >= y_min && c[1] < y_max
                    }
                })
                .collect();

            let mut part = create_partitioner(&cfg.method, &cfg.kwargs)?;
            if !zone_coords.is_empty() {
                part.fit(&zone_coords)?;
            }
            let nc = part.n_cells();
            self.zones.push((y_min, y_max, part));
            let next_off = *self.cell_offsets.last().unwrap() + nc;
            self.cell_offsets.push(next_off);
        }

        self.total_cells = *self.cell_offsets.last().unwrap_or(&0);
        Ok(())
    }

    fn compute_states(
        &self,
        coordinates: &[[f64; 3]],
        _velocities: Option<&[[f64; 3]]>,
    ) -> Result<Vec<i64>> {
        let mut states = vec![0i64; coordinates.len()];
        let mut assigned = vec![false; coordinates.len()];
        let n_zones = self.zones.len();

        for (i, (y_min, y_max, part)) in self.zones.iter().enumerate() {
            let mut zone_idx = Vec::new();
            let mut zone_coords = Vec::new();
            for (p, &c) in coordinates.iter().enumerate() {
                if assigned[p] {
                    continue;
                }
                let in_zone = if i == n_zones - 1 {
                    c[1] >= *y_min && c[1] <= *y_max
                } else {
                    c[1] >= *y_min && c[1] < *y_max
                };
                if in_zone {
                    zone_idx.push(p);
                    zone_coords.push(c);
                    assigned[p] = true;
                }
            }
            if !zone_coords.is_empty() {
                let zs = part.compute_states(&zone_coords, None)?;
                let offset = self.cell_offsets[i] as i64;
                for (k, &p) in zone_idx.iter().enumerate() {
                    states[p] = zs[k] + offset;
                }
            }
        }
        Ok(states)
    }

    fn save_artifacts(&self) -> Result<PartitionerArtifacts> {
        let mut art = PartitionerArtifacts::default();
        let mut cfg = JsonValue::object();
        cfg.insert(
            "zones_config",
            JsonValue::Array(self.zones_config.iter().map(ZoneSpec::to_json).collect()),
        );
        cfg.insert("y_mode", JsonValue::String(self.y_mode.clone()));
        cfg.insert(
            "cell_offsets",
            JsonValue::Array(
                self.cell_offsets
                    .iter()
                    .map(|&o| JsonValue::Number(o as f64))
                    .collect(),
            ),
        );
        cfg.insert(
            "zones_bounds",
            JsonValue::Array(
                self.zones
                    .iter()
                    .map(|(ymin, ymax, _)| {
                        JsonValue::Array(vec![JsonValue::Number(*ymin), JsonValue::Number(*ymax)])
                    })
                    .collect(),
            ),
        );
        art.json_files.insert("multizone_config".to_string(), cfg);
        Ok(art)
    }

    fn geometry_hint(&self) -> PartitionerGeometry {
        if self.zones.is_empty() {
            return PartitionerGeometry::PointClusters;
        }
        let z = self
            .zones
            .iter()
            .enumerate()
            .map(|(i, (ymn, ymx, p))| {
                let off = *self.cell_offsets.get(i).unwrap_or(&0);
                (*ymn, *ymx, off, p.geometry_hint())
            })
            .collect();
        PartitionerGeometry::MultiZone { zones: z }
    }

    fn save(&self, dir: &Path) -> Result<()> {
        fs::create_dir_all(dir)?;
        let mut meta = JsonValue::object();
        meta.insert("type", JsonValue::String(self.type_name().to_string()));
        meta.insert("label", JsonValue::String(self.label()));
        meta.insert("n_cells", JsonValue::Number(self.n_cells() as f64));
        fs::write(dir.join("partitioner_meta.json"), meta.to_string_pretty())?;

        let art = self.save_artifacts()?;
        for (k, v) in art.json_files {
            fs::write(dir.join(format!("{k}.json")), v.to_string_pretty())?;
        }
        for (i, (_, _, part)) in self.zones.iter().enumerate() {
            part.save(&dir.join(format!("zone_{i}")))?;
        }
        Ok(())
    }

    fn load(&mut self, dir: &Path) -> Result<()> {
        let s = fs::read_to_string(dir.join("multizone_config.json"))?;
        let j = JsonValue::parse(&s)?;
        if let Some(arr) = j.get("zones_config").and_then(|v| v.as_array()) {
            self.zones_config = arr.iter().filter_map(ZoneSpec::from_json).collect();
        }
        if let Some(ym) = j.get("y_mode").and_then(|v| v.as_str()) {
            self.y_mode = ym.to_string();
        }
        if let Some(offs) = j.get("cell_offsets").and_then(|v| v.as_array()) {
            self.cell_offsets = offs.iter().filter_map(|v| v.as_usize()).collect();
            self.total_cells = *self.cell_offsets.last().unwrap_or(&0);
        }
        self.zones.clear();
        if let Some(bnds) = j.get("zones_bounds").and_then(|v| v.as_array()) {
            for (i, b) in bnds.iter().enumerate() {
                let pair = b.as_array().ok_or_else(|| {
                    DemMcmError::Serialization(
                        "Invalid zones_bounds in multizone_config".to_string(),
                    )
                })?;
                let y_min = pair.first().and_then(|v| v.as_f64()).unwrap_or(0.0);
                let y_max = pair.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0);
                let cfg = &self.zones_config[i];
                let mut part = create_partitioner(&cfg.method, &cfg.kwargs)?;
                part.load(&dir.join(format!("zone_{i}")))?;
                self.zones.push((y_min, y_max, part));
            }
        }
        Ok(())
    }
}

// ============================================================================
// 14. SINGLE CELL PARTITIONER
// ============================================================================

#[derive(Debug, Clone, Default)]
pub struct SingleCellPartitioner;

impl SingleCellPartitioner {
    pub fn new() -> Self {
        Self
    }
}

impl Partitioner for SingleCellPartitioner {
    fn method(&self) -> PartitioningMethod {
        PartitioningMethod::Single
    }

    fn type_name(&self) -> &'static str {
        "SingleCellPartitioner"
    }

    fn n_cells(&self) -> usize {
        1
    }

    fn label(&self) -> String {
        "single_cell".to_string()
    }

    fn fit(&mut self, _coordinates: &[[f64; 3]]) -> Result<()> {
        Ok(())
    }

    fn compute_states(
        &self,
        coordinates: &[[f64; 3]],
        _velocities: Option<&[[f64; 3]]>,
    ) -> Result<Vec<i64>> {
        Ok(vec![0i64; coordinates.len()])
    }

    fn geometry_hint(&self) -> PartitionerGeometry {
        PartitionerGeometry::SingleDomain
    }

    fn load(&mut self, _dir: &Path) -> Result<()> {
        Ok(())
    }
}
