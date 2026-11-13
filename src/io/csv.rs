//! Pure-Rust DEM CSV reader (`data_<idx>.csv` or combined multi-timestep CSV).

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use crate::data::DemFrame;
use crate::error::{DemMcmError, Result};

/// Parse timestep index from a `Fichier_Source` or filename such as `"data_42.csv"`.
pub fn parse_timestep_from_source(source: &str) -> Result<usize> {
    let trimmed = source.trim().trim_matches('"');
    let base = trimmed.rsplit('/').next().unwrap_or(trimmed);
    let stem = base
        .strip_suffix(".csv")
        .or_else(|| base.strip_suffix(".CSV"))
        .unwrap_or(base);
    let num_str = stem
        .strip_prefix("data_")
        .or_else(|| stem.strip_prefix("file_"))
        .unwrap_or(stem);
    num_str.parse::<usize>().map_err(|e| {
        DemMcmError::Serialization(format!(
            "Cannot parse timestep index from source {source:?}: {e}"
        ))
    })
}

/// Load a DEM CSV file.
///
/// If the CSV has a `Fichier_Source` or `TimeStep` column, rows are grouped
/// by timestep; otherwise `fallback_timestep` is used (or parsed from the file
/// name).
pub fn read_dem_csv(
    path: &Path,
    fallback_timestep: Option<usize>,
) -> Result<BTreeMap<usize, DemFrame>> {
    let content = fs::read_to_string(path)?;
    read_dem_csv_str(
        &content,
        fallback_timestep.or_else(|| {
            path.file_name()
                .and_then(|n| n.to_str())
                .and_then(|s| parse_timestep_from_source(s).ok())
        }),
    )
}

/// Parse DEM CSV text into a sorted map `timestep_index -> DemFrame`.
pub fn read_dem_csv_str(
    content: &str,
    fallback_timestep: Option<usize>,
) -> Result<BTreeMap<usize, DemFrame>> {
    let mut lines = content.lines().filter(|l| !l.trim().is_empty());
    let header_line = lines
        .next()
        .ok_or_else(|| DemMcmError::Serialization("Empty CSV file".to_string()))?;
    let sep = if header_line.contains(',') {
        ','
    } else if header_line.contains(';') {
        ';'
    } else {
        ','
    };

    let headers: Vec<String> = header_line
        .split(sep)
        .map(|h| h.trim().trim_matches('"').to_string())
        .collect();
    let find_col = |name: &str| headers.iter().position(|h| h == name);

    let c_x = find_col("coordinates:0")
        .ok_or_else(|| DemMcmError::MissingData("CSV lacks column 'coordinates:0'".to_string()))?;
    let c_y = find_col("coordinates:1")
        .ok_or_else(|| DemMcmError::MissingData("CSV lacks column 'coordinates:1'".to_string()))?;
    let c_z = find_col("coordinates:2")
        .ok_or_else(|| DemMcmError::MissingData("CSV lacks column 'coordinates:2'".to_string()))?;
    let c_vx = find_col("Velocity:0");
    let c_vy = find_col("Velocity:1");
    let c_vz = find_col("Velocity:2");
    let c_diam = find_col("Diameter");
    let c_pid = find_col("Particle_ID");
    let c_src = find_col("Fichier_Source");
    let c_ts = find_col("TimeStep");
    let c_time = find_col("Time");
    let c_res = find_col("Residence_Time");

    let mut frames: BTreeMap<usize, DemFrame> = BTreeMap::new();

    for (row_no, line) in lines.enumerate() {
        let fields: Vec<&str> = line
            .split(sep)
            .map(|s| s.trim().trim_matches('"'))
            .collect();
        if fields.len() < headers.len() {
            return Err(DemMcmError::Serialization(format!(
                "CSV row {} has {} fields, expected {}",
                row_no + 2,
                fields.len(),
                headers.len()
            )));
        }

        let ts = if let Some(idx) = c_src {
            parse_timestep_from_source(fields[idx])?
        } else if let Some(idx) = c_ts {
            fields[idx].parse::<usize>().map_err(|e| {
                DemMcmError::Serialization(format!("Invalid TimeStep {:?}: {e}", fields[idx]))
            })?
        } else if let Some(fb) = fallback_timestep {
            fb
        } else {
            return Err(DemMcmError::MissingData(
                "CSV lacks 'Fichier_Source'/'TimeStep' and no fallback timestep given".to_string(),
            ));
        };

        let parse_f64 = |idx: usize| -> Result<f64> {
            fields[idx].parse::<f64>().map_err(|e| {
                DemMcmError::Serialization(format!(
                    "Invalid float {:?} at row {}: {e}",
                    fields[idx],
                    row_no + 2
                ))
            })
        };

        let pos = [parse_f64(c_x)?, parse_f64(c_y)?, parse_f64(c_z)?];
        let vel = match (c_vx, c_vy, c_vz) {
            (Some(ix), Some(iy), Some(iz)) => {
                Some([parse_f64(ix)?, parse_f64(iy)?, parse_f64(iz)?])
            }
            _ => None,
        };
        let diam = c_diam.map(parse_f64).transpose()?;
        let pid = c_pid
            .map(|idx| {
                fields[idx]
                    .parse::<i64>()
                    .map_err(|e| DemMcmError::Serialization(format!("Invalid Particle_ID: {e}")))
            })
            .transpose()?;
        let time_val = c_time.map(parse_f64).transpose()?;
        let res_val = c_res.map(parse_f64).transpose()?;

        let frame = frames.entry(ts).or_insert_with(|| DemFrame::new(ts));
        frame.coordinates.push(pos);
        if let Some(v) = vel {
            frame.velocities.push(v);
        }
        if let Some(d) = diam {
            frame.diameters.push(d);
        }
        if let Some(p) = pid {
            frame.particle_ids.push(p);
        }
        if let Some(t) = time_val {
            frame.time = Some(t);
        }
        if let Some(rt) = res_val {
            frame.residence_times.push(rt);
        }
    }

    Ok(frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_csv_grouped_by_fichier_source() {
        let csv = "\
coordinates:0,coordinates:1,coordinates:2,Velocity:0,Velocity:1,Velocity:2,Diameter,Particle_ID,Fichier_Source
0.01,0.02,0.03,0.1,0.2,0.3,0.004,0,data_250.csv
-0.01,-0.02,0.04,-0.1,0.0,0.2,0.008,1,data_250.csv
0.02,0.01,0.03,0.2,0.1,0.3,0.004,0,data_251.csv
-0.02,-0.01,0.04,0.0,0.1,-0.2,0.008,1,data_251.csv
";
        let map = read_dem_csv_str(csv, None).unwrap();
        assert_eq!(map.len(), 2);
        assert_eq!(map[&250].n_particles(), 2);
        assert_eq!(map[&250].diameters, vec![0.004, 0.008]);
        assert_eq!(map[&251].coordinates[0], [0.02, 0.01, 0.03]);
    }
}
