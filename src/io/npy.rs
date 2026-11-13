//! Pure-Rust NumPy `.npy` (v1.0 / v2.0) and `.npz` reader/writer.
//!
//! Files written by this module are binary-compatible with `numpy.load()`,
//! and files written by `numpy.save()` / `numpy.savez()` are directly readable.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::error::{DemMcmError, Result};
use crate::io::deflate::{read_zip_entries, write_zip_stored};
use crate::linalg::Matrix;

const MAGIC: &[u8; 6] = b"\x93NUMPY";

#[derive(Debug, Clone, PartialEq)]
pub enum NpyData {
    F64(Vec<f64>),
    F32(Vec<f32>),
    I64(Vec<i64>),
    I32(Vec<i32>),
    Bool(Vec<bool>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct NpyArray {
    pub shape: Vec<usize>,
    pub data: NpyData,
}

impl NpyArray {
    pub fn from_f64_vec(data: Vec<f64>) -> Self {
        let len = data.len();
        Self {
            shape: vec![len],
            data: NpyData::F64(data),
        }
    }

    pub fn from_i64_vec(data: Vec<i64>) -> Self {
        let len = data.len();
        Self {
            shape: vec![len],
            data: NpyData::I64(data),
        }
    }

    pub fn from_matrix(m: &Matrix) -> Self {
        Self {
            shape: vec![m.rows, m.cols],
            data: NpyData::F64(m.data.clone()),
        }
    }

    pub fn from_matrix_3d(blocks: &[Matrix]) -> Result<Self> {
        if blocks.is_empty() {
            return Ok(Self {
                shape: vec![0, 0, 0],
                data: NpyData::F64(Vec::new()),
            });
        }
        let (r, c) = blocks[0].shape();
        let mut flat = Vec::with_capacity(blocks.len() * r * c);
        for b in blocks {
            if b.shape() != (r, c) {
                return Err(DemMcmError::ShapeMismatch(
                    "All blocks in 3D .npy must share shape".to_string(),
                ));
            }
            flat.extend_from_slice(&b.data);
        }
        Ok(Self {
            shape: vec![blocks.len(), r, c],
            data: NpyData::F64(flat),
        })
    }

    pub fn from_i64_2d(rows: &[Vec<i64>]) -> Result<Self> {
        if rows.is_empty() {
            return Ok(Self {
                shape: vec![0, 0],
                data: NpyData::I64(Vec::new()),
            });
        }
        let cols = rows[0].len();
        let mut flat = Vec::with_capacity(rows.len() * cols);
        for r in rows {
            if r.len() != cols {
                return Err(DemMcmError::ShapeMismatch(
                    "Ragged 2D i64 array".to_string(),
                ));
            }
            flat.extend_from_slice(r);
        }
        Ok(Self {
            shape: vec![rows.len(), cols],
            data: NpyData::I64(flat),
        })
    }

    pub fn to_f64_vec(&self) -> Vec<f64> {
        match &self.data {
            NpyData::F64(v) => v.clone(),
            NpyData::F32(v) => v.iter().map(|&x| x as f64).collect(),
            NpyData::I64(v) => v.iter().map(|&x| x as f64).collect(),
            NpyData::I32(v) => v.iter().map(|&x| x as f64).collect(),
            NpyData::Bool(v) => v.iter().map(|&x| if x { 1.0 } else { 0.0 }).collect(),
        }
    }

    pub fn to_i64_vec(&self) -> Result<Vec<i64>> {
        match &self.data {
            NpyData::I64(v) => Ok(v.clone()),
            NpyData::I32(v) => Ok(v.iter().map(|&x| x as i64).collect()),
            NpyData::F64(v) => Ok(v.iter().map(|&x| x as i64).collect()),
            NpyData::F32(v) => Ok(v.iter().map(|&x| x as i64).collect()),
            NpyData::Bool(v) => Ok(v.iter().map(|&x| if x { 1 } else { 0 }).collect()),
        }
    }

    pub fn to_matrix(&self) -> Result<Matrix> {
        if self.shape.len() != 2 {
            return Err(DemMcmError::ShapeMismatch(format!(
                "Expected 2D array for Matrix, got shape {:?}",
                self.shape
            )));
        }
        Matrix::from_vec(self.shape[0], self.shape[1], self.to_f64_vec())
    }

    pub fn to_matrix_3d(&self) -> Result<Vec<Matrix>> {
        if self.shape.len() != 3 {
            return Err(DemMcmError::ShapeMismatch(format!(
                "Expected 3D array for P_blocks, got shape {:?}",
                self.shape
            )));
        }
        let (b, r, c) = (self.shape[0], self.shape[1], self.shape[2]);
        let flat = self.to_f64_vec();
        let stride = r * c;
        let mut out = Vec::with_capacity(b);
        for i in 0..b {
            out.push(Matrix::from_vec(
                r,
                c,
                flat[i * stride..(i + 1) * stride].to_vec(),
            )?);
        }
        Ok(out)
    }

    pub fn to_i64_2d(&self) -> Result<Vec<Vec<i64>>> {
        if self.shape.len() != 2 {
            return Err(DemMcmError::ShapeMismatch(format!(
                "Expected 2D i64 array, got {:?}",
                self.shape
            )));
        }
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let flat = self.to_i64_vec()?;
        let mut out = Vec::with_capacity(rows);
        for r in 0..rows {
            out.push(flat[r * cols..(r + 1) * cols].to_vec());
        }
        Ok(out)
    }

    /// Encode this array to `.npy` bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let descr = match &self.data {
            NpyData::F64(_) => "<f8",
            NpyData::F32(_) => "<f4",
            NpyData::I64(_) => "<i8",
            NpyData::I32(_) => "<i4",
            NpyData::Bool(_) => "|b1",
        };
        let shape_str = match self.shape.len() {
            0 => "()".to_string(),
            1 => format!("({},)", self.shape[0]),
            _ => {
                let parts: Vec<String> = self.shape.iter().map(|s| s.to_string()).collect();
                format!("({})", parts.join(", "))
            }
        };
        let mut header =
            format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_str}, }}");
        // Align (magic 6 + version 2 + header_len 2 + header + \n) to multiple of 64
        let base_len = 10 + header.len() + 1;
        let pad = (64 - (base_len % 64)) % 64;
        for _ in 0..pad {
            header.push(' ');
        }
        header.push('\n');

        let use_v2 = header.len() > u16::MAX as usize;
        let mut out = Vec::with_capacity(16 + header.len() + self.byte_len());
        out.extend_from_slice(MAGIC);
        if use_v2 {
            out.push(2);
            out.push(0);
            // Re-pad for 12-byte prefix
            let mut h2 = header.trim_end_matches('\n').trim_end().to_string();
            let b2 = 12 + h2.len() + 1;
            let p2 = (64 - (b2 % 64)) % 64;
            for _ in 0..p2 {
                h2.push(' ');
            }
            h2.push('\n');
            out.extend_from_slice(&(h2.len() as u32).to_le_bytes());
            out.extend_from_slice(h2.as_bytes());
        } else {
            out.push(1);
            out.push(0);
            out.extend_from_slice(&(header.len() as u16).to_le_bytes());
            out.extend_from_slice(header.as_bytes());
        }

        match &self.data {
            NpyData::F64(v) => {
                for &x in v {
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
            NpyData::F32(v) => {
                for &x in v {
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
            NpyData::I64(v) => {
                for &x in v {
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
            NpyData::I32(v) => {
                for &x in v {
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
            NpyData::Bool(v) => {
                for &x in v {
                    out.push(if x { 1 } else { 0 });
                }
            }
        }
        out
    }

    fn byte_len(&self) -> usize {
        match &self.data {
            NpyData::F64(v) => v.len() * 8,
            NpyData::F32(v) => v.len() * 4,
            NpyData::I64(v) => v.len() * 8,
            NpyData::I32(v) => v.len() * 4,
            NpyData::Bool(v) => v.len(),
        }
    }

    /// Decode an `.npy` byte buffer.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 10 || &bytes[0..6] != MAGIC {
            return Err(DemMcmError::Serialization(
                "Invalid .npy magic header".to_string(),
            ));
        }
        let major = bytes[6];
        let (header_len, data_offset) = match major {
            1 => {
                let hl = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
                (hl, 10 + hl)
            }
            2 | 3 => {
                if bytes.len() < 12 {
                    return Err(DemMcmError::Serialization(
                        "Truncated .npy v2 header".to_string(),
                    ));
                }
                let hl = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
                (hl, 12 + hl)
            }
            v => {
                return Err(DemMcmError::Serialization(format!(
                    "Unsupported .npy version {v}"
                )))
            }
        };
        if data_offset > bytes.len() {
            return Err(DemMcmError::Serialization(
                "Truncated .npy payload".to_string(),
            ));
        }
        let header_start = data_offset - header_len;
        let header = std::str::from_utf8(&bytes[header_start..data_offset])
            .map_err(|e| DemMcmError::Serialization(e.to_string()))?;

        let descr = extract_dict_str(header, "descr")?;
        let fortran =
            header.contains("'fortran_order': True") || header.contains("\"fortran_order\": True");
        let shape = extract_shape(header)?;
        let n_elems: usize = shape.iter().product();
        let payload = &bytes[data_offset..];

        let mut data = match descr.as_str() {
            "<f8" | "=f8" | "f8" => {
                if payload.len() < n_elems * 8 {
                    return Err(DemMcmError::Serialization(
                        ".npy f64 buffer too short".to_string(),
                    ));
                }
                let mut v = Vec::with_capacity(n_elems);
                for chunk in payload[..n_elems * 8].chunks_exact(8) {
                    v.push(f64::from_le_bytes(chunk.try_into().unwrap()));
                }
                NpyData::F64(v)
            }
            ">f8" => {
                if payload.len() < n_elems * 8 {
                    return Err(DemMcmError::Serialization(
                        ".npy >f8 buffer too short".to_string(),
                    ));
                }
                let mut v = Vec::with_capacity(n_elems);
                for chunk in payload[..n_elems * 8].chunks_exact(8) {
                    v.push(f64::from_be_bytes(chunk.try_into().unwrap()));
                }
                NpyData::F64(v)
            }
            "<f4" | "=f4" | "f4" => {
                if payload.len() < n_elems * 4 {
                    return Err(DemMcmError::Serialization(
                        ".npy f32 buffer too short".to_string(),
                    ));
                }
                let mut v = Vec::with_capacity(n_elems);
                for chunk in payload[..n_elems * 4].chunks_exact(4) {
                    v.push(f32::from_le_bytes(chunk.try_into().unwrap()));
                }
                NpyData::F32(v)
            }
            "<i8" | "=i8" | "i8" => {
                if payload.len() < n_elems * 8 {
                    return Err(DemMcmError::Serialization(
                        ".npy i64 buffer too short".to_string(),
                    ));
                }
                let mut v = Vec::with_capacity(n_elems);
                for chunk in payload[..n_elems * 8].chunks_exact(8) {
                    v.push(i64::from_le_bytes(chunk.try_into().unwrap()));
                }
                NpyData::I64(v)
            }
            "<i4" | "=i4" | "i4" => {
                if payload.len() < n_elems * 4 {
                    return Err(DemMcmError::Serialization(
                        ".npy i32 buffer too short".to_string(),
                    ));
                }
                let mut v = Vec::with_capacity(n_elems);
                for chunk in payload[..n_elems * 4].chunks_exact(4) {
                    v.push(i32::from_le_bytes(chunk.try_into().unwrap()));
                }
                NpyData::I32(v)
            }
            "|b1" | "b1" | "|u1" | "u1" => {
                if payload.len() < n_elems {
                    return Err(DemMcmError::Serialization(
                        ".npy bool/u1 buffer too short".to_string(),
                    ));
                }
                let v = payload[..n_elems].iter().map(|&b| b != 0).collect();
                NpyData::Bool(v)
            }
            other => {
                return Err(DemMcmError::Serialization(format!(
                    "Unsupported .npy dtype {other:?}"
                )))
            }
        };

        // Transpose if Fortran order and 2D
        if fortran && shape.len() == 2 {
            let (rows, cols) = (shape[0], shape[1]);
            data = match data {
                NpyData::F64(v) => NpyData::F64(transpose_flat(&v, rows, cols)),
                NpyData::F32(v) => NpyData::F32(transpose_flat(&v, rows, cols)),
                NpyData::I64(v) => NpyData::I64(transpose_flat(&v, rows, cols)),
                NpyData::I32(v) => NpyData::I32(transpose_flat(&v, rows, cols)),
                NpyData::Bool(v) => NpyData::Bool(transpose_flat(&v, rows, cols)),
            };
        }

        Ok(Self { shape, data })
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        fs::write(path, self.to_bytes())?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)?;
        Self::from_bytes(&bytes)
    }
}

/// Write a `.npz` archive (ZIP of `<name>.npy` entries) readable by `numpy.load`.
pub fn save_npz(path: &Path, arrays: &[(&str, NpyArray)]) -> Result<()> {
    let encoded: Vec<(String, Vec<u8>)> = arrays
        .iter()
        .map(|(name, arr)| {
            let entry_name = if name.ends_with(".npy") {
                (*name).to_string()
            } else {
                format!("{name}.npy")
            };
            (entry_name, arr.to_bytes())
        })
        .collect();
    let refs: Vec<(&str, &[u8])> = encoded
        .iter()
        .map(|(n, b)| (n.as_str(), b.as_slice()))
        .collect();
    fs::write(path, write_zip_stored(&refs))?;
    Ok(())
}

/// Load a `.npz` archive into a map `array_name -> NpyArray`.
pub fn load_npz(path: &Path) -> Result<HashMap<String, NpyArray>> {
    let raw = fs::read(path)?;
    let entries = read_zip_entries(&raw)?;
    let mut map = HashMap::new();
    for (name, bytes) in entries {
        let key = name.strip_suffix(".npy").unwrap_or(&name).to_string();
        map.insert(key, NpyArray::from_bytes(&bytes)?);
    }
    Ok(map)
}

fn transpose_flat<T: Copy>(fortran_data: &[T], rows: usize, cols: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(fortran_data.len());
    for r in 0..rows {
        for c in 0..cols {
            out.push(fortran_data[c * rows + r]);
        }
    }
    out
}

fn extract_dict_str(header: &str, key: &str) -> Result<String> {
    let key_pat1 = format!("'{key}'");
    let key_pat2 = format!("\"{key}\"");
    let idx = header
        .find(&key_pat1)
        .or_else(|| header.find(&key_pat2))
        .ok_or_else(|| DemMcmError::Serialization(format!("Missing {key} in .npy header")))?;
    let rest = &header[idx..];
    let colon = rest
        .find(':')
        .ok_or_else(|| DemMcmError::Serialization("Malformed .npy header".to_string()))?;
    let after = rest[colon + 1..].trim_start();
    let quote = after
        .chars()
        .next()
        .filter(|&c| c == '\'' || c == '"')
        .ok_or_else(|| DemMcmError::Serialization("Expected quoted descr in .npy".to_string()))?;
    let inner = &after[1..];
    let end = inner
        .find(quote)
        .ok_or_else(|| DemMcmError::Serialization("Unterminated descr in .npy".to_string()))?;
    Ok(inner[..end].to_string())
}

fn extract_shape(header: &str) -> Result<Vec<usize>> {
    let idx = header
        .find("'shape'")
        .or_else(|| header.find("\"shape\""))
        .ok_or_else(|| DemMcmError::Serialization("Missing shape in .npy header".to_string()))?;
    let rest = &header[idx..];
    let lparen = rest
        .find('(')
        .ok_or_else(|| DemMcmError::Serialization("Missing '(' in .npy shape".to_string()))?;
    let rparen = rest[lparen..]
        .find(')')
        .ok_or_else(|| DemMcmError::Serialization("Missing ')' in .npy shape".to_string()))?;
    let inner = rest[lparen + 1..lparen + rparen].trim();
    if inner.is_empty() {
        return Ok(vec![1]);
    }
    let mut dims = Vec::new();
    for part in inner.split(',') {
        let p = part.trim();
        if !p.is_empty() {
            let d: usize = p
                .parse()
                .map_err(|e| DemMcmError::Serialization(format!("Invalid .npy dim {p:?}: {e}")))?;
            dims.push(d);
        }
    }
    Ok(dims)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn npy_f64_and_3d_roundtrip() {
        let m = Matrix::from_rows(&[vec![1.5, 2.5, 3.5], vec![4.5, f64::NAN, 6.5]]).unwrap();
        let arr = NpyArray::from_matrix(&m);
        let bytes = arr.to_bytes();
        let dec = NpyArray::from_bytes(&bytes).unwrap();
        let m2 = dec.to_matrix().unwrap();
        assert_eq!(m2.shape(), (2, 3));
        assert_eq!(m2.get(0, 0), 1.5);
        assert!(m2.get(1, 1).is_nan());
    }

    #[test]
    fn npz_roundtrip() {
        let tmp = std::env::temp_dir().join("test_dem_mcm.npz");
        let a = NpyArray::from_f64_vec(vec![0.0, 0.5, 1.0]);
        let b = NpyArray::from_i64_vec(vec![10, 20]);
        save_npz(&tmp, &[("x", a.clone()), ("y", b.clone())]).unwrap();
        let loaded = load_npz(&tmp).unwrap();
        assert_eq!(loaded["x"], a);
        assert_eq!(loaded["y"], b);
        let _ = fs::remove_file(&tmp);
    }
}
