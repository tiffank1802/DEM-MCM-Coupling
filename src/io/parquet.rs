//! Pure-Rust Apache Parquet reader (`.parquet` and `.parquet.gz`).
//!
//! Supports flat table schemas with `REQUIRED` or `OPTIONAL` columns,
//! `UNCOMPRESSED`, `SNAPPY`, and `GZIP` compression, `DATA_PAGE` (V1) and
//! `DATA_PAGE_V2`, `DICTIONARY_PAGE`, `PLAIN` and `RLE_DICTIONARY` /
//! `PLAIN_DICTIONARY` encodings, and `INT32`, `INT64`, `FLOAT`, `DOUBLE`,
//! `BYTE_ARRAY` physical types.

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::Path;

use crate::data::DemFrame;
use crate::error::{DemMcmError, Result};
use crate::io::csv::parse_timestep_from_source;
use crate::io::deflate::{decompress_gzip_or_zlib, gunzip, snappy_decompress};

// ============================================================================
// THRIFT COMPACT PROTOCOL READER
// ============================================================================

struct CompactReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> CompactReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_u8(&mut self) -> Result<u8> {
        if self.pos >= self.data.len() {
            return Err(DemMcmError::Serialization(
                "Unexpected EOF in Thrift Compact stream".to_string(),
            ));
        }
        let b = self.data[self.pos];
        self.pos += 1;
        Ok(b)
    }

    fn read_uvarint(&mut self) -> Result<u64> {
        let mut res = 0u64;
        let mut shift = 0u32;
        loop {
            let b = self.read_u8()?;
            res |= ((b & 0x7F) as u64) << shift;
            if b & 0x80 == 0 {
                return Ok(res);
            }
            shift += 7;
            if shift >= 64 {
                return Err(DemMcmError::Serialization(
                    "Thrift varint overflow".to_string(),
                ));
            }
        }
    }

    #[inline]
    fn zigzag_to_i64(n: u64) -> i64 {
        ((n >> 1) as i64) ^ (-((n & 1) as i64))
    }

    fn read_i32(&mut self) -> Result<i32> {
        Ok(Self::zigzag_to_i64(self.read_uvarint()?) as i32)
    }

    fn read_i64(&mut self) -> Result<i64> {
        Ok(Self::zigzag_to_i64(self.read_uvarint()?))
    }

    fn read_binary(&mut self) -> Result<&'a [u8]> {
        let len = self.read_uvarint()? as usize;
        if self.pos + len > self.data.len() {
            return Err(DemMcmError::Serialization(
                "Truncated Thrift binary".to_string(),
            ));
        }
        let slice = &self.data[self.pos..self.pos + len];
        self.pos += len;
        Ok(slice)
    }

    fn read_string(&mut self) -> Result<String> {
        let b = self.read_binary()?;
        Ok(String::from_utf8_lossy(b).into_owned())
    }

    /// Read next struct field header `(field_id, field_type)`. Returns `None` on STOP (`0x00`).
    fn read_field_header(&mut self, last_field_id: &mut i16) -> Result<Option<(i16, u8)>> {
        let byte = self.read_u8()?;
        if byte == 0 {
            return Ok(None);
        }
        let delta = (byte >> 4) & 0x0F;
        let ftype = byte & 0x0F;
        let fid = if delta == 0 {
            Self::zigzag_to_i64(self.read_uvarint()?) as i16
        } else {
            *last_field_id + delta as i16
        };
        *last_field_id = fid;
        Ok(Some((fid, ftype)))
    }

    fn read_list_header(&mut self) -> Result<(usize, u8)> {
        let b = self.read_u8()?;
        let size_nibble = (b >> 4) & 0x0F;
        let elem_type = b & 0x0F;
        let size = if size_nibble == 0x0F {
            self.read_uvarint()? as usize
        } else {
            size_nibble as usize
        };
        Ok((size, elem_type))
    }

    fn skip_field(&mut self, ftype: u8, in_container: bool) -> Result<()> {
        match ftype {
            1 => Ok(()), // BOOL_TRUE (inside struct)
            2 => {
                if in_container {
                    let _ = self.read_u8()?;
                }
                Ok(())
            }
            3 => {
                let _ = self.read_u8()?;
                Ok(())
            }
            4..=6 => {
                let _ = self.read_uvarint()?;
                Ok(())
            }
            7 => {
                if self.pos + 8 > self.data.len() {
                    return Err(DemMcmError::Serialization(
                        "Truncated Thrift double".to_string(),
                    ));
                }
                self.pos += 8;
                Ok(())
            }
            8 => {
                let _ = self.read_binary()?;
                Ok(())
            }
            9 | 10 => {
                let (size, elem_type) = self.read_list_header()?;
                for _ in 0..size {
                    self.skip_field(elem_type, true)?;
                }
                Ok(())
            }
            11 => {
                let size = self.read_uvarint()? as usize;
                if size > 0 {
                    let kv = self.read_u8()?;
                    let kt = (kv >> 4) & 0x0F;
                    let vt = kv & 0x0F;
                    for _ in 0..size {
                        self.skip_field(kt, true)?;
                        self.skip_field(vt, true)?;
                    }
                }
                Ok(())
            }
            12 => {
                let mut last_id = 0i16;
                while let Some((_fid, child_type)) = self.read_field_header(&mut last_id)? {
                    self.skip_field(child_type, false)?;
                }
                Ok(())
            }
            other => Err(DemMcmError::Serialization(format!(
                "Unknown Thrift Compact type {other}"
            ))),
        }
    }
}

// ============================================================================
// PARQUET METADATA STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PhysicalType {
    Boolean = 0,
    Int32 = 1,
    Int64 = 2,
    Int96 = 3,
    Float = 4,
    Double = 5,
    ByteArray = 6,
    FixedLenByteArray = 7,
}

impl PhysicalType {
    fn from_i32(v: i32) -> Result<Self> {
        match v {
            0 => Ok(Self::Boolean),
            1 => Ok(Self::Int32),
            2 => Ok(Self::Int64),
            3 => Ok(Self::Int96),
            4 => Ok(Self::Float),
            5 => Ok(Self::Double),
            6 => Ok(Self::ByteArray),
            7 => Ok(Self::FixedLenByteArray),
            _ => Err(DemMcmError::Serialization(format!(
                "Unknown Parquet physical type {v}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompressionCodec {
    Uncompressed = 0,
    Snappy = 1,
    Gzip = 2,
}

impl CompressionCodec {
    fn from_i32(v: i32) -> Result<Self> {
        match v {
            0 => Ok(Self::Uncompressed),
            1 => Ok(Self::Snappy),
            2 => Ok(Self::Gzip),
            other => Err(DemMcmError::Serialization(format!(
                "Unsupported Parquet compression codec {other}"
            ))),
        }
    }
}

#[derive(Debug, Clone)]
struct SchemaElement {
    name: String,
    repetition_type: Option<i32>, // 0=REQUIRED, 1=OPTIONAL, 2=REPEATED
    num_children: Option<i32>,
}

#[derive(Debug, Clone)]
struct ColumnMetaData {
    phys_type: PhysicalType,
    path_in_schema: Vec<String>,
    codec: CompressionCodec,
    num_values: i64,
    total_compressed_size: i64,
    data_page_offset: i64,
    dictionary_page_offset: Option<i64>,
}

#[derive(Debug, Clone)]
struct ParquetMetaData {
    schema: Vec<SchemaElement>,
    num_rows: i64,
    row_groups: Vec<Vec<ColumnMetaData>>,
}

fn parse_schema_element(r: &mut CompactReader<'_>) -> Result<SchemaElement> {
    let mut name = String::new();
    let mut repetition_type = None;
    let mut num_children = None;
    let mut last_id = 0;
    while let Some((fid, ftype)) = r.read_field_header(&mut last_id)? {
        match fid {
            3 if ftype == 5 => repetition_type = Some(r.read_i32()?),
            4 if ftype == 8 => name = r.read_string()?,
            5 if ftype == 5 => num_children = Some(r.read_i32()?),
            _ => r.skip_field(ftype, false)?,
        }
    }
    Ok(SchemaElement {
        name,
        repetition_type,
        num_children,
    })
}

fn parse_column_metadata(r: &mut CompactReader<'_>) -> Result<ColumnMetaData> {
    let mut phys_type = None;
    let mut path_in_schema = Vec::new();
    let mut codec = CompressionCodec::Uncompressed;
    let mut num_values = 0i64;
    let mut total_compressed_size = 0i64;
    let mut data_page_offset = 0i64;
    let mut dictionary_page_offset = None;
    let mut last_id = 0;

    while let Some((fid, ftype)) = r.read_field_header(&mut last_id)? {
        match fid {
            1 if ftype == 5 => phys_type = Some(PhysicalType::from_i32(r.read_i32()?)?),
            3 if ftype == 9 => {
                let (sz, _) = r.read_list_header()?;
                for _ in 0..sz {
                    path_in_schema.push(r.read_string()?);
                }
            }
            4 if ftype == 5 => codec = CompressionCodec::from_i32(r.read_i32()?)?,
            5 if ftype == 6 => num_values = r.read_i64()?,
            7 if ftype == 6 => total_compressed_size = r.read_i64()?,
            9 if ftype == 6 => data_page_offset = r.read_i64()?,
            11 if ftype == 6 => dictionary_page_offset = Some(r.read_i64()?),
            _ => r.skip_field(ftype, false)?,
        }
    }
    Ok(ColumnMetaData {
        phys_type: phys_type.ok_or_else(|| {
            DemMcmError::Serialization("ColumnMetaData missing physical type".to_string())
        })?,
        path_in_schema,
        codec,
        num_values,
        total_compressed_size,
        data_page_offset,
        dictionary_page_offset,
    })
}

fn parse_column_chunk(r: &mut CompactReader<'_>) -> Result<Option<ColumnMetaData>> {
    let mut meta = None;
    let mut last_id = 0;
    while let Some((fid, ftype)) = r.read_field_header(&mut last_id)? {
        match fid {
            3 if ftype == 12 => meta = Some(parse_column_metadata(r)?),
            _ => r.skip_field(ftype, false)?,
        }
    }
    Ok(meta)
}

fn parse_row_group(r: &mut CompactReader<'_>) -> Result<Vec<ColumnMetaData>> {
    let mut cols = Vec::new();
    let mut last_id = 0;
    while let Some((fid, ftype)) = r.read_field_header(&mut last_id)? {
        match fid {
            1 if ftype == 9 => {
                let (sz, _) = r.read_list_header()?;
                for _ in 0..sz {
                    if let Some(c) = parse_column_chunk(r)? {
                        cols.push(c);
                    }
                }
            }
            _ => r.skip_field(ftype, false)?,
        }
    }
    Ok(cols)
}

fn parse_file_metadata(bytes: &[u8]) -> Result<ParquetMetaData> {
    let mut r = CompactReader::new(bytes);
    let mut schema = Vec::new();
    let mut num_rows = 0i64;
    let mut row_groups = Vec::new();
    let mut last_id = 0;

    while let Some((fid, ftype)) = r.read_field_header(&mut last_id)? {
        match fid {
            2 if ftype == 9 => {
                let (sz, _) = r.read_list_header()?;
                for _ in 0..sz {
                    schema.push(parse_schema_element(&mut r)?);
                }
            }
            3 if ftype == 6 => num_rows = r.read_i64()?,
            4 if ftype == 9 => {
                let (sz, _) = r.read_list_header()?;
                for _ in 0..sz {
                    row_groups.push(parse_row_group(&mut r)?);
                }
            }
            _ => r.skip_field(ftype, false)?,
        }
    }

    Ok(ParquetMetaData {
        schema,
        num_rows,
        row_groups,
    })
}

#[derive(Debug)]
enum PageHeaderKind {
    DataV1 {
        num_values: usize,
        encoding: i32,
    },
    Dictionary {
        num_values: usize,
    },
    DataV2 {
        num_values: usize,
        num_nulls: usize,
        encoding: i32,
        def_levels_len: usize,
        rep_levels_len: usize,
        is_compressed: bool,
    },
    Other,
}

#[derive(Debug)]
struct PageHeader {
    kind: PageHeaderKind,
    uncompressed_size: usize,
    compressed_size: usize,
    header_bytes: usize,
}

fn parse_page_header(bytes: &[u8]) -> Result<PageHeader> {
    let mut r = CompactReader::new(bytes);
    let mut page_type = -1i32;
    let mut uncompressed_size = 0usize;
    let mut compressed_size = 0usize;
    let mut dv1_num_values = 0usize;
    let mut dv1_encoding = 0i32;
    let mut dict_num_values = 0usize;
    let mut dv2_num_values = 0usize;
    let mut dv2_num_nulls = 0usize;
    let mut dv2_encoding = 0i32;
    let mut dv2_def_len = 0usize;
    let mut dv2_rep_len = 0usize;
    let mut dv2_is_compressed = true;
    let mut last_id = 0;

    while let Some((fid, ftype)) = r.read_field_header(&mut last_id)? {
        match fid {
            1 if ftype == 5 => page_type = r.read_i32()?,
            2 if ftype == 5 => uncompressed_size = r.read_i32()? as usize,
            3 if ftype == 5 => compressed_size = r.read_i32()? as usize,
            5 if ftype == 12 => {
                let mut sub_id = 0;
                while let Some((sfid, sftype)) = r.read_field_header(&mut sub_id)? {
                    match sfid {
                        1 if sftype == 5 => dv1_num_values = r.read_i32()? as usize,
                        2 if sftype == 5 => dv1_encoding = r.read_i32()?,
                        _ => r.skip_field(sftype, false)?,
                    }
                }
            }
            7 if ftype == 12 => {
                let mut sub_id = 0;
                while let Some((sfid, sftype)) = r.read_field_header(&mut sub_id)? {
                    match sfid {
                        1 if sftype == 5 => dict_num_values = r.read_i32()? as usize,
                        _ => r.skip_field(sftype, false)?,
                    }
                }
            }
            8 if ftype == 12 => {
                let mut sub_id = 0;
                while let Some((sfid, sftype)) = r.read_field_header(&mut sub_id)? {
                    match sfid {
                        1 if sftype == 5 => dv2_num_values = r.read_i32()? as usize,
                        2 if sftype == 5 => dv2_num_nulls = r.read_i32()? as usize,
                        4 if sftype == 5 => dv2_encoding = r.read_i32()?,
                        5 if sftype == 5 => dv2_def_len = r.read_i32()? as usize,
                        6 if sftype == 5 => dv2_rep_len = r.read_i32()? as usize,
                        7 if sftype == 1 || sftype == 2 => {
                            dv2_is_compressed = sftype == 1;
                        }
                        _ => r.skip_field(sftype, false)?,
                    }
                }
            }
            _ => r.skip_field(ftype, false)?,
        }
    }

    let kind = match page_type {
        0 => PageHeaderKind::DataV1 {
            num_values: dv1_num_values,
            encoding: dv1_encoding,
        },
        2 => PageHeaderKind::Dictionary {
            num_values: dict_num_values,
        },
        3 => PageHeaderKind::DataV2 {
            num_values: dv2_num_values,
            num_nulls: dv2_num_nulls,
            encoding: dv2_encoding,
            def_levels_len: dv2_def_len,
            rep_levels_len: dv2_rep_len,
            is_compressed: dv2_is_compressed,
        },
        _ => PageHeaderKind::Other,
    };

    Ok(PageHeader {
        kind,
        uncompressed_size,
        compressed_size,
        header_bytes: r.pos,
    })
}

// ============================================================================
// RLE / BIT-PACKING HYBRID & PLAIN DECODERS
// ============================================================================

fn decode_rle_hybrid(data: &[u8], bit_width: u32, count: usize) -> Result<Vec<u32>> {
    if bit_width == 0 {
        return Ok(vec![0; count]);
    }
    let mut out = Vec::with_capacity(count);
    let mut pos = 0usize;
    let byte_width = (bit_width as usize).div_ceil(8);
    let mask = if bit_width >= 32 {
        u32::MAX
    } else {
        (1u32 << bit_width) - 1
    };

    while out.len() < count && pos < data.len() {
        let mut header = 0u64;
        let mut shift = 0u32;
        loop {
            if pos >= data.len() {
                return Err(DemMcmError::Serialization(
                    "Truncated RLE header".to_string(),
                ));
            }
            let b = data[pos];
            pos += 1;
            header |= ((b & 0x7F) as u64) << shift;
            if b & 0x80 == 0 {
                break;
            }
            shift += 7;
        }

        if (header & 1) == 0 {
            // RLE run
            let run_len = (header >> 1) as usize;
            if pos + byte_width > data.len() {
                return Err(DemMcmError::Serialization(
                    "Truncated RLE value".to_string(),
                ));
            }
            let mut val = 0u32;
            for i in 0..byte_width {
                val |= (data[pos + i] as u32) << (8 * i);
            }
            pos += byte_width;
            let take = run_len.min(count - out.len());
            out.resize(out.len() + take, val);
        } else {
            // Bit-packed run: (header >> 1) * 8 values
            let n_groups = (header >> 1) as usize;
            let n_vals = n_groups * 8;
            let total_bytes = n_groups * bit_width as usize;
            if pos + total_bytes > data.len() {
                return Err(DemMcmError::Serialization(
                    "Truncated bit-packed run".to_string(),
                ));
            }
            let slice = &data[pos..pos + total_bytes];
            pos += total_bytes;

            let mut bit_buf = 0u64;
            let mut bits_in_buf = 0u32;
            let mut s_pos = 0usize;
            for _ in 0..n_vals {
                while bits_in_buf < bit_width && s_pos < slice.len() {
                    bit_buf |= (slice[s_pos] as u64) << bits_in_buf;
                    s_pos += 1;
                    bits_in_buf += 8;
                }
                let val = (bit_buf as u32) & mask;
                bit_buf >>= bit_width;
                bits_in_buf -= bit_width;
                if out.len() < count {
                    out.push(val);
                }
            }
        }
    }

    if out.len() < count {
        return Err(DemMcmError::Serialization(format!(
            "RLE decoded {}/{count} values",
            out.len()
        )));
    }
    Ok(out)
}

#[derive(Debug)]
enum ColumnValues {
    I64(Vec<i64>),
    F64(Vec<f64>),
    Str(Vec<String>),
}

fn decompress_page(codec: CompressionCodec, data: &[u8], expected_len: usize) -> Result<Vec<u8>> {
    match codec {
        CompressionCodec::Uncompressed => Ok(data.to_vec()),
        CompressionCodec::Snappy => snappy_decompress(data),
        CompressionCodec::Gzip => {
            let out = decompress_gzip_or_zlib(data)?;
            if out.len() != expected_len && expected_len > 0 {
                // Allow gzip even if expected matches
            }
            Ok(out)
        }
    }
}

fn decode_plain(
    phys_type: PhysicalType,
    bytes: &[u8],
    count: usize,
) -> Result<(ColumnValues, usize)> {
    match phys_type {
        PhysicalType::Int64 => {
            let need = count * 8;
            if bytes.len() < need {
                return Err(DemMcmError::Serialization(
                    "PLAIN INT64 truncated".to_string(),
                ));
            }
            let mut v = Vec::with_capacity(count);
            for c in bytes[..need].chunks_exact(8) {
                v.push(i64::from_le_bytes(c.try_into().unwrap()));
            }
            Ok((ColumnValues::I64(v), need))
        }
        PhysicalType::Int32 => {
            let need = count * 4;
            if bytes.len() < need {
                return Err(DemMcmError::Serialization(
                    "PLAIN INT32 truncated".to_string(),
                ));
            }
            let mut v = Vec::with_capacity(count);
            for c in bytes[..need].chunks_exact(4) {
                v.push(i32::from_le_bytes(c.try_into().unwrap()) as i64);
            }
            Ok((ColumnValues::I64(v), need))
        }
        PhysicalType::Double => {
            let need = count * 8;
            if bytes.len() < need {
                return Err(DemMcmError::Serialization(
                    "PLAIN DOUBLE truncated".to_string(),
                ));
            }
            let mut v = Vec::with_capacity(count);
            for c in bytes[..need].chunks_exact(8) {
                v.push(f64::from_le_bytes(c.try_into().unwrap()));
            }
            Ok((ColumnValues::F64(v), need))
        }
        PhysicalType::Float => {
            let need = count * 4;
            if bytes.len() < need {
                return Err(DemMcmError::Serialization(
                    "PLAIN FLOAT truncated".to_string(),
                ));
            }
            let mut v = Vec::with_capacity(count);
            for c in bytes[..need].chunks_exact(4) {
                v.push(f32::from_le_bytes(c.try_into().unwrap()) as f64);
            }
            Ok((ColumnValues::F64(v), need))
        }
        PhysicalType::ByteArray => {
            let mut pos = 0usize;
            let mut v = Vec::with_capacity(count);
            for _ in 0..count {
                if pos + 4 > bytes.len() {
                    return Err(DemMcmError::Serialization(
                        "PLAIN BYTE_ARRAY len truncated".to_string(),
                    ));
                }
                let len = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
                pos += 4;
                if pos + len > bytes.len() {
                    return Err(DemMcmError::Serialization(
                        "PLAIN BYTE_ARRAY data truncated".to_string(),
                    ));
                }
                v.push(String::from_utf8_lossy(&bytes[pos..pos + len]).into_owned());
                pos += len;
            }
            Ok((ColumnValues::Str(v), pos))
        }
        other => Err(DemMcmError::Serialization(format!(
            "Unsupported Parquet physical type {other:?}"
        ))),
    }
}

fn read_column_chunk(
    file_bytes: &[u8],
    col: &ColumnMetaData,
    is_optional: bool,
) -> Result<ColumnValues> {
    let start_offset = col
        .dictionary_page_offset
        .filter(|&o| o > 0 && o < col.data_page_offset)
        .unwrap_or(col.data_page_offset) as usize;
    let end_offset = (start_offset + col.total_compressed_size as usize).min(file_bytes.len());

    let mut pos = start_offset;
    let mut dict: Option<ColumnValues> = None;
    let mut values_read = 0usize;
    let target_values = col.num_values as usize;

    let mut out_i64: Vec<i64> = Vec::new();
    let mut out_f64: Vec<f64> = Vec::new();
    let mut out_str: Vec<String> = Vec::new();

    while values_read < target_values && pos < end_offset {
        let ph = parse_page_header(&file_bytes[pos..])?;
        pos += ph.header_bytes;
        if pos + ph.compressed_size > file_bytes.len() {
            return Err(DemMcmError::Serialization(
                "Truncated Parquet page payload".to_string(),
            ));
        }
        let raw_payload = &file_bytes[pos..pos + ph.compressed_size];
        pos += ph.compressed_size;

        match ph.kind {
            PageHeaderKind::Dictionary { num_values } => {
                let uncomp = decompress_page(col.codec, raw_payload, ph.uncompressed_size)?;
                let (d_vals, _) = decode_plain(col.phys_type, &uncomp, num_values)?;
                dict = Some(d_vals);
            }
            PageHeaderKind::DataV1 {
                num_values,
                encoding,
            } => {
                let uncomp = decompress_page(col.codec, raw_payload, ph.uncompressed_size)?;
                let mut cursor = 0usize;
                let n_present = if is_optional {
                    if cursor + 4 > uncomp.len() {
                        return Err(DemMcmError::Serialization(
                            "Truncated Parquet V1 definition levels".to_string(),
                        ));
                    }
                    let def_len =
                        u32::from_le_bytes(uncomp[cursor..cursor + 4].try_into().unwrap()) as usize;
                    cursor += 4;
                    let def_levels =
                        decode_rle_hybrid(&uncomp[cursor..cursor + def_len], 1, num_values)?;
                    cursor += def_len;
                    def_levels.iter().filter(|&&l| l > 0).count()
                } else {
                    num_values
                };

                append_page_values(
                    col.phys_type,
                    encoding,
                    &uncomp[cursor..],
                    n_present,
                    dict.as_ref(),
                    &mut out_i64,
                    &mut out_f64,
                    &mut out_str,
                )?;
                values_read += num_values;
            }
            PageHeaderKind::DataV2 {
                num_values,
                num_nulls,
                encoding,
                def_levels_len,
                rep_levels_len,
                is_compressed,
            } => {
                let level_bytes = rep_levels_len + def_levels_len;
                let data_part = if is_compressed {
                    decompress_page(
                        col.codec,
                        &raw_payload[level_bytes..],
                        ph.uncompressed_size.saturating_sub(level_bytes),
                    )?
                } else {
                    raw_payload[level_bytes..].to_vec()
                };
                let n_present = num_values.saturating_sub(num_nulls);
                append_page_values(
                    col.phys_type,
                    encoding,
                    &data_part,
                    n_present,
                    dict.as_ref(),
                    &mut out_i64,
                    &mut out_f64,
                    &mut out_str,
                )?;
                values_read += num_values;
            }
            PageHeaderKind::Other => {}
        }
    }

    match col.phys_type {
        PhysicalType::Int64 | PhysicalType::Int32 => Ok(ColumnValues::I64(out_i64)),
        PhysicalType::Double | PhysicalType::Float => Ok(ColumnValues::F64(out_f64)),
        PhysicalType::ByteArray => Ok(ColumnValues::Str(out_str)),
        _ => unreachable!(),
    }
}

#[allow(clippy::too_many_arguments)]
fn append_page_values(
    phys_type: PhysicalType,
    encoding: i32,
    data: &[u8],
    n_present: usize,
    dict: Option<&ColumnValues>,
    out_i64: &mut Vec<i64>,
    out_f64: &mut Vec<f64>,
    out_str: &mut Vec<String>,
) -> Result<()> {
    if n_present == 0 {
        return Ok(());
    }
    match encoding {
        // PLAIN
        0 => {
            let (vals, _) = decode_plain(phys_type, data, n_present)?;
            match vals {
                ColumnValues::I64(v) => out_i64.extend(v),
                ColumnValues::F64(v) => out_f64.extend(v),
                ColumnValues::Str(v) => out_str.extend(v),
            }
        }
        // PLAIN_DICTIONARY (2) or RLE_DICTIONARY (8)
        2 | 8 => {
            let dict_vals = dict.ok_or_else(|| {
                DemMcmError::Serialization(
                    "Dictionary encoding without dictionary page".to_string(),
                )
            })?;
            if data.is_empty() {
                return Err(DemMcmError::Serialization(
                    "Empty dictionary page data".to_string(),
                ));
            }
            let bit_width = data[0] as u32;
            let indices = decode_rle_hybrid(&data[1..], bit_width, n_present)?;
            match dict_vals {
                ColumnValues::I64(d) => {
                    for idx in indices {
                        out_i64.push(d[idx as usize]);
                    }
                }
                ColumnValues::F64(d) => {
                    for idx in indices {
                        out_f64.push(d[idx as usize]);
                    }
                }
                ColumnValues::Str(d) => {
                    for idx in indices {
                        out_str.push(d[idx as usize].clone());
                    }
                }
            }
        }
        other => {
            return Err(DemMcmError::Serialization(format!(
                "Unsupported Parquet page encoding {other}"
            )))
        }
    }
    Ok(())
}

// ============================================================================
// PUBLIC DEM PARQUET READER
// ============================================================================

/// Read a `.parquet` or `.parquet.gz` file into timestep-indexed [`DemFrame`]s.
///
/// Only the DEM coupling columns (`coordinates:0..2`, `Velocity:0..2`,
/// `Diameter`, `Particle_ID`, `Fichier_Source`, `TimeStep`, `Time`,
/// `Residence_Time`) are decoded, skipping unnecessary columns for speed.
pub fn read_dem_parquet(
    path: &Path,
    timestep_filter: Option<&HashSet<usize>>,
) -> Result<BTreeMap<usize, DemFrame>> {
    let raw = fs::read(path)?;
    let is_gz = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("gz"))
        || (raw.len() >= 2 && raw[0] == 0x1F && raw[1] == 0x8B);
    let file_bytes = if is_gz { gunzip(&raw)? } else { raw };
    read_dem_parquet_bytes(&file_bytes, timestep_filter)
}

/// Read raw (uncompressed) Parquet file bytes into timestep-indexed [`DemFrame`]s.
pub fn read_dem_parquet_bytes(
    file_bytes: &[u8],
    timestep_filter: Option<&HashSet<usize>>,
) -> Result<BTreeMap<usize, DemFrame>> {
    if file_bytes.len() < 12
        || &file_bytes[..4] != b"PAR1"
        || &file_bytes[file_bytes.len() - 4..] != b"PAR1"
    {
        return Err(DemMcmError::Serialization(
            "Invalid Parquet magic bytes".to_string(),
        ));
    }
    let meta_len = u32::from_le_bytes(
        file_bytes[file_bytes.len() - 8..file_bytes.len() - 4]
            .try_into()
            .unwrap(),
    ) as usize;
    let meta_start = file_bytes.len() - 8 - meta_len;
    let meta = parse_file_metadata(&file_bytes[meta_start..file_bytes.len() - 8])?;
    let _ = meta.num_rows;

    // Build optionality lookup for leaf schema elements
    let mut is_optional_map: BTreeMap<String, bool> = BTreeMap::new();
    for se in &meta.schema {
        if se.num_children.unwrap_or(0) == 0 {
            let opt = se.repetition_type == Some(1);
            is_optional_map.insert(se.name.clone(), opt);
        }
    }

    let wanted_cols: HashSet<&str> = [
        "coordinates:0",
        "coordinates:1",
        "coordinates:2",
        "Velocity:0",
        "Velocity:1",
        "Velocity:2",
        "Diameter",
        "Particle_ID",
        "Fichier_Source",
        "TimeStep",
        "Time",
        "Residence_Time",
    ]
    .into_iter()
    .collect();

    let mut frames: BTreeMap<usize, DemFrame> = BTreeMap::new();

    for rg in &meta.row_groups {
        // When filtering timesteps, decode only the timestep/source column first so
        // non-matching row groups (or chunks) skip decoding all kinematic columns.
        let mut cols_i64: BTreeMap<String, Vec<i64>> = BTreeMap::new();
        let mut cols_f64: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        let mut cols_str: BTreeMap<String, Vec<String>> = BTreeMap::new();

        let ts_vec: Vec<usize>;
        let keep_mask: Option<Vec<bool>>;

        if let Some(filter) = timestep_filter {
            let mut decoded_ts = false;
            if let Some(col_src) = rg
                .iter()
                .find(|c| c.path_in_schema.last().map(String::as_str) == Some("Fichier_Source"))
            {
                let opt = *is_optional_map.get("Fichier_Source").unwrap_or(&true);
                if let ColumnValues::Str(srcs) = read_column_chunk(file_bytes, col_src, opt)? {
                    let mut out_ts = Vec::with_capacity(srcs.len());
                    let mut last_src: Option<(String, usize)> = None;
                    let mut mask = Vec::with_capacity(srcs.len());
                    let mut any = false;
                    for s in &srcs {
                        let ts = if let Some((ref ps, pt)) = last_src {
                            if ps == s {
                                pt
                            } else {
                                let p = parse_timestep_from_source(s)?;
                                last_src = Some((s.clone(), p));
                                p
                            }
                        } else {
                            let p = parse_timestep_from_source(s)?;
                            last_src = Some((s.clone(), p));
                            p
                        };
                        let k = filter.contains(&ts);
                        any |= k;
                        mask.push(k);
                        out_ts.push(ts);
                    }
                    if !any {
                        continue;
                    }
                    ts_vec = out_ts;
                    keep_mask = Some(mask);
                    decoded_ts = true;
                } else {
                    ts_vec = Vec::new();
                    keep_mask = None;
                }
            } else if let Some(col_ts) = rg
                .iter()
                .find(|c| c.path_in_schema.last().map(String::as_str) == Some("TimeStep"))
            {
                let opt = *is_optional_map.get("TimeStep").unwrap_or(&true);
                if let ColumnValues::I64(tss) = read_column_chunk(file_bytes, col_ts, opt)? {
                    let mut out_ts = Vec::with_capacity(tss.len());
                    let mut mask = Vec::with_capacity(tss.len());
                    let mut any = false;
                    for &t in &tss {
                        let ts = t as usize;
                        let k = filter.contains(&ts);
                        any |= k;
                        mask.push(k);
                        out_ts.push(ts);
                    }
                    if !any {
                        continue;
                    }
                    ts_vec = out_ts;
                    keep_mask = Some(mask);
                    decoded_ts = true;
                } else {
                    ts_vec = Vec::new();
                    keep_mask = None;
                }
            } else {
                ts_vec = Vec::new();
                keep_mask = None;
            }
            if !decoded_ts {
                return Err(DemMcmError::MissingData(
                    "Parquet lacks both 'Fichier_Source' and 'TimeStep' columns".to_string(),
                ));
            }
        } else {
            ts_vec = Vec::new();
            keep_mask = None;
        }

        for col in rg {
            let name = col.path_in_schema.last().map(String::as_str).unwrap_or("");
            if !wanted_cols.contains(name) {
                continue;
            }
            if keep_mask.is_some() && (name == "Fichier_Source" || name == "TimeStep") {
                continue;
            }
            let opt = *is_optional_map.get(name).unwrap_or(&true);
            match read_column_chunk(file_bytes, col, opt)? {
                ColumnValues::I64(v) => {
                    cols_i64.insert(name.to_string(), v);
                }
                ColumnValues::F64(v) => {
                    cols_f64.insert(name.to_string(), v);
                }
                ColumnValues::Str(v) => {
                    cols_str.insert(name.to_string(), v);
                }
            }
        }

        let xs = cols_f64
            .get("coordinates:0")
            .ok_or_else(|| DemMcmError::MissingData("Parquet missing coordinates:0".to_string()))?;
        let ys = cols_f64
            .get("coordinates:1")
            .ok_or_else(|| DemMcmError::MissingData("Parquet missing coordinates:1".to_string()))?;
        let zs = cols_f64
            .get("coordinates:2")
            .ok_or_else(|| DemMcmError::MissingData("Parquet missing coordinates:2".to_string()))?;
        let n_rows = xs.len();

        let vxs = cols_f64.get("Velocity:0");
        let vys = cols_f64.get("Velocity:1");
        let vzs = cols_f64.get("Velocity:2");
        let diams = cols_f64.get("Diameter");
        let pids = cols_i64.get("Particle_ID");
        let times = cols_f64.get("Time");
        let res_times = cols_f64.get("Residence_Time");
        let sources = cols_str.get("Fichier_Source");
        let timesteps = cols_i64.get("TimeStep");

        if keep_mask.is_none() && sources.is_none() && timesteps.is_none() {
            return Err(DemMcmError::MissingData(
                "Parquet lacks both 'Fichier_Source' and 'TimeStep' columns".to_string(),
            ));
        }

        // Fast cache for consecutive rows sharing the same Fichier_Source string
        let mut last_src: Option<(String, usize)> = None;

        for row in 0..n_rows {
            if let Some(ref mask) = keep_mask {
                if !mask[row] {
                    continue;
                }
            }
            let ts = if !ts_vec.is_empty() {
                ts_vec[row]
            } else if let Some(srcs) = sources {
                let s = &srcs[row];
                if let Some((ref prev_s, prev_ts)) = last_src {
                    if prev_s == s {
                        prev_ts
                    } else {
                        let parsed = parse_timestep_from_source(s)?;
                        last_src = Some((s.clone(), parsed));
                        parsed
                    }
                } else {
                    let parsed = parse_timestep_from_source(s)?;
                    last_src = Some((s.clone(), parsed));
                    parsed
                }
            } else {
                timesteps.unwrap()[row] as usize
            };

            let frame = frames.entry(ts).or_insert_with(|| DemFrame::new(ts));
            frame.coordinates.push([xs[row], ys[row], zs[row]]);
            if let (Some(vx), Some(vy), Some(vz)) = (vxs, vys, vzs) {
                frame.velocities.push([vx[row], vy[row], vz[row]]);
            }
            if let Some(d) = diams {
                frame.diameters.push(d[row]);
            }
            if let Some(p) = pids {
                frame.particle_ids.push(p[row]);
            }
            if let Some(t) = times {
                frame.time = Some(t[row]);
            }
            if let Some(rt) = res_times {
                frame.residence_times.push(rt[row]);
            }
        }
    }

    Ok(frames)
}
