//! Pure-Rust RFC 1951 DEFLATE, RFC 1952 GZIP, Snappy block, and ZIP codecs.
//!
//! Zero external dependencies; used to read `.parquet.gz`, Snappy-compressed
//! Parquet pages, and `.npz` (ZIP of `.npy` files).

use crate::error::{DemMcmError, Result};

// ============================================================================
// CRC32 (IEEE 802.3)
// ============================================================================

const fn make_crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut crc = i as u32;
        let mut b = 0;
        while b < 8 {
            crc = if (crc & 1) != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
            b += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

static CRC32_TABLE: [u32; 256] = make_crc32_table();

pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc = (crc >> 8) ^ CRC32_TABLE[((crc ^ b as u32) & 0xFF) as usize];
    }
    !crc
}

// ============================================================================
// RFC 1951 DEFLATE DECOMPRESSOR
// ============================================================================

struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_buf: u64,
    bit_cnt: u32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_buf: 0,
            bit_cnt: 0,
        }
    }

    #[inline(always)]
    fn fill(&mut self) {
        if self.bit_cnt <= 32 && self.byte_pos + 4 <= self.data.len() {
            let w = u32::from_le_bytes(
                self.data[self.byte_pos..self.byte_pos + 4]
                    .try_into()
                    .unwrap(),
            );
            self.bit_buf |= (w as u64) << self.bit_cnt;
            self.byte_pos += 4;
            self.bit_cnt += 32;
        }
        while self.bit_cnt < 56 && self.byte_pos < self.data.len() {
            self.bit_buf |= (self.data[self.byte_pos] as u64) << self.bit_cnt;
            self.byte_pos += 1;
            self.bit_cnt += 8;
        }
    }

    #[inline(always)]
    fn ensure(&mut self, n: u32) -> Result<()> {
        if self.bit_cnt < n {
            self.fill();
            if self.bit_cnt < n {
                return Err(DemMcmError::Serialization(
                    "Unexpected EOF in DEFLATE stream".to_string(),
                ));
            }
        }
        Ok(())
    }

    #[inline(always)]
    fn read_bits(&mut self, n: u32) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        self.ensure(n)?;
        let val = (self.bit_buf & ((1u64 << n) - 1)) as u32;
        self.bit_buf >>= n;
        self.bit_cnt -= n;
        Ok(val)
    }

    #[inline(always)]
    fn peek_up_to(&mut self, n: u32) -> u32 {
        if self.bit_cnt < n {
            self.fill();
        }
        (self.bit_buf & ((1u64 << n) - 1)) as u32
    }

    #[inline(always)]
    fn consume(&mut self, n: u32) {
        self.bit_buf >>= n;
        self.bit_cnt -= n;
    }

    fn align_byte(&mut self) {
        let discard = self.bit_cnt % 8;
        self.bit_buf >>= discard;
        self.bit_cnt -= discard;
    }

    fn read_bytes(&mut self, len: usize) -> Result<&'a [u8]> {
        self.align_byte();
        let buffered_bytes = (self.bit_cnt / 8) as usize;
        self.byte_pos -= buffered_bytes;
        self.bit_buf = 0;
        self.bit_cnt = 0;
        if self.byte_pos + len > self.data.len() {
            return Err(DemMcmError::Serialization(
                "Unexpected EOF reading stored DEFLATE block".to_string(),
            ));
        }
        let slice = &self.data[self.byte_pos..self.byte_pos + len];
        self.byte_pos += len;
        Ok(slice)
    }

    fn bytes_consumed(&self) -> usize {
        self.byte_pos - (self.bit_cnt / 8) as usize
    }
}

const FAST_BITS: usize = 9;
const FAST_SIZE: usize = 1 << FAST_BITS;

struct HuffmanTree {
    counts: [u16; 16],
    symbols: Vec<u16>,
    /// Packed `(len << 16) | sym` when `len > 0 && len <= FAST_BITS`, else `0`.
    fast: [u32; FAST_SIZE],
}

impl HuffmanTree {
    fn from_lengths(lengths: &[u8]) -> Result<Self> {
        let mut counts = [0u16; 16];
        for &l in lengths {
            if (l as usize) >= 16 {
                return Err(DemMcmError::Serialization(
                    "Invalid Huffman code length".to_string(),
                ));
            }
            if l > 0 {
                counts[l as usize] += 1;
            }
        }

        let mut offsets = [0u16; 16];
        let mut next_code = [0u32; 16];
        let mut total = 0u16;
        let mut code = 0u32;
        for bits in 1..16 {
            offsets[bits] = total;
            total += counts[bits];
            code = (code + counts[bits - 1] as u32) << 1;
            next_code[bits] = code;
        }

        let mut symbols = vec![0u16; total as usize];
        let mut fast = [0u32; FAST_SIZE];
        for (sym, &len) in lengths.iter().enumerate() {
            if len > 0 {
                let idx = offsets[len as usize] as usize;
                offsets[len as usize] += 1;
                symbols[idx] = sym as u16;
                let l = len as usize;
                let c = next_code[l];
                next_code[l] += 1;
                if l <= FAST_BITS {
                    // Reverse the `l` MSB-first bits to LSB-first bit order in BitReader.
                    let mut rev = 0usize;
                    for b in 0..l {
                        if (c >> (l - 1 - b)) & 1 != 0 {
                            rev |= 1 << b;
                        }
                    }
                    let packed = ((l as u32) << 16) | (sym as u32);
                    let step = 1usize << l;
                    let mut k = rev;
                    while k < FAST_SIZE {
                        fast[k] = packed;
                        k += step;
                    }
                }
            }
        }
        Ok(Self {
            counts,
            symbols,
            fast,
        })
    }

    #[inline(always)]
    fn decode(&self, reader: &mut BitReader<'_>) -> Result<u16> {
        let peek = reader.peek_up_to(FAST_BITS as u32) as usize;
        let entry = self.fast[peek];
        if entry != 0 {
            let len = entry >> 16;
            if reader.bit_cnt >= len {
                reader.consume(len);
                return Ok((entry & 0xFFFF) as u16);
            }
        }
        let mut code = 0i32;
        let mut first = 0i32;
        let mut index = 0i32;
        for len in 1..16 {
            code |= reader.read_bits(1)? as i32;
            let count = self.counts[len as usize] as i32;
            if code - count < first {
                return Ok(self.symbols[(index + (code - first)) as usize]);
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err(DemMcmError::Serialization(
            "Invalid Huffman symbol in DEFLATE stream".to_string(),
        ))
    }
}

/// Decompress raw RFC 1951 DEFLATE bytes, returning `(output, bytes_consumed)`.
pub fn inflate_raw_with_consumed(input: &[u8]) -> Result<(Vec<u8>, usize)> {
    let mut reader = BitReader::new(input);
    let mut out = Vec::new();

    const LEN_BASE: [u16; 29] = [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115,
        131, 163, 195, 227, 258,
    ];
    const LEN_EXTRA: [u32; 29] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
    ];
    const DIST_BASE: [u16; 30] = [
        1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
        2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
    ];
    const DIST_EXTRA: [u32; 30] = [
        0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
        13, 13,
    ];
    const CL_ORDER: [usize; 19] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];

    loop {
        let bfinal = reader.read_bits(1)?;
        let btype = reader.read_bits(2)?;

        match btype {
            0 => {
                // Stored block
                let header = reader.read_bytes(4)?;
                let len = u16::from_le_bytes([header[0], header[1]]);
                let nlen = u16::from_le_bytes([header[2], header[3]]);
                if len != !nlen {
                    return Err(DemMcmError::Serialization(
                        "Corrupt stored DEFLATE block length".to_string(),
                    ));
                }
                let raw = reader.read_bytes(len as usize)?;
                out.extend_from_slice(raw);
            }
            1 | 2 => {
                let (lit_tree, dist_tree) = if btype == 1 {
                    let mut lit_len = [8u8; 288];
                    lit_len[144..256].fill(9);
                    lit_len[256..280].fill(7);
                    let dist_len = [5u8; 32];
                    (
                        HuffmanTree::from_lengths(&lit_len)?,
                        HuffmanTree::from_lengths(&dist_len)?,
                    )
                } else {
                    let hlit = reader.read_bits(5)? as usize + 257;
                    let hdist = reader.read_bits(5)? as usize + 1;
                    let hclen = reader.read_bits(4)? as usize + 4;
                    let mut cl_lengths = [0u8; 19];
                    for i in 0..hclen {
                        cl_lengths[CL_ORDER[i]] = reader.read_bits(3)? as u8;
                    }
                    let cl_tree = HuffmanTree::from_lengths(&cl_lengths)?;
                    let total = hlit + hdist;
                    let mut all_lens = Vec::with_capacity(total);
                    while all_lens.len() < total {
                        let sym = cl_tree.decode(&mut reader)?;
                        match sym {
                            0..=15 => all_lens.push(sym as u8),
                            16 => {
                                let prev = *all_lens.last().ok_or_else(|| {
                                    DemMcmError::Serialization("Invalid DEFLATE repeat".to_string())
                                })?;
                                let rep = reader.read_bits(2)? as usize + 3;
                                all_lens.resize(all_lens.len() + rep, prev);
                            }
                            17 => {
                                let rep = reader.read_bits(3)? as usize + 3;
                                all_lens.resize(all_lens.len() + rep, 0);
                            }
                            18 => {
                                let rep = reader.read_bits(7)? as usize + 11;
                                all_lens.resize(all_lens.len() + rep, 0);
                            }
                            _ => {
                                return Err(DemMcmError::Serialization(
                                    "Invalid code length symbol".to_string(),
                                ))
                            }
                        }
                    }
                    (
                        HuffmanTree::from_lengths(&all_lens[..hlit])?,
                        HuffmanTree::from_lengths(&all_lens[hlit..total])?,
                    )
                };

                loop {
                    let sym = lit_tree.decode(&mut reader)?;
                    if sym < 256 {
                        out.push(sym as u8);
                    } else if sym == 256 {
                        break;
                    } else {
                        let len_idx = (sym - 257) as usize;
                        if len_idx >= LEN_BASE.len() {
                            return Err(DemMcmError::Serialization(
                                "Invalid DEFLATE length code".to_string(),
                            ));
                        }
                        let length = LEN_BASE[len_idx] as usize
                            + reader.read_bits(LEN_EXTRA[len_idx])? as usize;
                        let dist_sym = dist_tree.decode(&mut reader)? as usize;
                        if dist_sym >= DIST_BASE.len() {
                            return Err(DemMcmError::Serialization(
                                "Invalid DEFLATE distance code".to_string(),
                            ));
                        }
                        let dist = DIST_BASE[dist_sym] as usize
                            + reader.read_bits(DIST_EXTRA[dist_sym])? as usize;
                        if dist == 0 || dist > out.len() {
                            return Err(DemMcmError::Serialization(
                                "Invalid DEFLATE back-reference distance".to_string(),
                            ));
                        }
                        if dist >= length {
                            let start = out.len() - dist;
                            out.extend_from_within(start..start + length);
                        } else {
                            for _ in 0..length {
                                let b = out[out.len() - dist];
                                out.push(b);
                            }
                        }
                    }
                }
            }
            _ => {
                return Err(DemMcmError::Serialization(
                    "Reserved DEFLATE block type".to_string(),
                ))
            }
        }

        if bfinal == 1 {
            break;
        }
    }

    reader.align_byte();
    Ok((out, reader.bytes_consumed()))
}

pub fn inflate_raw(input: &[u8]) -> Result<Vec<u8>> {
    Ok(inflate_raw_with_consumed(input)?.0)
}

/// Decompress a GZIP (RFC 1952) byte stream (supports multi-member streams).
pub fn gunzip(input: &[u8]) -> Result<Vec<u8>> {
    let mut pos = 0usize;
    let mut total_out = Vec::new();

    while pos < input.len() {
        if pos + 10 > input.len() || input[pos] != 0x1F || input[pos + 1] != 0x8B {
            return Err(DemMcmError::Serialization(
                "Invalid GZIP magic header".to_string(),
            ));
        }
        let cm = input[pos + 2];
        if cm != 8 {
            return Err(DemMcmError::Serialization(format!(
                "Unsupported GZIP compression method {cm}"
            )));
        }
        let flg = input[pos + 3];
        pos += 10;

        if flg & 0x04 != 0 {
            if pos + 2 > input.len() {
                return Err(DemMcmError::Serialization(
                    "Truncated GZIP FEXTRA".to_string(),
                ));
            }
            let xlen = u16::from_le_bytes([input[pos], input[pos + 1]]) as usize;
            pos += 2 + xlen;
        }
        if flg & 0x08 != 0 {
            while pos < input.len() && input[pos] != 0 {
                pos += 1;
            }
            pos += 1;
        }
        if flg & 0x10 != 0 {
            while pos < input.len() && input[pos] != 0 {
                pos += 1;
            }
            pos += 1;
        }
        if flg & 0x02 != 0 {
            pos += 2;
        }
        if pos > input.len() {
            return Err(DemMcmError::Serialization(
                "Truncated GZIP header".to_string(),
            ));
        }

        let (member_out, consumed) = inflate_raw_with_consumed(&input[pos..])?;
        pos += consumed;
        if pos + 8 > input.len() {
            return Err(DemMcmError::Serialization(
                "Truncated GZIP trailer".to_string(),
            ));
        }
        let expected_crc =
            u32::from_le_bytes([input[pos], input[pos + 1], input[pos + 2], input[pos + 3]]);
        let expected_size = u32::from_le_bytes([
            input[pos + 4],
            input[pos + 5],
            input[pos + 6],
            input[pos + 7],
        ]);
        pos += 8;

        if crc32(&member_out) != expected_crc {
            return Err(DemMcmError::Serialization(
                "GZIP CRC32 mismatch".to_string(),
            ));
        }
        if (member_out.len() as u32) != expected_size {
            return Err(DemMcmError::Serialization(
                "GZIP ISIZE mismatch".to_string(),
            ));
        }
        total_out.extend_from_slice(&member_out);
    }

    Ok(total_out)
}

/// Decompress a zlib (RFC 1950) stream (used by Parquet GZIP codec if zlib-wrapped).
pub fn decompress_gzip_or_zlib(input: &[u8]) -> Result<Vec<u8>> {
    if input.len() >= 2 && input[0] == 0x1F && input[1] == 0x8B {
        return gunzip(input);
    }
    if input.len() >= 6 && (input[0] & 0x0F) == 8 {
        // Zlib header
        let fdict = (input[1] & 0x20) != 0;
        let offset = if fdict { 6 } else { 2 };
        return inflate_raw(&input[offset..]);
    }
    inflate_raw(input)
}

// ============================================================================
// SNAPPY BLOCK DECOMPRESSOR
// ============================================================================

/// Decompress a raw Snappy block (Parquet `CompressionCodec::SNAPPY`).
pub fn snappy_decompress(input: &[u8]) -> Result<Vec<u8>> {
    let mut pos = 0usize;
    let mut uncompressed_len = 0usize;
    let mut shift = 0u32;
    loop {
        if pos >= input.len() {
            return Err(DemMcmError::Serialization(
                "Truncated Snappy varint".to_string(),
            ));
        }
        let b = input[pos];
        pos += 1;
        uncompressed_len |= ((b & 0x7F) as usize) << shift;
        if b & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift >= 35 {
            return Err(DemMcmError::Serialization(
                "Overflow in Snappy varint".to_string(),
            ));
        }
    }

    let mut out = Vec::with_capacity(uncompressed_len);
    while pos < input.len() {
        let tag = input[pos];
        pos += 1;
        match tag & 0x03 {
            0 => {
                let mut len = ((tag >> 2) as usize) + 1;
                if len > 60 {
                    let extra_bytes = len - 60;
                    if pos + extra_bytes > input.len() {
                        return Err(DemMcmError::Serialization(
                            "Truncated Snappy literal len".to_string(),
                        ));
                    }
                    let mut val = 0usize;
                    for i in 0..extra_bytes {
                        val |= (input[pos + i] as usize) << (8 * i);
                    }
                    pos += extra_bytes;
                    len = val + 1;
                }
                if pos + len > input.len() {
                    return Err(DemMcmError::Serialization(
                        "Truncated Snappy literal".to_string(),
                    ));
                }
                out.extend_from_slice(&input[pos..pos + len]);
                pos += len;
            }
            1 => {
                if pos >= input.len() {
                    return Err(DemMcmError::Serialization(
                        "Truncated Snappy copy-1".to_string(),
                    ));
                }
                let len = (((tag >> 2) & 0x07) as usize) + 4;
                let off = (((tag & 0xE0) as usize) << 3) | (input[pos] as usize);
                pos += 1;
                copy_backref(&mut out, off, len)?;
            }
            2 => {
                if pos + 2 > input.len() {
                    return Err(DemMcmError::Serialization(
                        "Truncated Snappy copy-2".to_string(),
                    ));
                }
                let len = ((tag >> 2) as usize) + 1;
                let off = u16::from_le_bytes([input[pos], input[pos + 1]]) as usize;
                pos += 2;
                copy_backref(&mut out, off, len)?;
            }
            _ => {
                if pos + 4 > input.len() {
                    return Err(DemMcmError::Serialization(
                        "Truncated Snappy copy-4".to_string(),
                    ));
                }
                let len = ((tag >> 2) as usize) + 1;
                let off = u32::from_le_bytes([
                    input[pos],
                    input[pos + 1],
                    input[pos + 2],
                    input[pos + 3],
                ]) as usize;
                pos += 4;
                copy_backref(&mut out, off, len)?;
            }
        }
    }

    if out.len() != uncompressed_len {
        return Err(DemMcmError::Serialization(format!(
            "Snappy length mismatch: expected {uncompressed_len}, got {}",
            out.len()
        )));
    }
    Ok(out)
}

#[inline]
fn copy_backref(out: &mut Vec<u8>, off: usize, len: usize) -> Result<()> {
    if off == 0 || off > out.len() {
        return Err(DemMcmError::Serialization(
            "Invalid Snappy back-reference offset".to_string(),
        ));
    }
    for _ in 0..len {
        let b = out[out.len() - off];
        out.push(b);
    }
    Ok(())
}

// ============================================================================
// MINIMAL ZIP ARCHIVE (FOR .NPZ INTEROPERABILITY WITH NUMPY)
// ============================================================================

/// Read all entries from a ZIP archive (supports stored and deflated entries).
pub fn read_zip_entries(data: &[u8]) -> Result<Vec<(String, Vec<u8>)>> {
    // Locate End of Central Directory (0x06054b50) from end
    if data.len() < 22 {
        return Err(DemMcmError::Serialization(
            "File too short to be ZIP".to_string(),
        ));
    }
    let min_pos = data.len().saturating_sub(65_557);
    let mut eocd = None;
    for i in (min_pos..=data.len() - 22).rev() {
        if &data[i..i + 4] == b"PK\x05\x06" {
            eocd = Some(i);
            break;
        }
    }
    let eocd_pos =
        eocd.ok_or_else(|| DemMcmError::Serialization("Missing ZIP EOCD".to_string()))?;
    let n_entries = u16::from_le_bytes([data[eocd_pos + 10], data[eocd_pos + 11]]) as usize;
    let cd_offset = u32::from_le_bytes([
        data[eocd_pos + 16],
        data[eocd_pos + 17],
        data[eocd_pos + 18],
        data[eocd_pos + 19],
    ]) as usize;

    let mut pos = cd_offset;
    let mut entries = Vec::with_capacity(n_entries);
    for _ in 0..n_entries {
        if pos + 46 > data.len() || &data[pos..pos + 4] != b"PK\x01\x02" {
            return Err(DemMcmError::Serialization(
                "Invalid ZIP central directory entry".to_string(),
            ));
        }
        let method = u16::from_le_bytes([data[pos + 10], data[pos + 11]]);
        let comp_size = u32::from_le_bytes([
            data[pos + 20],
            data[pos + 21],
            data[pos + 22],
            data[pos + 23],
        ]) as usize;
        let name_len = u16::from_le_bytes([data[pos + 28], data[pos + 29]]) as usize;
        let extra_len = u16::from_le_bytes([data[pos + 30], data[pos + 31]]) as usize;
        let comment_len = u16::from_le_bytes([data[pos + 32], data[pos + 33]]) as usize;
        let local_off = u32::from_le_bytes([
            data[pos + 42],
            data[pos + 43],
            data[pos + 44],
            data[pos + 45],
        ]) as usize;

        let name = String::from_utf8_lossy(&data[pos + 46..pos + 46 + name_len]).into_owned();
        pos += 46 + name_len + extra_len + comment_len;

        if local_off + 30 > data.len() || &data[local_off..local_off + 4] != b"PK\x03\x04" {
            return Err(DemMcmError::Serialization(
                "Invalid ZIP local file header".to_string(),
            ));
        }
        let l_name_len = u16::from_le_bytes([data[local_off + 26], data[local_off + 27]]) as usize;
        let l_extra_len = u16::from_le_bytes([data[local_off + 28], data[local_off + 29]]) as usize;
        let payload_start = local_off + 30 + l_name_len + l_extra_len;
        let payload_end = payload_start + comp_size;
        if payload_end > data.len() {
            return Err(DemMcmError::Serialization(
                "Truncated ZIP payload".to_string(),
            ));
        }
        let payload = &data[payload_start..payload_end];
        let content = match method {
            0 => payload.to_vec(),
            8 => inflate_raw(payload)?,
            _ => {
                return Err(DemMcmError::Serialization(format!(
                    "Unsupported ZIP compression method {method}"
                )))
            }
        };
        entries.push((name, content));
    }
    Ok(entries)
}

/// Write entries as an uncompressed (STORED) ZIP archive (100% readable by Python `numpy.load`).
pub fn write_zip_stored(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut central = Vec::new();

    for &(name, data) in entries {
        let local_offset = out.len() as u32;
        let name_bytes = name.as_bytes();
        let c = crc32(data);
        let len = data.len() as u32;

        // Local File Header
        out.extend_from_slice(b"PK\x03\x04");
        out.extend_from_slice(&20u16.to_le_bytes()); // version needed
        out.extend_from_slice(&0u16.to_le_bytes()); // flags
        out.extend_from_slice(&0u16.to_le_bytes()); // method: stored
        out.extend_from_slice(&0u16.to_le_bytes()); // mod time
        out.extend_from_slice(&0u16.to_le_bytes()); // mod date
        out.extend_from_slice(&c.to_le_bytes());
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(name_bytes);
        out.extend_from_slice(data);

        // Central Directory Header
        central.extend_from_slice(b"PK\x01\x02");
        central.extend_from_slice(&20u16.to_le_bytes());
        central.extend_from_slice(&20u16.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&c.to_le_bytes());
        central.extend_from_slice(&len.to_le_bytes());
        central.extend_from_slice(&len.to_le_bytes());
        central.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&0u32.to_le_bytes());
        central.extend_from_slice(&local_offset.to_le_bytes());
        central.extend_from_slice(name_bytes);
    }

    let cd_offset = out.len() as u32;
    let cd_size = central.len() as u32;
    out.extend_from_slice(&central);

    // End of Central Directory
    out.extend_from_slice(b"PK\x05\x06");
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    out.extend_from_slice(&cd_size.to_le_bytes());
    out.extend_from_slice(&cd_offset.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zip_stored_roundtrip() {
        let bytes = write_zip_stored(&[("a.txt", b"hello"), ("b.bin", &[1, 2, 3, 4])]);
        let entries = read_zip_entries(&bytes).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0], ("a.txt".to_string(), b"hello".to_vec()));
        assert_eq!(entries[1], ("b.bin".to_string(), vec![1, 2, 3, 4]));
    }

    #[test]
    fn snappy_literal_and_copy() {
        // Uncompressed len=10: literal "abcde" (5 bytes), copy off=5 len=5 -> "abcdeabcde"
        let mut compressed = vec![10u8]; // varint len = 10
        compressed.push(4 << 2); // tag=0 (literal), literal len 5
        compressed.extend_from_slice(b"abcde");
        compressed.push(((5 - 4) << 2) | 1); // copy-1 len=5
        compressed.push(5); // off=5
        let out = snappy_decompress(&compressed).unwrap();
        assert_eq!(&out, b"abcdeabcde");
    }
}
