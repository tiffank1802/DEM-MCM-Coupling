//! Pure-Rust JSON parser and pretty-printer.

use std::collections::BTreeMap;
use std::fmt;

use crate::error::{DemMcmError, Result};

/// Dynamically-typed JSON value preserving deterministic key ordering via [`BTreeMap`].
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(BTreeMap<String, JsonValue>),
}

impl JsonValue {
    pub fn object() -> Self {
        Self::Object(BTreeMap::new())
    }

    pub fn as_object(&self) -> Option<&BTreeMap<String, JsonValue>> {
        match self {
            Self::Object(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_object_mut(&mut self) -> Option<&mut BTreeMap<String, JsonValue>> {
        match self {
            Self::Object(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&Vec<JsonValue>> {
        match self {
            Self::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Number(n) if n.is_finite() => Some(*n as i64),
            _ => None,
        }
    }

    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Self::Number(n) if n.is_finite() && *n >= 0.0 => Some(*n as usize),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        self.as_object()?.get(key)
    }

    pub fn insert(&mut self, key: impl Into<String>, value: JsonValue) {
        if let Self::Object(map) = self {
            map.insert(key.into(), value);
        }
    }

    /// Parse a JSON string into [`JsonValue`].
    pub fn parse(input: &str) -> Result<Self> {
        let mut p = Parser {
            bytes: input.as_bytes(),
            pos: 0,
        };
        let val = p.parse_value()?;
        p.skip_ws();
        if p.pos != p.bytes.len() {
            return Err(DemMcmError::Serialization(format!(
                "Trailing characters in JSON at byte {}",
                p.pos
            )));
        }
        Ok(val)
    }

    /// Render compact JSON string.
    pub fn to_string_compact(&self) -> String {
        let mut out = String::new();
        self.write_fmt(&mut out, None, 0);
        out
    }

    /// Render 2-space indented JSON string (matching `json.dump(..., indent=2)`).
    pub fn to_string_pretty(&self) -> String {
        let mut out = String::new();
        self.write_fmt(&mut out, Some(2), 0);
        out
    }

    fn write_fmt(&self, out: &mut String, indent: Option<usize>, depth: usize) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Self::Number(n) => {
                if !n.is_finite() {
                    out.push_str("null");
                } else if n.fract() == 0.0 && n.abs() < 9e15 {
                    out.push_str(&format!("{}", *n as i64));
                } else {
                    out.push_str(&format!("{n}"));
                }
            }
            Self::String(s) => write_escaped_str(out, s),
            Self::Array(arr) => {
                if arr.is_empty() {
                    out.push_str("[]");
                    return;
                }
                out.push('[');
                for (i, item) in arr.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    if let Some(ind) = indent {
                        out.push('\n');
                        for _ in 0..(depth + 1) * ind {
                            out.push(' ');
                        }
                    }
                    item.write_fmt(out, indent, depth + 1);
                }
                if let Some(ind) = indent {
                    out.push('\n');
                    for _ in 0..depth * ind {
                        out.push(' ');
                    }
                }
                out.push(']');
            }
            Self::Object(map) => {
                if map.is_empty() {
                    out.push_str("{}");
                    return;
                }
                out.push('{');
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    if let Some(ind) = indent {
                        out.push('\n');
                        for _ in 0..(depth + 1) * ind {
                            out.push(' ');
                        }
                    }
                    write_escaped_str(out, k);
                    out.push(':');
                    if indent.is_some() {
                        out.push(' ');
                    }
                    v.write_fmt(out, indent, depth + 1);
                }
                if let Some(ind) = indent {
                    out.push('\n');
                    for _ in 0..depth * ind {
                        out.push(' ');
                    }
                }
                out.push('}');
            }
        }
    }
}

impl fmt::Display for JsonValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_string_pretty())
    }
}

impl From<&str> for JsonValue {
    fn from(s: &str) -> Self {
        Self::String(s.to_string())
    }
}

impl From<String> for JsonValue {
    fn from(s: String) -> Self {
        Self::String(s)
    }
}

impl From<bool> for JsonValue {
    fn from(b: bool) -> Self {
        Self::Bool(b)
    }
}

impl From<f64> for JsonValue {
    fn from(n: f64) -> Self {
        Self::Number(n)
    }
}

impl From<usize> for JsonValue {
    fn from(n: usize) -> Self {
        Self::Number(n as f64)
    }
}

impl From<i64> for JsonValue {
    fn from(n: i64) -> Self {
        Self::Number(n as f64)
    }
}

impl<T: Into<JsonValue>> From<Vec<T>> for JsonValue {
    fn from(v: Vec<T>) -> Self {
        Self::Array(v.into_iter().map(Into::into).collect())
    }
}

impl<T: Into<JsonValue>> From<Option<T>> for JsonValue {
    fn from(v: Option<T>) -> Self {
        match v {
            Some(inner) => inner.into(),
            None => Self::Null,
        }
    }
}

fn write_escaped_str(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_ws();
        self.bytes.get(self.pos).copied()
    }

    fn expect(&mut self, ch: u8) -> Result<()> {
        self.skip_ws();
        if self.bytes.get(self.pos) == Some(&ch) {
            self.pos += 1;
            Ok(())
        } else {
            Err(DemMcmError::Serialization(format!(
                "Expected '{}' at byte {}",
                ch as char, self.pos
            )))
        }
    }

    fn parse_value(&mut self) -> Result<JsonValue> {
        match self.peek() {
            Some(b'n') => self.parse_null(),
            Some(b't') | Some(b'f') => self.parse_bool(),
            Some(b'"') => Ok(JsonValue::String(self.parse_string()?)),
            Some(b'[') => self.parse_array(),
            Some(b'{') => self.parse_object(),
            Some(b'-') | Some(b'0'..=b'9') => self.parse_number(),
            Some(b'N') if self.bytes[self.pos..].starts_with(b"NaN") => {
                self.pos += 3;
                Ok(JsonValue::Number(f64::NAN))
            }
            other => Err(DemMcmError::Serialization(format!(
                "Unexpected byte {:?} in JSON at {}",
                other.map(|b| b as char),
                self.pos
            ))),
        }
    }

    fn parse_null(&mut self) -> Result<JsonValue> {
        if self.bytes[self.pos..].starts_with(b"null") {
            self.pos += 4;
            Ok(JsonValue::Null)
        } else {
            Err(DemMcmError::Serialization("Invalid literal".to_string()))
        }
    }

    fn parse_bool(&mut self) -> Result<JsonValue> {
        if self.bytes[self.pos..].starts_with(b"true") {
            self.pos += 4;
            Ok(JsonValue::Bool(true))
        } else if self.bytes[self.pos..].starts_with(b"false") {
            self.pos += 5;
            Ok(JsonValue::Bool(false))
        } else {
            Err(DemMcmError::Serialization(
                "Invalid boolean literal".to_string(),
            ))
        }
    }

    fn parse_string(&mut self) -> Result<String> {
        self.expect(b'"')?;
        let mut s = String::new();
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            self.pos += 1;
            if b == b'"' {
                return Ok(s);
            }
            if b == b'\\' {
                if self.pos >= self.bytes.len() {
                    return Err(DemMcmError::Serialization(
                        "Unterminated escape".to_string(),
                    ));
                }
                let esc = self.bytes[self.pos];
                self.pos += 1;
                match esc {
                    b'"' => s.push('"'),
                    b'\\' => s.push('\\'),
                    b'/' => s.push('/'),
                    b'n' => s.push('\n'),
                    b'r' => s.push('\r'),
                    b't' => s.push('\t'),
                    b'b' => s.push('\x08'),
                    b'f' => s.push('\x0c'),
                    b'u' => {
                        if self.pos + 4 > self.bytes.len() {
                            return Err(DemMcmError::Serialization(
                                "Invalid \\u escape".to_string(),
                            ));
                        }
                        let hex = std::str::from_utf8(&self.bytes[self.pos..self.pos + 4])
                            .map_err(|e| DemMcmError::Serialization(e.to_string()))?;
                        let code = u32::from_str_radix(hex, 16)
                            .map_err(|e| DemMcmError::Serialization(e.to_string()))?;
                        self.pos += 4;
                        if let Some(ch) = char::from_u32(code) {
                            s.push(ch);
                        }
                    }
                    _ => return Err(DemMcmError::Serialization("Unknown escape".to_string())),
                }
            } else if b < 0x80 {
                s.push(b as char);
            } else {
                // Multi-byte UTF-8 sequence starting at self.pos - 1
                let start = self.pos - 1;
                let rest = std::str::from_utf8(&self.bytes[start..])
                    .map_err(|e| DemMcmError::Serialization(e.to_string()))?;
                let ch = rest.chars().next().unwrap();
                s.push(ch);
                self.pos = start + ch.len_utf8();
            }
        }
        Err(DemMcmError::Serialization(
            "Unterminated JSON string".to_string(),
        ))
    }

    fn parse_number(&mut self) -> Result<JsonValue> {
        let start = self.pos;
        if self.bytes.get(self.pos) == Some(&b'-') {
            self.pos += 1;
        }
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            if b.is_ascii_digit() || matches!(b, b'.' | b'e' | b'E' | b'+' | b'-') {
                self.pos += 1;
            } else {
                break;
            }
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|e| DemMcmError::Serialization(e.to_string()))?;
        let val: f64 = s
            .parse()
            .map_err(|e| DemMcmError::Serialization(format!("Invalid JSON number {s:?}: {e}")))?;
        Ok(JsonValue::Number(val))
    }

    fn parse_array(&mut self) -> Result<JsonValue> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(JsonValue::Array(items));
        }
        loop {
            items.push(self.parse_value()?);
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    return Ok(JsonValue::Array(items));
                }
                _ => {
                    return Err(DemMcmError::Serialization(format!(
                        "Expected ',' or ']' in array at {}",
                        self.pos
                    )))
                }
            }
        }
    }

    fn parse_object(&mut self) -> Result<JsonValue> {
        self.expect(b'{')?;
        let mut map = BTreeMap::new();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonValue::Object(map));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.expect(b':')?;
            let val = self.parse_value()?;
            map.insert(key, val);
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(JsonValue::Object(map));
                }
                _ => {
                    return Err(DemMcmError::Serialization(format!(
                        "Expected ',' or '}}' in object at {}",
                        self.pos
                    )))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_roundtrip() {
        let mut obj = JsonValue::object();
        obj.insert("method", JsonValue::from("voronoi"));
        obj.insert("nlt", JsonValue::from(2usize));
        obj.insert("vw", JsonValue::from(0.5));
        obj.insert("inhomogeneous", JsonValue::from(false));
        obj.insert("diameter", JsonValue::Null);
        obj.insert("species", JsonValue::from(vec!["small", "large"]));

        let s = obj.to_string_pretty();
        let parsed = JsonValue::parse(&s).unwrap();
        assert_eq!(parsed, obj);
    }
}
