use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Path, PathBuf};

use safetensors::Dtype;
use serde_json::Value;

use crate::error::ToolError;
use crate::files::{Sha256Hex, open_input};

/// The authors' manifest: the restricted `tensors[]` shape with one
/// hash pair per named tensor. Parsed as data only; nothing from a
/// source repository is ever executed.
#[derive(Debug)]
pub struct ManifestTensor {
    pub shape: Vec<u64>,
    pub dtype: Dtype,
    pub base_sha256: Sha256Hex,
    pub edited_sha256: Sha256Hex,
}

pub struct Manifest {
    pub path: PathBuf,
    pub tensors: BTreeMap<String, ManifestTensor>,
}

/// Upper bound on a manifest file read into memory.
const MAX_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;

pub fn read_manifest(path: &Path) -> Result<Manifest, ToolError> {
    let fail = |detail: String| ToolError::Manifest {
        path: path.to_path_buf(),
        detail,
    };
    let (_resolved, mut file) = open_input(path)?;
    let mut bytes = Vec::new();
    file.by_ref()
        .take(MAX_MANIFEST_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| ToolError::io(path, e))?;
    if bytes.len() as u64 > MAX_MANIFEST_BYTES {
        return Err(fail("manifest too large".into()));
    }
    let root: Value = serde_json::from_slice(&bytes).map_err(|e| fail(e.to_string()))?;
    // Duplicate fields inside a tensors[] entry, and a duplicated
    // root "tensors" key, must not collapse last-wins unnoticed in
    // the Value parse above. (Safetensors headers need no such
    // scan: the official reader's own deserialization rejects
    // duplicate tensor-entry fields. The manifest is plain JSON
    // with no official reader, so the scan below is retained for
    // it alone.)
    reject_duplicate_entry_keys(&bytes).map_err(fail)?;
    let obj = root
        .as_object()
        .ok_or_else(|| fail("root must be an object".into()))?;
    if obj.keys().any(|k| k != "tensors") {
        return Err(fail("only the \"tensors\" field is allowed".into()));
    }
    let entries = obj
        .get("tensors")
        .and_then(Value::as_array)
        .ok_or_else(|| fail("\"tensors\" must be an array".into()))?;
    let mut tensors = BTreeMap::new();
    for entry in entries {
        let e = entry
            .as_object()
            .ok_or_else(|| fail("each tensors[] entry must be an object".into()))?;
        const FIELDS: [&str; 5] = ["name", "shape", "dtype", "base_sha256", "edited_sha256"];
        if e.len() != FIELDS.len() || FIELDS.iter().any(|f| !e.contains_key(*f)) {
            return Err(fail(format!(
                "each tensors[] entry must have exactly the fields {FIELDS:?}"
            )));
        }
        let name = e["name"]
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| fail("tensor name must be a non-empty string".into()))?
            .to_owned();
        let shape: Vec<u64> = e["shape"]
            .as_array()
            .ok_or_else(|| fail(format!("{name}: shape must be an array")))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .ok_or_else(|| fail(format!("{name}: shape entries must be unsigned integers")))
            })
            .collect::<Result<_, _>>()?;
        let dtype_str = e["dtype"]
            .as_str()
            .ok_or_else(|| fail(format!("{name}: dtype must be a string")))?;
        let dtype: Dtype = serde_json::from_value(Value::String(dtype_str.to_owned()))
            .map_err(|_| fail(format!("{name}: unknown dtype {dtype_str:?}")))?;
        let hash = |key: &str| -> Result<Sha256Hex, ToolError> {
            let s = e[key]
                .as_str()
                .ok_or_else(|| fail(format!("{name}: {key} must be a string")))?;
            Sha256Hex::parse(s).map_err(|_| fail(format!("{name}: {key} is not 64 hex characters")))
        };
        let tensor = ManifestTensor {
            shape,
            dtype,
            base_sha256: hash("base_sha256")?,
            edited_sha256: hash("edited_sha256")?,
        };
        if tensors.insert(name.clone(), tensor).is_some() {
            return Err(fail(format!("{name}: tensor listed twice")));
        }
    }
    Ok(Manifest {
        path: path.to_path_buf(),
        tensors,
    })
}

/// Rejects duplicate keys inside the objects directly under the root
/// of an already-valid JSON document, and a duplicated root
/// `"tensors"` key. Used only for the authors' manifest, reached
/// through its `tensors` array, whose per-tensor entries are
/// exactly those objects. Other keys of the root object itself are
/// not checked here (the caller allows only `"tensors"` there),
/// and neither are more deeply nested objects.
///
/// The input must already have passed `serde_json` parsing (which
/// bounds nesting depth and rejects a BOM and invalid UTF-8); this
/// scanner only re-walks the validated token structure, decoding
/// each key with `serde_json` so escaped spellings of the same key
/// still count as duplicates.
fn reject_duplicate_entry_keys(json: &[u8]) -> Result<(), String> {
    let text = std::str::from_utf8(json).map_err(|e| e.to_string())?;
    let mut scanner = DuplicateScanner {
        bytes: text.as_bytes(),
        pos: 0,
    };
    scanner.value(0)?;
    scanner.skip_ws();
    if scanner.pos != scanner.bytes.len() {
        return Err("trailing data after JSON value".into());
    }
    Ok(())
}

struct DuplicateScanner<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl DuplicateScanner<'_> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    /// Walks one value. `object_depth` is the depth an object at
    /// this position would have: the root object is 0, objects
    /// directly under it (or under its arrays) are 1.
    fn value(&mut self, object_depth: usize) -> Result<(), String> {
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.object(object_depth),
            Some(b'[') => self.array(object_depth),
            Some(b'"') => self.string().map(drop),
            Some(_) => {
                self.primitive();
                Ok(())
            }
            None => Err("unexpected end of JSON".into()),
        }
    }

    fn object(&mut self, depth: usize) -> Result<(), String> {
        self.pos += 1; // '{'
        let mut seen: BTreeSet<String> = BTreeSet::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(());
        }
        loop {
            self.skip_ws();
            let key = self.string()?;
            if depth == 1 && !seen.insert(key.clone()) {
                return Err(format!("duplicate field `{key}`"));
            }
            // Only the known root field is checked at depth 0: a
            // duplicated "tensors" (however spelled — `key` is the
            // decoded string) must refuse, not collapse last-wins.
            if depth == 0 && key == "tensors" && !seen.insert(key.clone()) {
                return Err("duplicate field `tensors`".into());
            }
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err("expected ':' in JSON object".into());
            }
            self.pos += 1;
            self.value(depth + 1)?;
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(());
                }
                _ => return Err("expected ',' or '}' in JSON object".into()),
            }
        }
    }

    fn array(&mut self, object_depth: usize) -> Result<(), String> {
        self.pos += 1; // '['
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(());
        }
        loop {
            self.value(object_depth)?;
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(());
                }
                _ => return Err("expected ',' or ']' in JSON array".into()),
            }
        }
    }

    /// Consumes one JSON string and returns its decoded value.
    fn string(&mut self) -> Result<String, String> {
        if self.peek() != Some(b'"') {
            return Err("expected JSON string".into());
        }
        let start = self.pos;
        self.pos += 1;
        while let Some(byte) = self.peek() {
            match byte {
                b'"' => {
                    self.pos += 1;
                    return serde_json::from_slice(&self.bytes[start..self.pos])
                        .map_err(|e| e.to_string());
                }
                b'\\' => {
                    self.pos += 2;
                    if self.pos > self.bytes.len() {
                        return Err("unterminated JSON string".into());
                    }
                }
                _ => self.pos += 1,
            }
        }
        Err("unterminated JSON string".into())
    }

    fn primitive(&mut self) {
        while let Some(byte) = self.peek() {
            if matches!(byte, b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r') {
                break;
            }
            self.pos += 1;
        }
    }
}
