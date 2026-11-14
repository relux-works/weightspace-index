//! Patch planning: replacement-file parsing and validation, the sparse
//! index stream, the base/edited tensor diff, the full-vs-sparse stored
//! byte comparison, and the authors' manifest shape. No float
//! arithmetic anywhere: comparison and assignment are raw stored bytes.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use safetensors::Dtype;
use serde_json::{Value, json};

use crate::checkpoint::{Checkpoint, TensorMeta};
use crate::error::ToolError;
use crate::files::{CHUNK_BYTES, Sha256Hex, check_cancelled, open_input};

/// Which part of a sparse trio an entry name denotes, or a full tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntryKind {
    Full,
    Idx,
    New,
    Old,
}

/// Splits an entry name into its tensor name and kind, mirroring
/// `layout::check_replacement_file`: a `#` introduces exactly one of
/// the `idx`/`new`/`old` suffixes; anything else is refused.
pub fn parse_entry_name<'a>(path: &Path, name: &'a str) -> Result<(&'a str, EntryKind), ToolError> {
    match name.split_once('#') {
        None => Ok((name, EntryKind::Full)),
        Some((base, "idx")) => Ok((base, EntryKind::Idx)),
        Some((base, "new")) => Ok((base, EntryKind::New)),
        Some((base, "old")) => Ok((base, EntryKind::Old)),
        Some((_, suffix)) => Err(ToolError::malformed(
            path,
            format!("{name}: unknown suffix #{suffix}"),
        )),
    }
}

/// One patched tensor in a parsed replacement file.
#[derive(Clone, Debug)]
pub enum PatchTensor {
    Full {
        meta: TensorMeta,
    },
    Sparse {
        idx: TensorMeta,
        new: TensorMeta,
        old: TensorMeta,
        /// Number of sparse entries (`#idx` length).
        count: u64,
    },
}

/// A replacement file, parsed and cross-validated (sparse index values
/// themselves are validated while streaming, see [`for_each_index_block`]).
pub struct PatchFile {
    pub checkpoint: Checkpoint,
    pub tensors: BTreeMap<String, PatchTensor>,
}

pub fn parse_patch(path: &Path) -> Result<PatchFile, ToolError> {
    // Patch files take the same snapshot-backed official reader
    // path as checkpoints; there is no separate patch parser.
    let checkpoint = Checkpoint::open(path)?;
    let malformed = |detail: String| ToolError::malformed(path, detail);
    let mut full: BTreeMap<String, TensorMeta> = BTreeMap::new();
    let mut sparse: BTreeMap<String, [Option<TensorMeta>; 3]> = BTreeMap::new();
    for (name, meta) in &checkpoint.tensors {
        let (base, kind) = parse_entry_name(path, name)?;
        match kind {
            EntryKind::Full => drop(full.insert(base.to_owned(), meta.clone())),
            EntryKind::Idx => sparse.entry(base.to_owned()).or_default()[0] = Some(meta.clone()),
            EntryKind::New => sparse.entry(base.to_owned()).or_default()[1] = Some(meta.clone()),
            EntryKind::Old => sparse.entry(base.to_owned()).or_default()[2] = Some(meta.clone()),
        }
    }
    if full.is_empty() && sparse.is_empty() {
        return Err(malformed("no tensors".into()));
    }
    let mut tensors: BTreeMap<String, PatchTensor> = BTreeMap::new();
    for (base, meta) in full {
        tensors.insert(base, PatchTensor::Full { meta });
    }
    for (base, trio) in sparse {
        if tensors.contains_key(&base) {
            return Err(malformed(format!("{base}: both full and sparse")));
        }
        let [Some(idx), Some(new), Some(old)] = trio else {
            return Err(malformed(format!(
                "{base}: sparse tensors need #idx, #new and #old"
            )));
        };
        let count = match idx.shape.as_slice() {
            [n] if *n >= 1 && idx.dtype == Dtype::I64 && idx.byte_len() == 8 * *n => *n,
            _ => {
                return Err(malformed(format!(
                    "{base}#idx: must be a non-empty 1-D I64 tensor"
                )));
            }
        };
        // No legacy `#old` shape alias: the official reader has
        // already refused any entry whose stored bytes do not match
        // its declared shape, and here `#old` must declare exactly
        // one value per sparse index, like `#new`.
        if new.dtype != old.dtype || new.shape != [count] || old.shape != [count] {
            return Err(malformed(format!(
                "{base}: #new and #old must match #idx in shape and each other in dtype"
            )));
        }
        tensors.insert(
            base,
            PatchTensor::Sparse {
                idx,
                new,
                old,
                count,
            },
        );
    }
    Ok(PatchFile {
        checkpoint,
        tensors,
    })
}

/// Rejects a descriptor no checkpoint in this crate can produce —
/// inverted byte offsets or a sub-byte dtype whose
/// `elem_bytes()` is 0 — with a typed error, so the public diff
/// and index-stream entry points below never panic (for example
/// dividing by a zero element size) on a caller-constructed
/// [`TensorMeta`]. File-derived descriptors always pass: the
/// official reader enforces `start <= end` and `Checkpoint::open`
/// refuses sub-byte dtypes.
fn check_meta(path: &Path, tensor: &str, meta: &TensorMeta) -> Result<(), ToolError> {
    if meta.offsets_inverted() {
        return Err(ToolError::malformed(
            path,
            format!(
                "{tensor}: inverted data offsets ({}, {})",
                meta.data_offsets.0, meta.data_offsets.1
            ),
        ));
    }
    if meta.elem_bytes() == 0 {
        return Err(ToolError::UnsupportedDtype {
            path: path.to_path_buf(),
            tensor: tensor.to_owned(),
            dtype: meta.dtype,
        });
    }
    Ok(())
}

/// Largest number of sparse indices handled in one streaming block.
pub(crate) const INDEX_BLOCK: u64 = 4096;

/// Streams a `#idx` tensor in bounded blocks, validating that indices
/// are non-negative and strictly increasing across the whole stream.
/// The callback receives each block and the ordinal of its first
/// entry, and reads the matching `#new`/`#old` values through the
/// checkpoint's snapshot mapping.
pub fn for_each_index_block(
    checkpoint: &Checkpoint,
    tensor: &str,
    idx: &TensorMeta,
    cancelled: &AtomicBool,
    mut f: impl FnMut(&[i64], u64) -> Result<(), ToolError>,
) -> Result<(), ToolError> {
    check_meta(&checkpoint.path, tensor, idx)?;
    let count = idx.byte_len() / 8;
    let mut done = 0u64;
    let mut previous = -1i64;
    let mut buf = vec![0u8; usize::try_from(INDEX_BLOCK * 8).expect("block fits usize")];
    let mut block: Vec<i64> =
        Vec::with_capacity(usize::try_from(INDEX_BLOCK).expect("block fits usize"));
    while done < count {
        check_cancelled(cancelled)?;
        let n = INDEX_BLOCK.min(count - done);
        let bytes = usize::try_from(n * 8).expect("block fits usize");
        checkpoint.read_tensor_at(idx, done * 8, &mut buf[..bytes])?;
        block.clear();
        block.extend(
            buf[..bytes]
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().expect("8-byte chunk"))),
        );
        for &index in &block {
            if index <= previous {
                return Err(ToolError::malformed(
                    &checkpoint.path,
                    format!("{tensor}#idx: indices must be sorted, unique and non-negative"),
                ));
            }
            previous = index;
        }
        f(&block, done)?;
        done += n;
    }
    Ok(())
}

/// Full or sparse storage for one changed tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Representation {
    Full,
    Sparse,
}

/// One changed tensor considered by [`choose_representations`].
pub struct ChangedTensor<'a> {
    pub name: &'a str,
    pub dtype: Dtype,
    pub shape: &'a [u64],
    pub numel: u64,
    pub changed: u64,
}

/// Chooses a representation for a replacement file that would contain
/// only this one changed tensor (plus the standard
/// `__metadata__ {"format":"pt"}`), by the exact serialized sizes of
/// the two candidate files. Real files hold several changed tensors,
/// whose layout interacts, so `ops` uses [`choose_representations`]
/// for them; this single-tensor form stays for callers deciding one
/// tensor in isolation.
pub fn choose_representation(
    name: &str,
    dtype: Dtype,
    shape: &[u64],
    numel: u64,
    changed: u64,
) -> Result<Representation, ToolError> {
    let choices = choose_representations(&[ChangedTensor {
        name,
        dtype,
        shape,
        numel,
        changed,
    }])?;
    Ok(choices[0])
}

/// Chooses full or sparse storage for every changed tensor in one
/// replacement file, comparing actual serializer-bound file sizes.
///
/// The size of a given assignment is computed exactly as
/// `safetensors::serialize` (0.8.0 `prepare`) would lay the file out:
/// entries sorted by descending `Dtype` alignment order then name,
/// `data_offsets` cumulative from 0 in that order, header JSON with
/// those real start/end offsets (hence their real decimal digit
/// lengths) and the `__metadata__ {"format":"pt"}` entry this tool
/// always writes, the header padded with spaces up to a multiple of
/// 8, then `8 + padded header + tensor data`. The unit tests pin
/// this computation against `safetensors::serialize` output itself.
///
/// Decisions interact (one tensor's choice shifts later offsets,
/// digit lengths and the padding), so the choice is iterated from
/// all-full to a fixed point: each tensor in turn is set to sparse
/// only when the whole file is strictly smaller with it sparse,
/// holding the others at their current choices; ties store full.
/// The fixed point is guaranteed — flipping any single tensor from
/// the result would not make the serialized file smaller — but no
/// exhaustive search over all combinations is claimed.
pub fn choose_representations(
    tensors: &[ChangedTensor<'_>],
) -> Result<Vec<Representation>, ToolError> {
    if tensors.is_empty() {
        return Ok(Vec::new());
    }
    let templates = entry_templates(tensors)?;
    let mut choices = vec![Representation::Full; tensors.len()];
    let mut current = serialized_len(&templates, &choices)?;
    loop {
        let mut flipped = false;
        for i in 0..tensors.len() {
            let original = choices[i];
            choices[i] = match original {
                Representation::Full => Representation::Sparse,
                Representation::Sparse => Representation::Full,
            };
            let alternative = serialized_len(&templates, &choices)?;
            let (full_len, sparse_len) = match original {
                Representation::Full => (current, alternative),
                Representation::Sparse => (alternative, current),
            };
            let desired = if sparse_len < full_len {
                Representation::Sparse
            } else {
                Representation::Full
            };
            choices[i] = desired;
            if desired != original {
                current = if desired == Representation::Sparse {
                    sparse_len
                } else {
                    full_len
                };
                flipped = true;
            }
        }
        if !flipped {
            return Ok(choices);
        }
    }
}

/// One possible header/data entry of the replacement file: the full
/// entry of a tensor, or one leg of its sparse trio.
struct EntryTemplate {
    owner: usize,
    representation: Representation,
    dtype: Dtype,
    name: String,
    byte_len: u64,
    /// Rendered header entry length with `data_offsets` digits
    /// removed; the real length adds the decimal digits of the
    /// entry's actual start and end offsets.
    header_fixed_len: u64,
}

fn entry_templates(tensors: &[ChangedTensor<'_>]) -> Result<Vec<EntryTemplate>, ToolError> {
    let overflow = || ToolError::refused("stored byte comparison overflows");
    let mul = |a: u64, b: u64| a.checked_mul(b).ok_or_else(overflow);
    let mut templates = Vec::with_capacity(tensors.len() * 4);
    for (owner, tensor) in tensors.iter().enumerate() {
        if tensor.dtype.bitsize() % 8 != 0 {
            return Err(ToolError::refused(format!(
                "{}: dtype {:?} elements are not whole bytes",
                tensor.name, tensor.dtype
            )));
        }
        if tensor.changed == 0 || tensor.changed > tensor.numel {
            return Err(ToolError::refused(format!(
                "{}: {} changed elements out of {}",
                tensor.name, tensor.changed, tensor.numel
            )));
        }
        let elem = (tensor.dtype.bitsize() / 8) as u64;
        let value_bytes = mul(tensor.changed, elem)?;
        let sparse_shape = [tensor.changed];
        let specs = [
            (
                Representation::Full,
                tensor.name.to_owned(),
                tensor.dtype,
                tensor.shape,
                mul(tensor.numel, elem)?,
            ),
            (
                Representation::Sparse,
                format!("{}#idx", tensor.name),
                Dtype::I64,
                &sparse_shape[..],
                mul(tensor.changed, 8)?,
            ),
            (
                Representation::Sparse,
                format!("{}#new", tensor.name),
                tensor.dtype,
                &sparse_shape[..],
                value_bytes,
            ),
            (
                Representation::Sparse,
                format!("{}#old", tensor.name),
                tensor.dtype,
                &sparse_shape[..],
                value_bytes,
            ),
        ];
        for (representation, name, dtype, shape, byte_len) in specs {
            templates.push(EntryTemplate {
                header_fixed_len: header_entry_fixed_len(&name, dtype, shape),
                owner,
                representation,
                dtype,
                name,
                byte_len,
            });
        }
    }
    // The serializer sorts by descending dtype alignment, then name;
    // each entry's key is fixed, so sorting every possible entry once
    // gives each assignment's order as a subsequence.
    templates.sort_by(|a, b| b.dtype.cmp(&a.dtype).then_with(|| a.name.cmp(&b.name)));
    Ok(templates)
}

/// Exact serialized byte length of the replacement file for one
/// assignment, laid out as `safetensors::serialize` lays it out
/// (see [`choose_representations`]).
fn serialized_len(
    templates: &[EntryTemplate],
    choices: &[Representation],
) -> Result<u64, ToolError> {
    let overflow = || ToolError::refused("stored byte comparison overflows");
    let add = |a: u64, b: u64| a.checked_add(b).ok_or_else(overflow);
    // The "__metadata__":{"format":"pt"} entry always written first.
    let mut header_len = metadata_entry_len();
    let mut entries = 1u64;
    let mut offset = 0u64;
    for template in templates {
        if choices[template.owner] != template.representation {
            continue;
        }
        let end = add(offset, template.byte_len)?;
        let entry_len = add(
            template.header_fixed_len,
            add(decimal_digits(offset), decimal_digits(end))?,
        )?;
        header_len = add(header_len, entry_len)?;
        entries += 1;
        offset = end;
    }
    // Object braces plus one comma between adjacent entries.
    header_len = add(header_len, add(2, entries - 1)?)?;
    let padded = add(header_len, 7)? / 8 * 8;
    add(add(8, padded)?, offset)
}

fn decimal_digits(value: u64) -> u64 {
    let mut digits = 1;
    let mut rest = value;
    while rest >= 10 {
        rest /= 10;
        digits += 1;
    }
    digits
}

/// Header bytes of one `"name":{…}` entry as the serializer renders
/// it, minus the two `data_offsets` numbers (rendered here as the
/// one-digit placeholders `0,0`).
fn header_entry_fixed_len(name: &str, dtype: Dtype, shape: &[u64]) -> u64 {
    let rendered = json!({ name: {
        "dtype": dtype.to_string(),
        "shape": shape,
        "data_offsets": [0, 0],
    }});
    let text = serde_json::to_string(&rendered).expect("JSON values serialize");
    // Strip the outer braces of the one-entry object and the two
    // placeholder digits.
    (text.len() - 2 - 2) as u64
}

/// Header bytes of the `"__metadata__":{"format":"pt"}` entry.
fn metadata_entry_len() -> u64 {
    let rendered = json!({ "__metadata__": { "format": "pt" } });
    let text = serde_json::to_string(&rendered).expect("JSON values serialize");
    (text.len() - 2) as u64
}

/// Serialized length for a complete candidate list and assignment;
/// the spelling the oracle unit tests use.
#[cfg(test)]
fn serialized_file_len(
    tensors: &[ChangedTensor<'_>],
    choices: &[Representation],
) -> Result<u64, ToolError> {
    serialized_len(&entry_templates(tensors)?, choices)
}

/// Result of scanning one tensor pair.
pub struct TensorDiff {
    pub changed: u64,
    /// Present only when collection was requested and something changed:
    /// `(idx bytes, new bytes, old bytes)`.
    pub sparse: Option<(Vec<u8>, Vec<u8>, Vec<u8>)>,
}

/// Compares one tensor's stored bytes between base and edited in
/// bounded chunks over their snapshot mappings (never a whole
/// tensor at once). With `collect`, also materializes the sparse
/// arrays, which the replacement format itself requires as owned
/// tensors.
pub fn diff_tensor(
    name: &str,
    base: &Checkpoint,
    base_meta: &TensorMeta,
    edited: &Checkpoint,
    edited_meta: &TensorMeta,
    cancelled: &AtomicBool,
    collect: bool,
) -> Result<TensorDiff, ToolError> {
    check_meta(&base.path, name, base_meta)?;
    check_meta(&edited.path, name, edited_meta)?;
    let elem = usize::try_from(base_meta.elem_bytes()).expect("element size fits usize");
    let numel = base_meta.numel(&base.path, name)?;
    let chunk_elems = (CHUNK_BYTES / elem).max(1) as u64;
    let mut changed = 0u64;
    let mut arrays = collect.then(|| (Vec::new(), Vec::new(), Vec::new()));
    let mut base_buf = vec![0u8; CHUNK_BYTES];
    let mut edited_buf = vec![0u8; CHUNK_BYTES];
    let mut at = 0u64;
    while at < numel {
        check_cancelled(cancelled)?;
        let n = chunk_elems.min(numel - at);
        let bytes = usize::try_from(n).expect("chunk fits usize") * elem;
        base.read_tensor_at(
            base_meta,
            at * base_meta.elem_bytes(),
            &mut base_buf[..bytes],
        )?;
        edited.read_tensor_at(
            edited_meta,
            at * edited_meta.elem_bytes(),
            &mut edited_buf[..bytes],
        )?;
        for i in 0..usize::try_from(n).expect("chunk fits usize") {
            let lo = i * elem;
            let hi = lo + elem;
            if base_buf[lo..hi] != edited_buf[lo..hi] {
                let flat = at + i as u64;
                changed += 1;
                if let Some((idx, new, old)) = &mut arrays {
                    idx.extend_from_slice(&(flat as i64).to_le_bytes());
                    new.extend_from_slice(&edited_buf[lo..hi]);
                    old.extend_from_slice(&base_buf[lo..hi]);
                }
            }
        }
        at += n;
    }
    Ok(TensorDiff {
        changed,
        sparse: if changed == 0 { None } else { arrays },
    })
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choose_thresholds() {
        // BF16, 16 elements: full data is 32 B; one sparse change is
        // 12 B of data but needs three header entries instead of one,
        // so the exact stored size still favors the full tensor.
        let shape = [16u64];
        assert_eq!(
            choose_representation("w", Dtype::BF16, &shape, 16, 1).unwrap(),
            Representation::Full
        );
        assert_eq!(
            choose_representation("w", Dtype::BF16, &shape, 16, 16).unwrap(),
            Representation::Full
        );
        // At 512 elements the data saving (1024 B vs 12 B) outweighs the
        // two extra header entries.
        assert_eq!(
            choose_representation("w", Dtype::BF16, &[512], 512, 1).unwrap(),
            Representation::Sparse
        );
        // A scalar can never win sparse: one element costs 12 B sparse
        // against 2 B full, before headers.
        assert_eq!(
            choose_representation("s", Dtype::BF16, &[], 1, 1).unwrap(),
            Representation::Full
        );
        // Nearby control just past the break-even flips to full.
        let big = [1000u64];
        let mut last_sparse = 0;
        for changed in 1..=1000 {
            if choose_representation("w", Dtype::F32, &big, 1000, changed).unwrap()
                == Representation::Sparse
            {
                last_sparse = changed;
            }
        }
        assert!((1..1000).contains(&last_sparse));
        assert_eq!(
            choose_representation("w", Dtype::F32, &big, 1000, last_sparse + 1).unwrap(),
            Representation::Full
        );
    }

    /// Serialized length of the replacement file for one assignment,
    /// measured on actual `safetensors::serialize` output (zero-filled
    /// data of the right lengths; only sizes matter here).
    fn oracle_len(tensors: &[ChangedTensor<'_>], choices: &[Representation]) -> usize {
        use safetensors::tensor::TensorView;
        use std::collections::HashMap;
        let mut specs: Vec<(String, Dtype, Vec<usize>, usize)> = Vec::new();
        for (tensor, choice) in tensors.iter().zip(choices) {
            let elem = tensor.dtype.bitsize() / 8;
            let usize_of = |v: u64| usize::try_from(v).unwrap();
            match choice {
                Representation::Full => specs.push((
                    tensor.name.to_owned(),
                    tensor.dtype,
                    tensor.shape.iter().map(|&d| usize_of(d)).collect(),
                    usize_of(tensor.numel) * elem,
                )),
                Representation::Sparse => {
                    let n = usize_of(tensor.changed);
                    specs.push((format!("{}#idx", tensor.name), Dtype::I64, vec![n], n * 8));
                    specs.push((
                        format!("{}#new", tensor.name),
                        tensor.dtype,
                        vec![n],
                        n * elem,
                    ));
                    specs.push((
                        format!("{}#old", tensor.name),
                        tensor.dtype,
                        vec![n],
                        n * elem,
                    ));
                }
            }
        }
        let datas: Vec<Vec<u8>> = specs.iter().map(|(_, _, _, len)| vec![0u8; *len]).collect();
        let views: Vec<(String, TensorView<'_>)> = specs
            .iter()
            .zip(&datas)
            .map(|((name, dtype, shape, _), data)| {
                (
                    name.clone(),
                    TensorView::new(*dtype, shape.clone(), data).unwrap(),
                )
            })
            .collect();
        let mut metadata: HashMap<String, String> = HashMap::new();
        metadata.insert("format".into(), "pt".into());
        safetensors::serialize(
            views.iter().map(|(name, view)| (name.as_str(), view)),
            Some(metadata),
        )
        .unwrap()
        .len()
    }

    fn oracle_choice(tensors: &[ChangedTensor<'_>], choices: &[Representation]) -> u64 {
        let ours = serialized_file_len(tensors, choices).unwrap();
        assert_eq!(
            ours as usize,
            oracle_len(tensors, choices),
            "serializer-bound size disagrees with safetensors::serialize for {choices:?}"
        );
        ours
    }

    #[test]
    fn reviewer_flip_bf16_w200_23_changes() {
        // The independent reviewer's verified flip: official
        // serialize gives full 504 bytes vs sparse 508 for BF16
        // w[200] with 23 changes; sparse must not be chosen.
        let tensors = [ChangedTensor {
            name: "w",
            dtype: Dtype::BF16,
            shape: &[200],
            numel: 200,
            changed: 23,
        }];
        assert_eq!(oracle_choice(&tensors, &[Representation::Full]), 504);
        assert_eq!(oracle_choice(&tensors, &[Representation::Sparse]), 508);
        assert_eq!(
            choose_representation("w", Dtype::BF16, &[200], 200, 23).unwrap(),
            Representation::Full
        );
        assert_eq!(
            choose_representations(&tensors).unwrap(),
            [Representation::Full]
        );
    }

    #[test]
    fn serialized_len_matches_official_serializer_sweep() {
        // Single-tensor boundary sweep: for every changed count the
        // computed size must equal the official serializer's output
        // size for both forms, and the choice must follow the oracle
        // (sparse only when strictly smaller).
        for (dtype, numel) in [
            (Dtype::BF16, 200u64),
            (Dtype::F32, 100),
            (Dtype::U8, 257),
            (Dtype::I64, 64),
        ] {
            let shape = [numel];
            for changed in 1..=numel {
                let tensors = [ChangedTensor {
                    name: "w",
                    dtype,
                    shape: &shape,
                    numel,
                    changed,
                }];
                let full = oracle_choice(&tensors, &[Representation::Full]);
                let sparse = oracle_choice(&tensors, &[Representation::Sparse]);
                let expected = if sparse < full {
                    Representation::Sparse
                } else {
                    Representation::Full
                };
                assert_eq!(
                    choose_representations(&tensors).unwrap(),
                    [expected],
                    "{dtype:?} numel={numel} changed={changed}: full={full} sparse={sparse}"
                );
            }
        }
    }

    #[test]
    fn multi_tensor_choice_matches_brute_force_oracle() {
        // Multi-tensor cases: every assignment's computed size must
        // equal the official serializer's, and the chosen assignment
        // must achieve the brute-force minimum size (ties toward
        // full are checked separately at the fixed point).
        let cases: Vec<Vec<ChangedTensor<'_>>> = vec![
            vec![
                ChangedTensor {
                    name: "a.weight",
                    dtype: Dtype::F64,
                    shape: &[37],
                    numel: 37,
                    changed: 5,
                },
                ChangedTensor {
                    name: "w",
                    dtype: Dtype::BF16,
                    shape: &[200],
                    numel: 200,
                    changed: 23,
                },
                ChangedTensor {
                    name: "z.bias",
                    dtype: Dtype::U8,
                    shape: &[1000],
                    numel: 1000,
                    changed: 42,
                },
            ],
            vec![
                ChangedTensor {
                    name: "m.mid",
                    dtype: Dtype::F32,
                    shape: &[8, 8],
                    numel: 64,
                    changed: 9,
                },
                ChangedTensor {
                    name: "a",
                    dtype: Dtype::I32,
                    shape: &[128],
                    numel: 128,
                    changed: 30,
                },
                ChangedTensor {
                    name: "zz",
                    dtype: Dtype::F16,
                    shape: &[333],
                    numel: 333,
                    changed: 17,
                },
                ChangedTensor {
                    name: "b",
                    dtype: Dtype::I64,
                    shape: &[50],
                    numel: 50,
                    changed: 4,
                },
            ],
        ];
        for tensors in &cases {
            let chosen = choose_representations(tensors).unwrap();
            let chosen_len = oracle_choice(tensors, &chosen);
            let mut best = u64::MAX;
            for mask in 0..(1u32 << tensors.len()) {
                let assignment: Vec<Representation> = (0..tensors.len())
                    .map(|i| {
                        if mask & (1 << i) == 0 {
                            Representation::Full
                        } else {
                            Representation::Sparse
                        }
                    })
                    .collect();
                best = best.min(oracle_choice(tensors, &assignment));
            }
            assert_eq!(chosen_len, best, "chosen {chosen:?} is not minimal");
            // Fixed point with ties toward full: no single flip is
            // strictly smaller, and no sparse entry is a tie.
            for i in 0..tensors.len() {
                let mut flipped = chosen.clone();
                flipped[i] = match flipped[i] {
                    Representation::Full => Representation::Sparse,
                    Representation::Sparse => Representation::Full,
                };
                let flipped_len = oracle_choice(tensors, &flipped);
                match chosen[i] {
                    Representation::Full => assert!(flipped_len >= chosen_len),
                    Representation::Sparse => assert!(flipped_len > chosen_len),
                }
            }
        }
    }
}
