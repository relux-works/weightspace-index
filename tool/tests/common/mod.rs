#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use serde_json::Value;
use weightspace_tools::records::{Records, read_json};

use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;

use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};
use serde_json::json;
use sha2::{Digest, Sha256};
use weightspace_tools::error::ToolError;
use weightspace_tools::files::Sha256Hex;
use weightspace_tools::ops::{
    ApplyReplacement, BuildReplacement, Cx, Progress, Report, apply_replacement, build_replacement,
};
use weightspace_tools::patch::{PatchTensor, parse_patch};

pub fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

static RECORDS: LazyLock<Records> =
    LazyLock::new(|| Records::load(&repo().join("schema/records.schema.json")).unwrap());

/// The compiled schema, built once per test binary.
pub fn records() -> &'static Records {
    &RECORDS
}

pub fn example_models() -> PathBuf {
    repo().join("examples/models")
}

/// The synthetic example base, relative to a models/ root.
pub const BASE: &str = "example-org/Example-Model-1B";

pub fn example(rel: &str) -> Value {
    read(&example_models().join(BASE).join(rel))
}

pub fn read(path: &Path) -> Value {
    read_json(path).unwrap()
}

pub fn write(path: &Path, v: &Value) {
    fs::write(path, serde_json::to_vec_pretty(v).unwrap()).unwrap();
}

pub fn edit(path: &Path, f: impl FnOnce(&mut Value)) {
    let mut v = read(path);
    f(&mut v);
    write(path, &v);
}

pub fn remove(v: &mut Value, key: &str) {
    v.as_object_mut().unwrap().remove(key);
}

pub fn copy_dir(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for entry in fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), target).unwrap();
        }
    }
}

pub type TensorSpec<'a> = (&'a str, Dtype, &'a [usize], Vec<u8>);

pub fn st_bytes(tensors: &[TensorSpec]) -> Vec<u8> {
    let views: Vec<(&str, TensorView<'_>)> = tensors
        .iter()
        .map(|(name, dtype, shape, data)| {
            (
                *name,
                TensorView::new(*dtype, shape.to_vec(), data).unwrap(),
            )
        })
        .collect();
    safetensors::serialize(views, None).unwrap()
}

/// Like `st_bytes`, but with the `__metadata__ {"format":"pt"}` entry
/// the replacement writer always emits, so byte lengths are directly
/// comparable with a built replacement file.
pub fn st_bytes_with_metadata(tensors: &[TensorSpec]) -> Vec<u8> {
    let views: Vec<(&str, TensorView<'_>)> = tensors
        .iter()
        .map(|(name, dtype, shape, data)| {
            (
                *name,
                TensorView::new(*dtype, shape.to_vec(), data).unwrap(),
            )
        })
        .collect();
    let mut metadata = std::collections::HashMap::new();
    metadata.insert("format".to_string(), "pt".to_string());
    safetensors::serialize(views, Some(metadata)).unwrap()
}

/// A safetensors file from raw header bytes (for headers the crate
/// writer cannot produce: duplicate fields, BOM, invalid UTF-8).
pub fn file_with_header(header: &[u8], data: &[u8]) -> Vec<u8> {
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header);
    out.extend_from_slice(data);
    out
}

pub fn u16s(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub fn u32s(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub fn i64s(values: &[i64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Base fixture: a 512-element tensor edited in two places (sparse), a
/// fully rewritten small tensor (full), an odd shape, a scalar, an
/// empty tensor, and one unchanged tensor.
pub fn base_tensors() -> Vec<TensorSpec<'static>> {
    let big: Vec<u16> = (0..512).map(|i| (i * 7 + 1) as u16).collect();
    vec![
        ("big.sparse", Dtype::BF16, &[512], u16s(&big)),
        ("w.full", Dtype::F32, &[4], u32s(&[10, 20, 30, 40])),
        ("w.odd", Dtype::I32, &[2, 3], u32s(&[1, 2, 3, 4, 5, 6])),
        ("scalar", Dtype::F32, &[], u32s(&[99])),
        ("empty", Dtype::F32, &[0], Vec::new()),
        ("w.same", Dtype::U8, &[3], vec![7, 8, 9]),
    ]
}

pub fn edited_tensors() -> Vec<TensorSpec<'static>> {
    let mut big: Vec<u16> = (0..512).map(|i| (i * 7 + 1) as u16).collect();
    big[5] = 0x7fff;
    big[500] = 0x0001;
    vec![
        ("big.sparse", Dtype::BF16, &[512], u16s(&big)),
        ("w.full", Dtype::F32, &[4], u32s(&[11, 21, 31, 41])),
        ("w.odd", Dtype::I32, &[2, 3], u32s(&[1, 2, 3, 4, 55, 6])),
        ("scalar", Dtype::F32, &[], u32s(&[100])),
        ("empty", Dtype::F32, &[0], Vec::new()),
        ("w.same", Dtype::U8, &[3], vec![7, 8, 9]),
    ]
}

pub fn tensor_hashes(bytes: &[u8]) -> BTreeMap<String, String> {
    let parsed = SafeTensors::deserialize(bytes).unwrap();
    parsed
        .tensors()
        .into_iter()
        .map(|(name, view)| (name, hex(&Sha256::digest(view.data()))))
        .collect()
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn file_hash(path: &Path) -> String {
    hex(&Sha256::digest(fs::read(path).unwrap()))
}

pub fn with_cx<R>(
    cancelled: bool,
    f: impl FnOnce(&mut Cx) -> Result<R, ToolError>,
) -> Result<R, ToolError> {
    let flag = AtomicBool::new(cancelled);
    let mut sink = |_: Progress| {};
    let mut cx = Cx::new(&flag, &mut sink);
    f(&mut cx)
}

pub fn build(req: &BuildReplacement) -> Result<Report, ToolError> {
    with_cx(false, |cx| build_replacement(req, cx))
}

pub fn apply(req: &ApplyReplacement) -> Result<Report, ToolError> {
    with_cx(false, |cx| apply_replacement(req, cx))
}

/// Expected digests for every full-form tensor in a built patch,
/// computed independently from the base fixture bytes.
pub fn expected_digests(patch: &Path, base_bytes: &[u8]) -> BTreeMap<String, Sha256Hex> {
    let parsed = parse_patch(patch).unwrap();
    let base_hashes = tensor_hashes(base_bytes);
    parsed
        .tensors
        .iter()
        .filter(|(_, t)| matches!(t, PatchTensor::Full { .. }))
        .map(|(name, _)| (name.clone(), Sha256Hex::parse(&base_hashes[name]).unwrap()))
        .collect()
}

pub fn write_pair(dir: &Path) -> (PathBuf, PathBuf, Vec<u8>, Vec<u8>) {
    let base_bytes = st_bytes(&base_tensors());
    let edited_bytes = st_bytes(&edited_tensors());
    let base = dir.join("base.safetensors");
    let edited = dir.join("edited.safetensors");
    fs::write(&base, &base_bytes).unwrap();
    fs::write(&edited, &edited_bytes).unwrap();
    (base, edited, base_bytes, edited_bytes)
}

pub fn make_model(dir: &Path, weights: &[u8]) -> PathBuf {
    let model = dir.join("model");
    fs::create_dir_all(model.join("assets")).unwrap();
    fs::write(model.join("model.safetensors"), weights).unwrap();
    fs::write(model.join("config.json"), "{\"synthetic\":true}\n").unwrap();
    fs::write(model.join("assets/note.txt"), "keep me\n").unwrap();
    model
}

/// Asserts a refusal happened, no final output exists, and every input
/// file is byte-unchanged.
pub fn assert_refusal<T>(
    result: Result<T, ToolError>,
    out: &Path,
    inputs: &BTreeMap<PathBuf, String>,
) -> ToolError {
    assert!(!out.exists(), "refusal published {}", out.display());
    for (path, before) in inputs {
        assert_eq!(&file_hash(path), before, "{} changed", path.display());
    }
    result.err().expect("operation must refuse")
}

pub fn snapshot(paths: &[&Path]) -> BTreeMap<PathBuf, String> {
    paths
        .iter()
        .map(|p| (p.to_path_buf(), file_hash(p)))
        .collect()
}

pub fn manifest_json(base_bytes: &[u8], edited_bytes: &[u8], names: &[&str]) -> serde_json::Value {
    let base_parsed = SafeTensors::deserialize(base_bytes).unwrap();
    let base_hashes = tensor_hashes(base_bytes);
    let edited_hashes = tensor_hashes(edited_bytes);
    let tensors: Vec<_> = names
        .iter()
        .map(|name| {
            let view = base_parsed.tensor(name).unwrap();
            json!({
                "name": name,
                "shape": view.shape(),
                "dtype": view.dtype().to_string(),
                "base_sha256": base_hashes[*name],
                "edited_sha256": edited_hashes[*name],
            })
        })
        .collect();
    json!({ "tensors": tensors })
}

/// Hand-rolled safetensors bytes with explicit shapes (for fixtures the
/// crate writer cannot produce, such as the legacy `#old` shape).
pub fn raw_st(entries: &[(&str, &str, &[u64], Vec<u8>)]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, dtype, shape, bytes) in entries {
        let start = data.len();
        data.extend_from_slice(bytes);
        header.insert(
            name.to_string(),
            json!({"dtype": dtype, "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend(header);
    out.extend(data);
    out
}

pub fn simple_patch(dir: &Path) -> (PathBuf, PathBuf, Vec<u8>) {
    // One sparse tensor over a BF16[4] base, changing flat index 2.
    let base_bytes = st_bytes(&[("w", Dtype::BF16, &[4], u16s(&[10, 20, 30, 40]))]);
    let patch_bytes = st_bytes(&[
        ("w#idx", Dtype::I64, &[1], i64s(&[2])),
        ("w#new", Dtype::BF16, &[1], u16s(&[99])),
        ("w#old", Dtype::BF16, &[1], u16s(&[30])),
    ]);
    let patch = dir.join("patch.safetensors");
    fs::write(&patch, patch_bytes).unwrap();
    let model = make_model(dir, &base_bytes);
    (patch, model, base_bytes)
}
