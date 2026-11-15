//! Slice 1 patch operations on synthetic safetensors fixtures only:
//! round trips, determinism, the manifest gate, and every refusal
//! asserting zero final output plus byte-unchanged inputs.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};
use serde_json::json;
use sha2::{Digest, Sha256};
use weightspace_tools::error::ToolError;
use weightspace_tools::files::Sha256Hex;
use weightspace_tools::ops::{
    ApplyReplacement, BuildReplacement, Cx, Progress, Report, apply_replacement, build_replacement,
};
use weightspace_tools::patch::{
    ChangedTensor, PatchTensor, Representation, choose_representations, parse_patch,
};

type TensorSpec<'a> = (&'a str, Dtype, &'a [usize], Vec<u8>);

fn st_bytes(tensors: &[TensorSpec]) -> Vec<u8> {
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
fn st_bytes_with_metadata(tensors: &[TensorSpec]) -> Vec<u8> {
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
fn file_with_header(header: &[u8], data: &[u8]) -> Vec<u8> {
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header);
    out.extend_from_slice(data);
    out
}

fn u16s(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn u32s(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn i64s(values: &[i64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Base fixture: a 512-element tensor edited in two places (sparse), a
/// fully rewritten small tensor (full), an odd shape, a scalar, an
/// empty tensor, and one unchanged tensor.
fn base_tensors() -> Vec<TensorSpec<'static>> {
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

fn edited_tensors() -> Vec<TensorSpec<'static>> {
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

fn tensor_hashes(bytes: &[u8]) -> BTreeMap<String, String> {
    let parsed = SafeTensors::deserialize(bytes).unwrap();
    parsed
        .tensors()
        .into_iter()
        .map(|(name, view)| (name, hex(&Sha256::digest(view.data()))))
        .collect()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn file_hash(path: &Path) -> String {
    hex(&Sha256::digest(fs::read(path).unwrap()))
}

fn with_cx<R>(
    cancelled: bool,
    f: impl FnOnce(&mut Cx) -> Result<R, ToolError>,
) -> Result<R, ToolError> {
    let flag = AtomicBool::new(cancelled);
    let mut sink = |_: Progress| {};
    let mut cx = Cx::new(&flag, &mut sink);
    f(&mut cx)
}

fn build(req: &BuildReplacement) -> Result<Report, ToolError> {
    with_cx(false, |cx| build_replacement(req, cx))
}

fn apply(req: &ApplyReplacement) -> Result<Report, ToolError> {
    with_cx(false, |cx| apply_replacement(req, cx))
}

/// Expected digests for every full-form tensor in a built patch,
/// computed independently from the base fixture bytes.
fn expected_digests(patch: &Path, base_bytes: &[u8]) -> BTreeMap<String, Sha256Hex> {
    let parsed = parse_patch(patch).unwrap();
    let base_hashes = tensor_hashes(base_bytes);
    parsed
        .tensors
        .iter()
        .filter(|(_, t)| matches!(t, PatchTensor::Full { .. }))
        .map(|(name, _)| (name.clone(), Sha256Hex::parse(&base_hashes[name]).unwrap()))
        .collect()
}

fn write_pair(dir: &Path) -> (PathBuf, PathBuf, Vec<u8>, Vec<u8>) {
    let base_bytes = st_bytes(&base_tensors());
    let edited_bytes = st_bytes(&edited_tensors());
    let base = dir.join("base.safetensors");
    let edited = dir.join("edited.safetensors");
    fs::write(&base, &base_bytes).unwrap();
    fs::write(&edited, &edited_bytes).unwrap();
    (base, edited, base_bytes, edited_bytes)
}

fn make_model(dir: &Path, weights: &[u8]) -> PathBuf {
    let model = dir.join("model");
    fs::create_dir_all(model.join("assets")).unwrap();
    fs::write(model.join("model.safetensors"), weights).unwrap();
    fs::write(model.join("config.json"), "{\"synthetic\":true}\n").unwrap();
    fs::write(model.join("assets/note.txt"), "keep me\n").unwrap();
    model
}

/// Asserts a refusal happened, no final output exists, and every input
/// file is byte-unchanged.
fn assert_refusal<T>(
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

fn snapshot(paths: &[&Path]) -> BTreeMap<PathBuf, String> {
    paths
        .iter()
        .map(|p| (p.to_path_buf(), file_hash(p)))
        .collect()
}

#[test]
fn round_trip_sparse_and_full() {
    let tmp = tempfile::tempdir().unwrap();
    let (base, edited, base_bytes, edited_bytes) = write_pair(tmp.path());
    let patch_path = tmp.path().join("patch.safetensors");
    let report = build(&BuildReplacement {
        base: base.clone(),
        edited,
        manifest: None,
        out: patch_path.clone(),
    })
    .unwrap();
    assert_eq!(report.tensors_changed, 4);
    assert_eq!(report.out_bytes, fs::metadata(&patch_path).unwrap().len());
    assert_eq!(report.out_sha256.as_str(), file_hash(&patch_path));
    weightspace_tools::layout::check_replacement_file(&patch_path).unwrap();

    let parsed = parse_patch(&patch_path).unwrap();
    assert!(matches!(
        parsed.tensors["big.sparse"],
        PatchTensor::Sparse { count: 2, .. }
    ));
    assert!(matches!(parsed.tensors["w.full"], PatchTensor::Full { .. }));
    assert!(matches!(parsed.tensors["scalar"], PatchTensor::Full { .. }));
    assert!(!parsed.tensors.contains_key("w.same"));
    assert!(!parsed.tensors.contains_key("empty"));

    let model = make_model(tmp.path(), &base_bytes);
    let out = tmp.path().join("patched");
    let report = apply(&ApplyReplacement {
        patch: patch_path.clone(),
        model: model.clone(),
        out: out.clone(),
        expect_tensor_sha256: expected_digests(&patch_path, &base_bytes),
    })
    .unwrap();
    assert_eq!(report.tensors_changed, 4);

    let result_bytes = fs::read(out.join("model.safetensors")).unwrap();
    assert_eq!(tensor_hashes(&result_bytes), tensor_hashes(&edited_bytes));
    assert_eq!(result_bytes.len(), base_bytes.len());
    // Every unchanged byte is preserved: the whole header prefix and the
    // unchanged tensors' bytes are identical to the base file.
    let header_len = u64::from_le_bytes(base_bytes[..8].try_into().unwrap()) as usize;
    assert_eq!(
        &result_bytes[..8 + header_len],
        &base_bytes[..8 + header_len]
    );
    assert_eq!(
        fs::read(out.join("config.json")).unwrap(),
        b"{\"synthetic\":true}\n"
    );
    assert_eq!(fs::read(out.join("assets/note.txt")).unwrap(), b"keep me\n");
    assert_eq!(
        report.out_sha256.as_str(),
        file_hash(&out.join("model.safetensors"))
    );
    // Inputs untouched.
    assert_eq!(file_hash(&base), hex(&Sha256::digest(&base_bytes)));
    assert_eq!(
        file_hash(&model.join("model.safetensors")),
        hex(&Sha256::digest(&base_bytes))
    );
}

#[test]
fn determinism() {
    let tmp = tempfile::tempdir().unwrap();
    let (base, edited, _, _) = write_pair(tmp.path());
    let mut outputs = Vec::new();
    for i in 0..2 {
        let out = tmp.path().join(format!("patch{i}.safetensors"));
        build(&BuildReplacement {
            base: base.clone(),
            edited: edited.clone(),
            manifest: None,
            out: out.clone(),
        })
        .unwrap();
        outputs.push(fs::read(out).unwrap());
    }
    assert_eq!(outputs[0], outputs[1]);
}

fn manifest_json(base_bytes: &[u8], edited_bytes: &[u8], names: &[&str]) -> serde_json::Value {
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

#[test]
fn manifest_gate() {
    let tmp = tempfile::tempdir().unwrap();
    let (base, edited, base_bytes, edited_bytes) = write_pair(tmp.path());
    let changed = ["big.sparse", "scalar", "w.full", "w.odd"];

    let good = tmp.path().join("manifest.json");
    fs::write(
        &good,
        serde_json::to_vec(&manifest_json(&base_bytes, &edited_bytes, &changed)).unwrap(),
    )
    .unwrap();
    let out = tmp.path().join("patch.safetensors");
    build(&BuildReplacement {
        base: base.clone(),
        edited: edited.clone(),
        manifest: Some(good),
        out,
    })
    .unwrap();

    let inputs = snapshot(&[&base, &edited]);
    // Wrong edited hash.
    let mut wrong = manifest_json(&base_bytes, &edited_bytes, &changed);
    wrong["tensors"][0]["edited_sha256"] = json!("0".repeat(64));
    let wrong_path = tmp.path().join("wrong.json");
    fs::write(&wrong_path, serde_json::to_vec(&wrong).unwrap()).unwrap();
    let out = tmp.path().join("patch2.safetensors");
    let err = assert_refusal(
        build(&BuildReplacement {
            base: base.clone(),
            edited: edited.clone(),
            manifest: Some(wrong_path),
            out: out.clone(),
        }),
        &out,
        &inputs,
    );
    assert!(matches!(err, ToolError::Manifest { .. }), "{err}");

    // Missing one changed tensor.
    let short = tmp.path().join("short.json");
    fs::write(
        &short,
        serde_json::to_vec(&manifest_json(&base_bytes, &edited_bytes, &changed[..3])).unwrap(),
    )
    .unwrap();
    let out = tmp.path().join("patch3.safetensors");
    let err = assert_refusal(
        build(&BuildReplacement {
            base: base.clone(),
            edited: edited.clone(),
            manifest: Some(short),
            out: out.clone(),
        }),
        &out,
        &inputs,
    );
    assert!(matches!(err, ToolError::Manifest { .. }), "{err}");

    // Extra unchanged tensor listed with correct (equal) hashes.
    let extra_names = ["big.sparse", "scalar", "w.full", "w.odd", "w.same"];
    let extra = tmp.path().join("extra.json");
    fs::write(
        &extra,
        serde_json::to_vec(&manifest_json(&base_bytes, &edited_bytes, &extra_names)).unwrap(),
    )
    .unwrap();
    let out = tmp.path().join("patch4.safetensors");
    let err = assert_refusal(
        build(&BuildReplacement {
            base,
            edited,
            manifest: Some(extra),
            out: out.clone(),
        }),
        &out,
        &inputs,
    );
    assert!(matches!(err, ToolError::Manifest { .. }), "{err}");
}

/// Hand-rolled safetensors bytes with explicit shapes (for fixtures the
/// crate writer cannot produce, such as the legacy `#old` shape).
fn raw_st(entries: &[(&str, &str, &[u64], Vec<u8>)]) -> Vec<u8> {
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

fn simple_patch(dir: &Path) -> (PathBuf, PathBuf, Vec<u8>) {
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

#[test]
fn wrong_sparse_old_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let (patch, model, _) = simple_patch(tmp.path());
    // Rewrite the patch with a lying #old.
    let bad = st_bytes(&[
        ("w#idx", Dtype::I64, &[1], i64s(&[2])),
        ("w#new", Dtype::BF16, &[1], u16s(&[99])),
        ("w#old", Dtype::BF16, &[1], u16s(&[31])),
    ]);
    fs::write(&patch, bad).unwrap();
    let weights = model.join("model.safetensors");
    let inputs = snapshot(&[&weights]);
    let out = tmp.path().join("patched");
    let err = assert_refusal(
        apply(&ApplyReplacement {
            patch,
            model,
            out: out.clone(),
            expect_tensor_sha256: BTreeMap::new(),
        }),
        &out,
        &inputs,
    );
    assert!(matches!(err, ToolError::WrongBase { .. }), "{err}");
}

#[test]
fn full_expected_digest_gate() {
    let tmp = tempfile::tempdir().unwrap();
    let (base, edited, base_bytes, _) = write_pair(tmp.path());
    let patch_path = tmp.path().join("patch.safetensors");
    build(&BuildReplacement {
        base,
        edited,
        manifest: None,
        out: patch_path.clone(),
    })
    .unwrap();
    let model = make_model(tmp.path(), &base_bytes);
    let weights = model.join("model.safetensors");
    let inputs = snapshot(&[&weights]);
    let expected = expected_digests(&patch_path, &base_bytes);
    assert!(!expected.is_empty());

    // Missing digests.
    let out = tmp.path().join("p1");
    let err = assert_refusal(
        apply(&ApplyReplacement {
            patch: patch_path.clone(),
            model: model.clone(),
            out: out.clone(),
            expect_tensor_sha256: BTreeMap::new(),
        }),
        &out,
        &inputs,
    );
    assert!(
        matches!(err, ToolError::MissingExpectedDigest { .. }),
        "{err}"
    );

    // Wrong digest.
    let mut wrong = expected.clone();
    let first = wrong.keys().next().unwrap().clone();
    wrong.insert(first, Sha256Hex::parse(&"0".repeat(64)).unwrap());
    let out = tmp.path().join("p2");
    let err = assert_refusal(
        apply(&ApplyReplacement {
            patch: patch_path.clone(),
            model: model.clone(),
            out: out.clone(),
            expect_tensor_sha256: wrong,
        }),
        &out,
        &inputs,
    );
    assert!(matches!(err, ToolError::WrongBase { .. }), "{err}");

    // Extra digest naming a sparse tensor.
    let mut extra = expected;
    extra.insert(
        "big.sparse".into(),
        Sha256Hex::parse(&tensor_hashes(&base_bytes)["big.sparse"]).unwrap(),
    );
    let out = tmp.path().join("p3");
    let err = assert_refusal(
        apply(&ApplyReplacement {
            patch: patch_path,
            model,
            out: out.clone(),
            expect_tensor_sha256: extra,
        }),
        &out,
        &inputs,
    );
    assert!(matches!(err, ToolError::InvalidArgument { .. }), "{err}");
}

#[test]
fn checkpoint_mismatches_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let base = tmp.path().join("base.safetensors");
    fs::write(&base, st_bytes(&base_tensors())).unwrap();
    let inputs = snapshot(&[&base]);

    let variants: Vec<Vec<u8>> = vec![
        // Shape differs.
        st_bytes(&[("w", Dtype::BF16, &[5], u16s(&[1, 2, 3, 4, 5]))]),
        // Name set differs (extra tensor, one missing).
        st_bytes(&[("other", Dtype::BF16, &[4], u16s(&[1, 2, 3, 4]))]),
    ];
    // A dtype variant needs the full set with one dtype changed.
    let mut dtype_changed = base_tensors();
    dtype_changed[1] = ("w.full", Dtype::F64, &[4], vec![0; 32]);
    let mut variants = variants;
    variants.push(st_bytes(&dtype_changed));

    for (i, edited_bytes) in variants.into_iter().enumerate() {
        let edited = tmp.path().join(format!("edited{i}.safetensors"));
        fs::write(&edited, edited_bytes).unwrap();
        let out = tmp.path().join(format!("patch{i}.safetensors"));
        assert_refusal(
            build(&BuildReplacement {
                base: base.clone(),
                edited,
                manifest: None,
                out: out.clone(),
            }),
            &out,
            &inputs,
        );
    }
}

#[test]
fn malformed_inputs_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let edited = tmp.path().join("edited.safetensors");
    fs::write(&edited, st_bytes(&edited_tensors())).unwrap();

    let truncated = tmp.path().join("truncated.safetensors");
    fs::write(&truncated, [255u8; 8].as_slice()).unwrap();
    let garbage = tmp.path().join("garbage.safetensors");
    let mut bytes = 4u64.to_le_bytes().to_vec();
    bytes.extend_from_slice(b"{bad");
    fs::write(&garbage, &bytes).unwrap();
    // Offsets beyond the data region.
    let overrun = tmp.path().join("overrun.safetensors");
    fs::write(&overrun, raw_st(&[("w", "BF16", &[4], vec![0; 2])])).unwrap();

    for (i, bad) in [&truncated, &garbage, &overrun].iter().enumerate() {
        let inputs = snapshot(&[bad, &edited]);
        let out = tmp.path().join(format!("patch{i}.safetensors"));
        let err = assert_refusal(
            build(&BuildReplacement {
                base: (*bad).clone(),
                edited: edited.clone(),
                manifest: None,
                out: out.clone(),
            }),
            &out,
            &inputs,
        );
        assert!(matches!(err, ToolError::Malformed { .. }), "{err}");
    }
}

#[test]
fn bad_sparse_indices_refused() {
    let tmp = tempfile::tempdir().unwrap();
    for (i, idx) in [&[2i64, 2][..], &[3, 1][..], &[-1, 2][..], &[9][..]]
        .iter()
        .enumerate()
    {
        let dir = tmp.path().join(format!("case{i}"));
        fs::create_dir(&dir).unwrap();
        let base_bytes = st_bytes(&[("w", Dtype::BF16, &[4], u16s(&[10, 20, 30, 40]))]);
        let n = idx.len();
        let patch_bytes = st_bytes(&[
            ("w#idx", Dtype::I64, &[n], i64s(idx)),
            ("w#new", Dtype::BF16, &[n], u16s(&vec![99u16; n])),
            ("w#old", Dtype::BF16, &[n], u16s(&vec![30u16; n])),
        ]);
        let patch = dir.join("patch.safetensors");
        fs::write(&patch, patch_bytes).unwrap();
        let model = make_model(&dir, &base_bytes);
        let weights = model.join("model.safetensors");
        let inputs = snapshot(&[&weights]);
        let out = dir.join("patched");
        assert_refusal(
            apply(&ApplyReplacement {
                patch,
                model,
                out: out.clone(),
                expect_tensor_sha256: BTreeMap::new(),
            }),
            &out,
            &inputs,
        );
    }
}

#[test]
fn existing_output_and_no_change_and_cancellation() {
    let tmp = tempfile::tempdir().unwrap();
    let (base, edited, base_bytes, _) = write_pair(tmp.path());
    let inputs = snapshot(&[&base, &edited]);

    let out = tmp.path().join("patch.safetensors");
    fs::write(&out, b"already here").unwrap();
    let err = build(&BuildReplacement {
        base: base.clone(),
        edited: edited.clone(),
        manifest: None,
        out: out.clone(),
    })
    .err()
    .unwrap();
    assert!(matches!(err, ToolError::OutputExists { .. }), "{err}");
    assert_eq!(fs::read(&out).unwrap(), b"already here");

    // Identical checkpoints: no replacement to write.
    let out = tmp.path().join("patch2.safetensors");
    let err = assert_refusal(
        build(&BuildReplacement {
            base: base.clone(),
            edited: base.clone(),
            manifest: None,
            out: out.clone(),
        }),
        &out,
        &inputs,
    );
    assert!(matches!(err, ToolError::Refused { .. }), "{err}");

    // Cancelled before starting.
    let out = tmp.path().join("patch3.safetensors");
    let err = assert_refusal(
        with_cx(true, |cx| {
            build_replacement(
                &BuildReplacement {
                    base: base.clone(),
                    edited: edited.clone(),
                    manifest: None,
                    out: out.clone(),
                },
                cx,
            )
        }),
        &out,
        &inputs,
    );
    assert!(matches!(err, ToolError::Cancelled), "{err}");

    // Existing apply output directory.
    let patch_path = tmp.path().join("real-patch.safetensors");
    build(&BuildReplacement {
        base: base.clone(),
        edited,
        manifest: None,
        out: patch_path.clone(),
    })
    .unwrap();
    let model = make_model(tmp.path(), &base_bytes);
    let out = tmp.path().join("patched");
    fs::create_dir(&out).unwrap();
    let err = apply(&ApplyReplacement {
        patch: patch_path.clone(),
        model,
        out: out.clone(),
        expect_tensor_sha256: expected_digests(&patch_path, &base_bytes),
    })
    .err()
    .unwrap();
    assert!(matches!(err, ToolError::OutputExists { .. }), "{err}");
}

#[test]
fn output_inside_model_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let (patch, model, _) = simple_patch(tmp.path());
    let weights = model.join("model.safetensors");
    let inputs = snapshot(&[&weights]);
    let out = model.join("patched-inside");
    let err = assert_refusal(
        apply(&ApplyReplacement {
            patch,
            model: model.clone(),
            out: out.clone(),
            expect_tensor_sha256: BTreeMap::new(),
        }),
        &out,
        &inputs,
    );
    assert!(matches!(err, ToolError::Refused { .. }), "{err}");
}

#[test]
fn sharded_model_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let (patch, model, _) = simple_patch(tmp.path());
    fs::write(model.join("model.safetensors.index.json"), "{}").unwrap();
    let out = tmp.path().join("patched");
    let err = apply(&ApplyReplacement {
        patch,
        model,
        out: out.clone(),
        expect_tensor_sha256: BTreeMap::new(),
    })
    .err()
    .unwrap();
    assert!(matches!(err, ToolError::ShardedModel { .. }), "{err}");
    assert!(!out.exists());
}

#[test]
fn legacy_old_shape_one_refused() {
    // Behavior change in the official-reader refactor: the former
    // hand parser tolerated a `#old` declaring shape [1] over bytes
    // holding one value per sparse index. The official reader
    // refuses any entry whose stored bytes do not match its shape,
    // so the alias is gone and this patch is refused with nothing
    // published.
    let tmp = tempfile::tempdir().unwrap();
    let base_bytes = st_bytes(&[("w", Dtype::BF16, &[4], u16s(&[10, 20, 30, 40]))]);
    // #old declares shape [1] but stores both replaced values.
    let patch_bytes = raw_st(&[
        ("w#idx", "I64", &[2], i64s(&[1, 3])),
        ("w#new", "BF16", &[2], u16s(&[21, 41])),
        ("w#old", "BF16", &[1], u16s(&[20, 40])),
    ]);
    let patch = tmp.path().join("patch.safetensors");
    fs::write(&patch, patch_bytes).unwrap();
    let model = make_model(tmp.path(), &base_bytes);
    let weights = model.join("model.safetensors");
    let inputs = snapshot(&[&weights, &patch]);
    let out = tmp.path().join("patched");
    let err = assert_refusal(
        apply(&ApplyReplacement {
            patch,
            model,
            out: out.clone(),
            expect_tensor_sha256: BTreeMap::new(),
        }),
        &out,
        &inputs,
    );
    assert!(matches!(err, ToolError::Malformed { .. }), "{err}");
}

#[cfg(unix)]
#[test]
fn symlink_rules() {
    use std::os::unix::fs::symlink;
    let tmp = tempfile::tempdir().unwrap();
    let (base, edited, base_bytes, _) = write_pair(tmp.path());

    // Read-only input file symlink is accepted (HF snapshot layout).
    let link = tmp.path().join("base-link.safetensors");
    symlink(&base, &link).unwrap();
    let patch_path = tmp.path().join("patch.safetensors");
    build(&BuildReplacement {
        base: link,
        edited: edited.clone(),
        manifest: None,
        out: patch_path.clone(),
    })
    .unwrap();

    // Model whose weights are a symlink to a blob elsewhere.
    let blob = tmp.path().join("blob.bin");
    fs::write(&blob, &base_bytes).unwrap();
    let model = tmp.path().join("snapshot");
    fs::create_dir(&model).unwrap();
    symlink(&blob, model.join("model.safetensors")).unwrap();
    fs::write(model.join("config.json"), "{}").unwrap();
    let out = tmp.path().join("patched");
    apply(&ApplyReplacement {
        patch: patch_path.clone(),
        model: model.clone(),
        out: out.clone(),
        expect_tensor_sha256: expected_digests(&patch_path, &base_bytes),
    })
    .unwrap();
    let copied = out.join("model.safetensors");
    assert!(
        !fs::symlink_metadata(&copied)
            .unwrap()
            .file_type()
            .is_symlink()
    );
    assert_eq!(
        tensor_hashes(&fs::read(&copied).unwrap()),
        tensor_hashes(&st_bytes(&edited_tensors()))
    );

    // Input directory symlink refused.
    let model_link = tmp.path().join("model-link");
    symlink(&model, &model_link).unwrap();
    let out = tmp.path().join("patched2");
    let err = apply(&ApplyReplacement {
        patch: patch_path.clone(),
        model: model_link,
        out: out.clone(),
        expect_tensor_sha256: BTreeMap::new(),
    })
    .err()
    .unwrap();
    assert!(matches!(err, ToolError::SymlinkRefused { .. }), "{err}");
    assert!(!out.exists());

    // Output path itself a symlink refused.
    let out_link = tmp.path().join("out-link.safetensors");
    symlink(&base, &out_link).unwrap();
    let err = build(&BuildReplacement {
        base: base.clone(),
        edited: edited.clone(),
        manifest: None,
        out: out_link,
    })
    .err()
    .unwrap();
    assert!(matches!(err, ToolError::SymlinkRefused { .. }), "{err}");

    // Output under an already-existing symlinked ancestor resolves
    // to the canonical parent (the system-symlinked TMPDIR shape:
    // macOS `/var`, `/tmp`): the build succeeds and the output lands
    // in the real directory, written via the canonical path.
    let real_dir = tmp.path().join("real");
    fs::create_dir(&real_dir).unwrap();
    let dir_link = tmp.path().join("dir-link");
    symlink(&real_dir, &dir_link).unwrap();
    let out = dir_link.join("patch.safetensors");
    build(&BuildReplacement {
        base,
        edited,
        manifest: None,
        out,
    })
    .unwrap();
    assert!(real_dir.join("patch.safetensors").exists());
    assert!(
        !fs::symlink_metadata(real_dir.join("patch.safetensors"))
            .unwrap()
            .file_type()
            .is_symlink()
    );
}

#[test]
fn representation_choice_matches_official_serializer_sizes() {
    // Reviewer's verified flip, end to end: BF16 w[200] with 23
    // changes serializes (official writer, standard metadata) to
    // 504 bytes full vs 508 bytes sparse, so the built replacement
    // must store the full tensor and be 504 bytes.
    let tmp = tempfile::tempdir().unwrap();
    let base_w: Vec<u16> = (0..200).collect();
    let mut edited_w = base_w.clone();
    for value in edited_w.iter_mut().take(23) {
        *value += 1000;
    }
    let base = tmp.path().join("base.safetensors");
    let edited = tmp.path().join("edited.safetensors");
    fs::write(
        &base,
        st_bytes(&[("w", Dtype::BF16, &[200], u16s(&base_w))]),
    )
    .unwrap();
    fs::write(
        &edited,
        st_bytes(&[("w", Dtype::BF16, &[200], u16s(&edited_w))]),
    )
    .unwrap();
    let inputs = snapshot(&[&base, &edited]);

    let full_oracle = st_bytes_with_metadata(&[("w", Dtype::BF16, &[200], u16s(&edited_w))]);
    assert_eq!(full_oracle.len(), 504);
    let idx: Vec<i64> = (0..23).collect();
    let sparse_oracle = st_bytes_with_metadata(&[
        ("w#idx", Dtype::I64, &[23], i64s(&idx)),
        ("w#new", Dtype::BF16, &[23], u16s(&edited_w[..23])),
        ("w#old", Dtype::BF16, &[23], u16s(&base_w[..23])),
    ]);
    assert_eq!(sparse_oracle.len(), 508);

    let patch_path = tmp.path().join("patch.safetensors");
    build(&BuildReplacement {
        base: base.clone(),
        edited: edited.clone(),
        manifest: None,
        out: patch_path.clone(),
    })
    .unwrap();
    assert_eq!(fs::metadata(&patch_path).unwrap().len(), 504);
    let parsed = parse_patch(&patch_path).unwrap();
    assert!(matches!(parsed.tensors["w"], PatchTensor::Full { .. }));
    for (path, before) in &inputs {
        assert_eq!(&file_hash(path), before);
    }

    // Multi-tensor: the same w[200] beside an F32[100] with three
    // changes. Brute-force every assignment with the official
    // writer and require the built file to reach the minimum size
    // with the representations the joint planner reports.
    let base_a: Vec<u32> = (0..100).collect();
    let mut edited_a = base_a.clone();
    for &i in &[0usize, 50, 99] {
        edited_a[i] += 5000;
    }
    let base2 = tmp.path().join("base2.safetensors");
    let edited2 = tmp.path().join("edited2.safetensors");
    fs::write(
        &base2,
        st_bytes(&[
            ("a", Dtype::F32, &[100], u32s(&base_a)),
            ("w", Dtype::BF16, &[200], u16s(&base_w)),
        ]),
    )
    .unwrap();
    fs::write(
        &edited2,
        st_bytes(&[
            ("a", Dtype::F32, &[100], u32s(&edited_a)),
            ("w", Dtype::BF16, &[200], u16s(&edited_w)),
        ]),
    )
    .unwrap();
    let patch2 = tmp.path().join("patch2.safetensors");
    build(&BuildReplacement {
        base: base2,
        edited: edited2,
        manifest: None,
        out: patch2.clone(),
    })
    .unwrap();

    let a_idx: Vec<i64> = vec![0, 50, 99];
    let a_new: Vec<u32> = a_idx.iter().map(|&i| edited_a[i as usize]).collect();
    let a_old: Vec<u32> = a_idx.iter().map(|&i| base_a[i as usize]).collect();
    let assignments = [
        (Representation::Full, Representation::Full),
        (Representation::Full, Representation::Sparse),
        (Representation::Sparse, Representation::Full),
        (Representation::Sparse, Representation::Sparse),
    ];
    let mut oracle_lens = Vec::new();
    for (a_choice, w_choice) in assignments {
        let mut specs: Vec<TensorSpec> = Vec::new();
        match a_choice {
            Representation::Full => {
                specs.push(("a", Dtype::F32, &[100], u32s(&edited_a)));
            }
            Representation::Sparse => {
                specs.push(("a#idx", Dtype::I64, &[3], i64s(&a_idx)));
                specs.push(("a#new", Dtype::F32, &[3], u32s(&a_new)));
                specs.push(("a#old", Dtype::F32, &[3], u32s(&a_old)));
            }
        }
        match w_choice {
            Representation::Full => {
                specs.push(("w", Dtype::BF16, &[200], u16s(&edited_w)));
            }
            Representation::Sparse => {
                specs.push(("w#idx", Dtype::I64, &[23], i64s(&idx)));
                specs.push(("w#new", Dtype::BF16, &[23], u16s(&edited_w[..23])));
                specs.push(("w#old", Dtype::BF16, &[23], u16s(&base_w[..23])));
            }
        }
        oracle_lens.push(st_bytes_with_metadata(&specs).len());
    }
    let built_len = usize::try_from(fs::metadata(&patch2).unwrap().len()).unwrap();
    assert_eq!(built_len, *oracle_lens.iter().min().unwrap());
    let planned = choose_representations(&[
        ChangedTensor {
            name: "a",
            dtype: Dtype::F32,
            shape: &[100],
            numel: 100,
            changed: 3,
        },
        ChangedTensor {
            name: "w",
            dtype: Dtype::BF16,
            shape: &[200],
            numel: 200,
            changed: 23,
        },
    ])
    .unwrap();
    let parsed2 = parse_patch(&patch2).unwrap();
    let built_choices = [
        match parsed2.tensors["a"] {
            PatchTensor::Full { .. } => Representation::Full,
            PatchTensor::Sparse { .. } => Representation::Sparse,
        },
        match parsed2.tensors["w"] {
            PatchTensor::Full { .. } => Representation::Full,
            PatchTensor::Sparse { .. } => Representation::Sparse,
        },
    ];
    assert_eq!(built_choices.as_slice(), planned.as_slice());
}

#[test]
fn duplicate_header_fields_bom_and_invalid_utf8_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let edited = tmp.path().join("edited.safetensors");
    fs::write(&edited, st_bytes(&edited_tensors())).unwrap();

    let valid_header = br#"{"w":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#.to_vec();
    let mut bom = vec![0xef, 0xbb, 0xbf];
    bom.extend_from_slice(&valid_header);
    let variants: Vec<(&str, Vec<u8>)> = vec![
        (
            "duplicate dtype field",
            br#"{"w":{"dtype":"F64","dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#.to_vec(),
        ),
        (
            "duplicate field, escaped spelling",
            br#"{"w":{"dtype":"BF16","d\u0074ype":"BF16","shape":[2],"data_offsets":[0,4]}}"#
                .to_vec(),
        ),
        (
            // The official reader's struct deserialization rejects a
            // duplicated top-level __metadata__ key (the former hand
            // parser accepted it last-wins). A duplicated key
            // *inside* __metadata__ is official-accepted last-wins
            // and is pinned as accepted in tests/checkpoint.rs.
            "duplicate top-level __metadata__",
            br#"{"__metadata__":{"format":"pt"},"__metadata__":{"format":"xx"},"w":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#
                .to_vec(),
        ),
        ("BOM before header", bom),
        (
            "invalid UTF-8 in header",
            b"{\"w\":{\"dtype\":\"BF\xff16\",\"shape\":[2],\"data_offsets\":[0,4]}}".to_vec(),
        ),
    ];
    for (i, (label, header)) in variants.iter().enumerate() {
        let bad = tmp.path().join(format!("bad{i}.safetensors"));
        fs::write(&bad, file_with_header(header, &[0u8; 4])).unwrap();
        let inputs = snapshot(&[&bad, &edited]);
        let out = tmp.path().join(format!("patch{i}.safetensors"));
        let err = assert_refusal(
            build(&BuildReplacement {
                base: bad.clone(),
                edited: edited.clone(),
                manifest: None,
                out: out.clone(),
            }),
            &out,
            &inputs,
        );
        assert!(matches!(err, ToolError::Malformed { .. }), "{label}: {err}");
        if label.starts_with("duplicate") {
            assert!(
                err.to_string().contains("duplicate field"),
                "{label}: {err}"
            );
        }
    }
}

#[test]
fn duplicate_manifest_field_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let (base, edited, _, _) = write_pair(tmp.path());
    let inputs = snapshot(&[&base, &edited]);
    let zeros = "0".repeat(64);
    let manifest = tmp.path().join("manifest.json");
    fs::write(
        &manifest,
        format!(
            r#"{{"tensors":[{{"name":"w","name":"w","shape":[1],"dtype":"BF16","base_sha256":"{zeros}","edited_sha256":"{zeros}"}}]}}"#
        ),
    )
    .unwrap();
    let out = tmp.path().join("patch.safetensors");
    let err = assert_refusal(
        build(&BuildReplacement {
            base,
            edited,
            manifest: Some(manifest),
            out: out.clone(),
        }),
        &out,
        &inputs,
    );
    assert!(matches!(err, ToolError::Manifest { .. }), "{err}");
    assert!(err.to_string().contains("duplicate field"), "{err}");
}

#[test]
fn duplicate_root_tensors_key_refused() {
    // Manifest root duplicates (F3): a duplicated root "tensors"
    // key, plain or escaped, refuses rather than collapsing
    // last-wins. Closure is limited to that known root field.
    use weightspace_tools::patch::read_manifest;
    let tmp = tempfile::tempdir().unwrap();
    for (i, text) in [
        r#"{"tensors":[],"tensors":[]}"#,
        r#"{"tensors":[],"tens\u006frs":[]}"#,
    ]
    .iter()
    .enumerate()
    {
        let manifest = tmp.path().join(format!("dup{i}.json"));
        fs::write(&manifest, text).unwrap();
        let before = file_hash(&manifest);
        let err = read_manifest(&manifest)
            .err()
            .expect("duplicated root tensors key must refuse");
        assert!(matches!(err, ToolError::Manifest { .. }), "{err}");
        assert!(
            err.to_string().contains("duplicate field `tensors`"),
            "{err}"
        );
        assert_eq!(file_hash(&manifest), before, "input mutated");
    }
    // Nearby control: a single root "tensors" key still parses.
    let good = tmp.path().join("good.json");
    fs::write(&good, r#"{"tensors":[]}"#).unwrap();
    assert!(read_manifest(&good).unwrap().tensors.is_empty());
}

#[test]
fn apply_snapshot_hash_gate_refuses_base_changed_during_verify() {
    // The apply snapshot-hash gate (F4): verification runs on the
    // base snapshot; if the base file on disk no longer hashes to
    // that snapshot when the copy is about to be made, apply
    // refuses and publishes nothing. The base file is mutated
    // from the first apply progress report, after verification
    // has begun on the snapshot.
    let tmp = tempfile::tempdir().unwrap();
    let (base, edited, base_bytes, _) = write_pair(tmp.path());
    let patch_path = tmp.path().join("patch.safetensors");
    build(&BuildReplacement {
        base,
        edited,
        manifest: None,
        out: patch_path.clone(),
    })
    .unwrap();
    let model = make_model(tmp.path(), &base_bytes);
    let digests = expected_digests(&patch_path, &base_bytes);

    // Unmutated control succeeds.
    let out_ok = tmp.path().join("patched-ok");
    apply(&ApplyReplacement {
        patch: patch_path.clone(),
        model: model.clone(),
        out: out_ok.clone(),
        expect_tensor_sha256: digests.clone(),
    })
    .unwrap();
    assert!(out_ok.join("model.safetensors").exists());

    let weights = model.join("model.safetensors");
    let weights_for_sink = weights.clone();
    let mutated = std::cell::Cell::new(false);
    let mut sink = move |_: Progress| {
        if !mutated.get() {
            mutated.set(true);
            let mut bytes = fs::read(&weights_for_sink).unwrap();
            let last = bytes.len() - 1;
            bytes[last] = bytes[last].wrapping_add(1);
            fs::write(&weights_for_sink, bytes).unwrap();
        }
    };
    let flag = AtomicBool::new(false);
    let mut cx = Cx::new(&flag, &mut sink);
    let out = tmp.path().join("patched-mutated");
    let err = apply_replacement(
        &ApplyReplacement {
            patch: patch_path,
            model,
            out: out.clone(),
            expect_tensor_sha256: digests,
        },
        &mut cx,
    )
    .err()
    .expect("base changed after snapshot verification must refuse");
    assert!(matches!(err, ToolError::Refused { .. }), "{err}");
    assert!(err.to_string().contains("snapshot-verified"), "{err}");
    assert!(!out.exists(), "refusal published {}", out.display());
}

#[test]
fn gapped_base_refused_through_apply() {
    // A gapped base (F4) is refused at Checkpoint::open inside
    // apply, with nothing published: official contiguous-coverage
    // validation, reached through the whole operation.
    let tmp = tempfile::tempdir().unwrap();
    let (patch, model, _) = simple_patch(tmp.path());
    let weights = model.join("model.safetensors");
    fs::write(
        &weights,
        file_with_header(
            br#"{"w":{"dtype":"BF16","shape":[2],"data_offsets":[4,8]}}"#,
            &[0u8; 8],
        ),
    )
    .unwrap();
    let inputs = snapshot(&[&weights, &patch]);
    let out = tmp.path().join("patched");
    let err = assert_refusal(
        apply(&ApplyReplacement {
            patch,
            model,
            out: out.clone(),
            expect_tensor_sha256: BTreeMap::new(),
        }),
        &out,
        &inputs,
    );
    assert!(matches!(err, ToolError::Malformed { .. }), "{err}");
}

#[test]
fn apply_cancellation_after_start_publishes_nothing() {
    let tmp = tempfile::tempdir().unwrap();
    let (base, edited, base_bytes, _) = write_pair(tmp.path());
    let patch_path = tmp.path().join("patch.safetensors");
    build(&BuildReplacement {
        base,
        edited,
        manifest: None,
        out: patch_path.clone(),
    })
    .unwrap();
    let model = make_model(tmp.path(), &base_bytes);
    let weights = model.join("model.safetensors");
    let inputs = snapshot(&[&weights, &patch_path]);
    let out = tmp.path().join("patched");

    // The flag starts clear and is set from the first progress
    // report, so the operation is cancelled after it has begun
    // verifying, not before it starts.
    let flag = AtomicBool::new(false);
    let flag_ref = &flag;
    let mut sink = move |progress: Progress| {
        if progress.tensors_done >= 1 {
            flag_ref.store(true, Ordering::Relaxed);
        }
    };
    let mut cx = Cx::new(&flag, &mut sink);
    let result = apply_replacement(
        &ApplyReplacement {
            patch: patch_path.clone(),
            model,
            out: out.clone(),
            expect_tensor_sha256: expected_digests(&patch_path, &base_bytes),
        },
        &mut cx,
    );
    let err = assert_refusal(result, &out, &inputs);
    assert!(matches!(err, ToolError::Cancelled), "{err}");
    // No temporary output is left beside the destination either.
    let leftovers: Vec<String> = fs::read_dir(tmp.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name.contains(".weightspace-"))
        .collect();
    assert!(leftovers.is_empty(), "temp leftovers: {leftovers:?}");
}

#[cfg(unix)]
#[test]
fn symlinked_model_dir_spellings_refused() {
    use std::os::unix::fs::symlink;
    let tmp = tempfile::tempdir().unwrap();
    let (patch, model, _) = simple_patch(tmp.path());
    let weights = model.join("model.safetensors");
    let inputs = snapshot(&[&weights]);

    let link = tmp.path().join("model-link");
    symlink(&model, &link).unwrap();
    let nested_parent = tmp.path().join("nested");
    fs::create_dir(&nested_parent).unwrap();
    let nested_link = nested_parent.join("model-link");
    symlink(&model, &nested_link).unwrap();

    let spelled = |path: &Path, suffix: &str| PathBuf::from(format!("{}{suffix}", path.display()));
    let spellings = [
        link.clone(),
        spelled(&link, "/"),
        spelled(&link, "/."),
        nested_link.clone(),
        spelled(&nested_link, "/"),
        spelled(&nested_link, "/."),
    ];
    for (i, model_path) in spellings.into_iter().enumerate() {
        let out = tmp.path().join(format!("patched-spelling{i}"));
        let err = assert_refusal(
            apply(&ApplyReplacement {
                patch: patch.clone(),
                model: model_path.clone(),
                out: out.clone(),
                expect_tensor_sha256: BTreeMap::new(),
            }),
            &out,
            &inputs,
        );
        assert!(
            matches!(err, ToolError::SymlinkRefused { .. }),
            "{}: {err}",
            model_path.display()
        );
    }

    // The same spellings of a real model directory still work: it
    // is the symlink that is refused, not the spelling.
    for (i, suffix) in ["", "/", "/."].iter().enumerate() {
        let out = tmp.path().join(format!("patched-real{i}"));
        apply(&ApplyReplacement {
            patch: patch.clone(),
            model: spelled(&model, suffix),
            out: out.clone(),
            expect_tensor_sha256: BTreeMap::new(),
        })
        .unwrap();
        assert!(out.join("model.safetensors").exists());
    }
}
