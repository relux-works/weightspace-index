//! The snapshot-backed official reader, exercised directly:
//! descriptor parity against `SafeTensors::read_metadata` on the
//! same fixture bytes, snapshot custody, official-refusal behavior,
//! and a sparse large checkpoint that is never loaded whole.

use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::AtomicBool;

use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};
use sha2::{Digest, Sha256};
use weightspace_tools::checkpoint::Checkpoint;
use weightspace_tools::error::ToolError;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn file_hash(path: &Path) -> String {
    hex(&Sha256::digest(fs::read(path).unwrap()))
}

/// Deterministic stored bytes for one tensor of `dtype`/`shape`.
fn data_for(dtype: Dtype, shape: &[usize], seed: u8) -> Vec<u8> {
    let numel: usize = shape.iter().product();
    let len = numel * (dtype.bitsize() / 8);
    (0..len).map(|i| seed.wrapping_add(i as u8)).collect()
}

fn serialize(specs: &[(&str, Dtype, &[usize])], metadata: bool) -> Vec<u8> {
    let datas: Vec<Vec<u8>> = specs
        .iter()
        .enumerate()
        .map(|(i, (_, dtype, shape))| data_for(*dtype, shape, i as u8))
        .collect();
    let views: Vec<(&str, TensorView<'_>)> = specs
        .iter()
        .zip(&datas)
        .map(|((name, dtype, shape), data)| {
            (
                *name,
                TensorView::new(*dtype, shape.to_vec(), data).unwrap(),
            )
        })
        .collect();
    let metadata = metadata.then(|| {
        let mut m = std::collections::HashMap::new();
        m.insert("format".to_string(), "pt".to_string());
        m.insert("source".to_string(), "synthetic".to_string());
        m
    });
    safetensors::serialize(views, metadata).unwrap()
}

/// Every whole-byte dtype of safetensors 0.8.0 (bitsize % 8 == 0).
const WHOLE_BYTE_DTYPES: [Dtype; 19] = [
    Dtype::BOOL,
    Dtype::U8,
    Dtype::I8,
    Dtype::F8_E5M2,
    Dtype::F8_E4M3,
    Dtype::F8_E8M0,
    Dtype::F8_E4M3FNUZ,
    Dtype::F8_E5M2FNUZ,
    Dtype::I16,
    Dtype::U16,
    Dtype::F16,
    Dtype::BF16,
    Dtype::I32,
    Dtype::U32,
    Dtype::F32,
    Dtype::C64,
    Dtype::F64,
    Dtype::I64,
    Dtype::U64,
];

#[test]
fn official_descriptor_parity() {
    let tmp = tempfile::tempdir().unwrap();
    let cancelled = AtomicBool::new(false);

    // One file holding every supported dtype, plus shape sweeps
    // (scalar, empty, zero-dimension, multi-dimension) on a few.
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();
    let all_names: Vec<String> = WHOLE_BYTE_DTYPES
        .iter()
        .map(|d| format!("t.{}", format!("{d:?}").to_lowercase()))
        .collect();
    let all_specs: Vec<(&str, Dtype, &[usize])> = all_names
        .iter()
        .zip(&WHOLE_BYTE_DTYPES)
        .map(|(name, dtype)| (name.as_str(), *dtype, &[3][..]))
        .collect();
    files.push(("all-dtypes".into(), serialize(&all_specs, true)));
    for (label, shape) in [
        ("scalar", &[][..]),
        ("empty", &[0][..]),
        ("one", &[1][..]),
        ("2d", &[2, 3][..]),
        ("zero-dim", &[3, 0][..]),
        ("zero-mid", &[2, 0, 3][..]),
    ] {
        let specs: Vec<(&str, Dtype, &[usize])> = vec![
            ("w", Dtype::BF16, shape),
            ("x", Dtype::F64, shape),
            ("y", Dtype::U8, shape),
        ];
        files.push((label.into(), serialize(&specs, label == "2d")));
    }

    for (label, bytes) in &files {
        let path = tmp.path().join(format!("{label}.safetensors"));
        fs::write(&path, bytes).unwrap();
        let checkpoint = Checkpoint::open(&path).unwrap();
        // Independent official parse of the same bytes.
        let (header_len, official) = SafeTensors::read_metadata(bytes).unwrap();
        assert_eq!(checkpoint.data_start(), 8 + header_len as u64, "{label}");
        assert_eq!(
            checkpoint.snapshot_sha256().as_str(),
            hex(&Sha256::digest(bytes)),
            "{label}: snapshot hash must be the parsed file's hash"
        );
        let views = SafeTensors::deserialize(bytes).unwrap();
        assert_eq!(
            checkpoint.tensors.len(),
            official.tensors().len(),
            "{label}"
        );
        for (name, meta) in &checkpoint.tensors {
            let info = official.info(name).unwrap();
            assert_eq!(meta.dtype, info.dtype, "{label}:{name}");
            assert_eq!(
                meta.shape,
                info.shape.iter().map(|&d| d as u64).collect::<Vec<_>>(),
                "{label}:{name}"
            );
            assert_eq!(
                meta.data_offsets,
                (info.data_offsets.0 as u64, info.data_offsets.1 as u64),
                "{label}:{name}"
            );
            let expected = hex(&Sha256::digest(views.tensor(name).unwrap().data()));
            assert_eq!(
                checkpoint.tensor_hash(meta, &cancelled).unwrap().as_str(),
                expected,
                "{label}:{name}"
            );
        }
        let expected_meta: std::collections::BTreeMap<String, String> = official
            .metadata()
            .as_ref()
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();
        assert_eq!(checkpoint.metadata, expected_meta, "{label}");
        // The owned official Metadata is retained and agrees.
        assert_eq!(
            checkpoint.official_metadata().tensors().len(),
            checkpoint.tensors.len(),
            "{label}"
        );
    }
}

#[test]
fn snapshot_independence() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("model.safetensors");
    let original = serialize(&[("w", Dtype::BF16, &[4]), ("x", Dtype::F32, &[2])], false);
    fs::write(&path, &original).unwrap();

    let checkpoint = Checkpoint::open(&path).unwrap();
    let cancelled = AtomicBool::new(false);
    let w_meta = checkpoint.tensor("w").unwrap().clone();
    let hash_before = checkpoint.tensor_hash(&w_meta, &cancelled).unwrap();
    let mut w_bytes = vec![0u8; 8];
    checkpoint.read_tensor_at(&w_meta, 0, &mut w_bytes).unwrap();
    assert_eq!(
        checkpoint.snapshot_sha256().as_str(),
        hex(&Sha256::digest(&original))
    );

    // Replace the original path with different valid bytes, then
    // delete it outright: the opened checkpoint must not notice.
    let replacement = serialize(&[("other", Dtype::U8, &[9])], false);
    fs::write(&path, &replacement).unwrap();
    assert_eq!(
        checkpoint.tensor_hash(&w_meta, &cancelled).unwrap(),
        hash_before
    );
    fs::remove_file(&path).unwrap();
    let mut reread = vec![0u8; 8];
    checkpoint.read_tensor_at(&w_meta, 0, &mut reread).unwrap();
    assert_eq!(reread, w_bytes);
    assert_eq!(
        checkpoint.tensor_hash(&w_meta, &cancelled).unwrap(),
        hash_before
    );
    assert!(checkpoint.tensors.contains_key("x"));
    assert!(!checkpoint.tensors.contains_key("other"));
}

fn file_with_header(header: &[u8], data: &[u8]) -> Vec<u8> {
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header);
    out.extend_from_slice(data);
    out
}

#[test]
fn malformed_refused_through_official_reader() {
    let tmp = tempfile::tempdir().unwrap();
    let valid = serialize(&[("w", Dtype::BF16, &[2])], false);
    let mut trailing = valid.clone();
    trailing.push(0);
    let mut bom_header = vec![0xef, 0xbb, 0xbf];
    bom_header.extend_from_slice(br#"{"w":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#);
    let variants: Vec<(&str, Vec<u8>)> = vec![
        ("empty", Vec::new()),
        ("four-bytes", vec![1, 2, 3, 4]),
        ("huge-header-len", {
            let mut v = u64::MAX.to_le_bytes().to_vec();
            v.extend_from_slice(b"{}");
            v
        }),
        ("truncated-data", valid[..valid.len() - 1].to_vec()),
        // The former hand parser accepted gaps and trailing bytes;
        // the official reader refuses both, and so does this tool now.
        ("trailing-byte", trailing),
        (
            "gap-before-tensor",
            file_with_header(
                br#"{"w":{"dtype":"BF16","shape":[2],"data_offsets":[4,8]}}"#,
                &[0u8; 8],
            ),
        ),
        (
            "overlap-not-contiguous",
            file_with_header(
                br#"{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"b":{"dtype":"U8","shape":[4],"data_offsets":[2,6]}}"#,
                &[0u8; 6],
            ),
        ),
        ("bom", file_with_header(&bom_header, &[0u8; 4])),
        (
            "invalid-utf8",
            file_with_header(
                b"{\"w\":{\"dtype\":\"BF\xff16\",\"shape\":[2],\"data_offsets\":[0,4]}}",
                &[0u8; 4],
            ),
        ),
        (
            "duplicate-dtype-field",
            file_with_header(
                br#"{"w":{"dtype":"F64","dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#,
                &[0u8; 4],
            ),
        ),
        (
            "shape-byte-mismatch",
            file_with_header(
                br#"{"w":{"dtype":"BF16","shape":[4],"data_offsets":[0,2]}}"#,
                &[0u8; 2],
            ),
        ),
    ];
    for (label, bytes) in &variants {
        let path = tmp.path().join(format!("{label}.safetensors"));
        fs::write(&path, bytes).unwrap();
        let before = file_hash(&path);
        let err = Checkpoint::open(&path).expect_err("must refuse");
        assert!(matches!(err, ToolError::Malformed { .. }), "{label}: {err}");
        if *label == "duplicate-dtype-field" {
            assert!(
                err.to_string().contains("duplicate field"),
                "{label}: {err}"
            );
        }
        assert_eq!(file_hash(&path), before, "{label}: input mutated");
    }
}

#[test]
fn official_acceptances_pinned() {
    // Acceptance follows the official reader even where the former
    // hand parser was stricter (its class-C duplicate scan): a
    // duplicated key inside __metadata__ and unknown (even
    // duplicated) tensor-entry fields are official-accepted, and a
    // duplicated top-level tensor name resolves last-wins in the
    // crate's hash map. Pinned here so the inheritance is explicit,
    // never accidental.
    let tmp = tempfile::tempdir().unwrap();
    let cases: Vec<(&str, Vec<u8>)> = vec![
        (
            "metadata-inner-duplicate",
            file_with_header(
                br#"{"__metadata__":{"format":"pt","format":"xx"},"w":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#,
                &[1, 2, 3, 4],
            ),
        ),
        (
            "unknown-entry-fields",
            file_with_header(
                br#"{"w":{"dtype":"BF16","shape":[2],"data_offsets":[0,4],"extra":1,"extra":2}}"#,
                &[1, 2, 3, 4],
            ),
        ),
        (
            "duplicate-tensor-name",
            file_with_header(
                br#"{"w":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"w":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#,
                &[1, 2, 3, 4],
            ),
        ),
    ];
    for (label, bytes) in &cases {
        let path = tmp.path().join(format!("{label}.safetensors"));
        fs::write(&path, bytes).unwrap();
        let checkpoint = Checkpoint::open(&path).unwrap();
        assert_eq!(checkpoint.tensors.len(), 1, "{label}");
        if *label == "metadata-inner-duplicate" {
            assert_eq!(checkpoint.metadata["format"], "xx", "{label}: last wins");
        }
    }
}

#[test]
fn sub_byte_dtype_refused() {
    // F4 with 4 elements is official-valid (16 bits over 2 bytes)
    // but its elements are not whole bytes, so element-wise sparse
    // addressing cannot name them: typed refusal, as before.
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("f4.safetensors");
    fs::write(
        &path,
        file_with_header(
            br#"{"w":{"dtype":"F4","shape":[4],"data_offsets":[0,2]}}"#,
            &[0, 0],
        ),
    )
    .unwrap();
    let err = Checkpoint::open(&path).expect_err("must refuse");
    assert!(matches!(err, ToolError::UnsupportedDtype { .. }), "{err}");
}

#[test]
fn sparse_large_checkpoint_streams_over_mapping() {
    // A 128 MiB single-tensor checkpoint as a sparse file on disk
    // (holes read as zeros; no tensor payload is allocated by this
    // test beyond bounded chunks). What this proves, exactly:
    // Checkpoint::open snapshot-copies and maps the file and then
    // reads/hashes it through the mapping in bounded chunks — at no
    // point does the reader path materialize the file (or the
    // tensor) as a heap Vec<u8>, which is a property of the code
    // path (snapshot.rs chunked copy + mmap, checkpoint.rs chunked
    // slice hashing), exercised here end to end at a size where a
    // whole-file Vec would be 128 MiB. It does not measure peak
    // RSS; the snapshot does occupy real disk in the temp dir.
    const LEN: u64 = 128 * 1024 * 1024;
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("big.safetensors");
    let header = format!(r#"{{"big":{{"dtype":"U8","shape":[{LEN}],"data_offsets":[0,{LEN}]}}}}"#);
    let mut file = fs::File::create(&path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header.as_bytes()).unwrap();
    file.set_len(8 + header.len() as u64 + LEN).unwrap();
    drop(file);

    let checkpoint = Checkpoint::open(&path).unwrap();
    let meta = checkpoint.tensor("big").unwrap().clone();
    assert_eq!(meta.byte_len(), LEN);
    assert_eq!(meta.shape, [LEN]);

    // Expected hash of LEN zero bytes, computed by streaming here.
    let mut hasher = Sha256::new();
    let zeros = vec![0u8; 1024 * 1024];
    for _ in 0..(LEN / zeros.len() as u64) {
        hasher.update(&zeros);
    }
    let expected = hex(&hasher.finalize());
    let cancelled = AtomicBool::new(false);
    assert_eq!(
        checkpoint.tensor_hash(&meta, &cancelled).unwrap().as_str(),
        expected
    );
    // Ranged read at the very end of the mapping.
    let mut tail = [9u8; 4];
    checkpoint
        .read_tensor_at(&meta, LEN - 4, &mut tail)
        .unwrap();
    assert_eq!(tail, [0; 4]);
}

#[test]
fn sub_byte_malformed_refused_as_malformed_before_unsupported() {
    // Ordering pinned (F4): a file that is both sub-byte and
    // malformed gets the official reader's Malformed refusal
    // first; the typed UnsupportedDtype fires only for an
    // official-valid sub-byte file (pinned above). Here F4[4]
    // needs 2 stored bytes but the header claims 4.
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("f4-malformed.safetensors");
    fs::write(
        &path,
        file_with_header(
            br#"{"w":{"dtype":"F4","shape":[4],"data_offsets":[0,4]}}"#,
            &[0, 0, 0, 0],
        ),
    )
    .unwrap();
    let before = file_hash(&path);
    let err = Checkpoint::open(&path).expect_err("malformed sub-byte file must refuse");
    assert!(matches!(err, ToolError::Malformed { .. }), "{err}");
    assert!(!matches!(err, ToolError::UnsupportedDtype { .. }), "{err}");
    assert_eq!(file_hash(&path), before, "input mutated");
}

#[test]
fn foreign_descriptors_refused_not_panicking() {
    // Public foreign descriptors (F2): caller-constructed
    // TensorMeta values no crate checkpoint can produce must give
    // typed refusals, never panics (byte_len overflow, division
    // by a zero element size).
    use weightspace_tools::checkpoint::TensorMeta;
    use weightspace_tools::patch::{diff_tensor, for_each_index_block};

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("w.safetensors");
    fs::write(&path, serialize(&[("w", Dtype::BF16, &[4])], false)).unwrap();
    let checkpoint = Checkpoint::open(&path).unwrap();
    let cancelled = AtomicBool::new(false);
    let good = checkpoint.tensor("w").unwrap().clone();

    let inverted = TensorMeta {
        dtype: Dtype::BF16,
        shape: vec![2],
        data_offsets: (5, 2),
    };
    assert_eq!(inverted.byte_len(), 0, "inverted byte_len saturates");
    assert_eq!(good.byte_len(), 8, "crate-produced byte_len unchanged");
    let err = diff_tensor(
        "w",
        &checkpoint,
        &inverted,
        &checkpoint,
        &good,
        &cancelled,
        false,
    )
    .err()
    .expect("inverted base meta must refuse");
    assert!(matches!(err, ToolError::Malformed { .. }), "{err}");
    let err = diff_tensor(
        "w",
        &checkpoint,
        &good,
        &checkpoint,
        &inverted,
        &cancelled,
        false,
    )
    .err()
    .expect("inverted edited meta must refuse");
    assert!(matches!(err, ToolError::Malformed { .. }), "{err}");
    let err = for_each_index_block(&checkpoint, "w", &inverted, &cancelled, |_, _| Ok(()))
        .expect_err("inverted idx meta must refuse");
    assert!(matches!(err, ToolError::Malformed { .. }), "{err}");
    let mut buf = [0u8; 2];
    let err = checkpoint
        .read_tensor_at(&inverted, 0, &mut buf)
        .expect_err("inverted read must refuse");
    assert!(matches!(err, ToolError::Malformed { .. }), "{err}");

    let sub_byte = TensorMeta {
        dtype: Dtype::F4,
        shape: vec![4],
        data_offsets: (0, 2),
    };
    assert_eq!(sub_byte.elem_bytes(), 0);
    assert_eq!(sub_byte.byte_len(), 2);
    let err = diff_tensor(
        "w",
        &checkpoint,
        &sub_byte,
        &checkpoint,
        &sub_byte,
        &cancelled,
        false,
    )
    .err()
    .expect("sub-byte meta must refuse");
    assert!(matches!(err, ToolError::UnsupportedDtype { .. }), "{err}");
    let err = for_each_index_block(&checkpoint, "w", &sub_byte, &cancelled, |_, _| Ok(()))
        .expect_err("sub-byte idx meta must refuse");
    assert!(matches!(err, ToolError::UnsupportedDtype { .. }), "{err}");
}

// M1: after Checkpoint::open, replacing the original with a different valid file of equal size does not change any tensor read.
#[test]
fn parse_sees_snapshot_not_original() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("model.safetensors");
    let original = serialize(&[("w", Dtype::BF16, &[4]), ("x", Dtype::F32, &[2])], false);
    fs::write(&path, &original).unwrap();

    let checkpoint = Checkpoint::open(&path).unwrap();
    let cancelled = AtomicBool::new(false);
    let w_meta = checkpoint.tensor("w").unwrap().clone();
    let x_meta = checkpoint.tensor("x").unwrap().clone();
    let w_hash = checkpoint.tensor_hash(&w_meta, &cancelled).unwrap();
    let x_hash = checkpoint.tensor_hash(&x_meta, &cancelled).unwrap();
    let snapshot_sha = checkpoint.snapshot_sha256().clone();

    // A different valid file of exactly the same size (same header
    // shape, every stored byte different) takes the original's
    // place; a reader mapping the original in place would now
    // parse and read these bytes instead.
    let mut replacement = original.clone();
    let data_start = checkpoint.data_start() as usize;
    for byte in &mut replacement[data_start..] {
        *byte = byte.wrapping_add(1);
    }
    assert_eq!(replacement.len(), original.len());
    assert_ne!(replacement, original);
    // The replacement is itself official-valid.
    SafeTensors::read_metadata(&replacement).unwrap();
    fs::write(&path, &replacement).unwrap();

    assert_eq!(checkpoint.tensor_hash(&w_meta, &cancelled).unwrap(), w_hash);
    assert_eq!(checkpoint.tensor_hash(&x_meta, &cancelled).unwrap(), x_hash);
    assert_eq!(checkpoint.snapshot_sha256(), &snapshot_sha);
}

// M3: truncating the original to 0 after open succeeds and all reads return the old bytes.
#[test]
fn truncating_original_after_open_is_harmless() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("model.safetensors");
    let original = serialize(&[("w", Dtype::BF16, &[4])], false);
    fs::write(&path, &original).unwrap();

    let checkpoint = Checkpoint::open(&path).unwrap();
    let cancelled = AtomicBool::new(false);
    let w_meta = checkpoint.tensor("w").unwrap().clone();
    let hash_before = checkpoint.tensor_hash(&w_meta, &cancelled).unwrap();
    let mut bytes_before = vec![0u8; 8];
    checkpoint
        .read_tensor_at(&w_meta, 0, &mut bytes_before)
        .unwrap();

    // Succeeds (on Windows truncating a mapped file would fail,
    // so success there also proves the original is not mapped)
    // and the opened checkpoint does not notice.
    fs::File::create(&path).unwrap().set_len(0).unwrap();
    assert_eq!(fs::metadata(&path).unwrap().len(), 0);
    assert_eq!(
        checkpoint.tensor_hash(&w_meta, &cancelled).unwrap(),
        hash_before
    );
    let mut bytes_after = vec![0u8; 8];
    checkpoint
        .read_tensor_at(&w_meta, 0, &mut bytes_after)
        .unwrap();
    assert_eq!(bytes_after, bytes_before);
}

// Gap B: tensor_hash_with_progress reports per-chunk bytes (0 start,
// intermediates, tensor total at the end), agrees with the plain
// hash and an independent SHA-256, and a cancellation set from the
// sink aborts the hash with nothing returned.
#[test]
fn tensor_hash_with_progress_reports_chunks_and_cancels() {
    use std::sync::atomic::Ordering;
    const LEN: usize = 1024 * 1024 + 12345;
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("model.safetensors");
    fs::write(&path, serialize(&[("big", Dtype::U8, &[LEN])], false)).unwrap();
    let before = file_hash(&path);

    let checkpoint = Checkpoint::open(&path).unwrap();
    let meta = checkpoint.tensor("big").unwrap().clone();
    assert_eq!(meta.byte_len(), LEN as u64);

    let cancelled = AtomicBool::new(false);
    let mut records: Vec<(u64, u64)> = Vec::new();
    let hashed = checkpoint
        .tensor_hash_with_progress(&meta, &cancelled, &mut |done, total| {
            records.push((done, total));
        })
        .unwrap();
    assert_eq!(
        hashed,
        checkpoint.tensor_hash(&meta, &cancelled).unwrap(),
        "progress form must agree with the plain hash"
    );
    let mut data = vec![0u8; LEN];
    checkpoint.read_tensor_at(&meta, 0, &mut data).unwrap();
    assert_eq!(hashed.as_str(), hex(&Sha256::digest(&data)));
    assert_eq!(records.first(), Some(&(0, LEN as u64)), "{records:?}");
    assert_eq!(
        records.last(),
        Some(&(LEN as u64, LEN as u64)),
        "{records:?}"
    );
    assert!(
        records
            .iter()
            .any(|&(done, total)| done > 0 && done < total),
        "no intermediate tensor-hash progress: {records:?}"
    );
    assert!(
        records.windows(2).all(|w| w[0].0 < w[1].0),
        "tensor-hash ramp must be strictly increasing: {records:?}"
    );
    assert!(
        records.iter().all(|&(_, total)| total == LEN as u64),
        "every report must carry the tensor total: {records:?}"
    );

    let flag = AtomicBool::new(false);
    let flag_ref = &flag;
    let err = checkpoint
        .tensor_hash_with_progress(&meta, &flag, &mut |done, total| {
            if done > 0 && done < total {
                flag_ref.store(true, Ordering::Relaxed);
            }
        })
        .expect_err("cancelled tensor hash must be refused");
    assert!(matches!(err, ToolError::Cancelled), "{err}");
    assert_eq!(file_hash(&path), before, "input mutated");
}
