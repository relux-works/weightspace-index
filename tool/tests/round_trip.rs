//! Slice 1 patch operations on synthetic safetensors fixtures only:
//! round trips, determinism, the manifest gate, and every refusal
//! asserting zero final output plus byte-unchanged inputs.

mod common;

use std::fs;

use common::*;
use sha2::{Digest, Sha256};
use weightspace_tools::ops::{ApplyReplacement, BuildReplacement};
use weightspace_tools::patch::{PatchTensor, parse_patch};

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
    weightspace_tools::patch::check_replacement_file(&patch_path).unwrap();

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

#[test]
fn full_form_streaming_writer_byte_identical_to_official_serializer() {
    // M4: the streaming writer (zero-copy full-form views over the
    // snapshot mapping, serialized straight into the temp handle)
    // must produce exactly the bytes `safetensors::serialize`
    // produces for the same entries, and the result must round-trip.
    // The fixture makes the representations unambiguous: a 512-wide
    // tensor with two changes (sparse), a 4-wide tensor with every
    // element changed (full) and a changed scalar (full).
    use safetensors::Dtype;
    let tmp = tempfile::tempdir().unwrap();
    let big_base: Vec<u16> = (0..512).map(|i| (i * 7 + 1) as u16).collect();
    let mut big_edited = big_base.clone();
    big_edited[5] = 0x7fff;
    big_edited[500] = 0x0001;
    let base_bytes = st_bytes(&[
        ("big.sparse", Dtype::BF16, &[512], u16s(&big_base)),
        ("w.full", Dtype::F32, &[4], u32s(&[10, 20, 30, 40])),
        ("scalar", Dtype::F32, &[], u32s(&[99])),
    ]);
    let edited_bytes = st_bytes(&[
        ("big.sparse", Dtype::BF16, &[512], u16s(&big_edited)),
        ("w.full", Dtype::F32, &[4], u32s(&[11, 21, 31, 41])),
        ("scalar", Dtype::F32, &[], u32s(&[100])),
    ]);
    let base = tmp.path().join("base.safetensors");
    let edited = tmp.path().join("edited.safetensors");
    fs::write(&base, &base_bytes).unwrap();
    fs::write(&edited, &edited_bytes).unwrap();
    let patch_path = tmp.path().join("patch.safetensors");
    let report = build(&BuildReplacement {
        base: base.clone(),
        edited,
        manifest: None,
        out: patch_path.clone(),
    })
    .unwrap();
    // Independent oracle: the official serializer over the entries
    // the format must contain, with the writer's one metadata key.
    let expected = st_bytes_with_metadata(&[
        ("big.sparse#idx", Dtype::I64, &[2], i64s(&[5, 500])),
        ("big.sparse#new", Dtype::BF16, &[2], u16s(&[0x7fff, 0x0001])),
        (
            "big.sparse#old",
            Dtype::BF16,
            &[2],
            u16s(&[big_base[5], big_base[500]]),
        ),
        ("w.full", Dtype::F32, &[4], u32s(&[11, 21, 31, 41])),
        ("scalar", Dtype::F32, &[], u32s(&[100])),
    ]);
    let actual = fs::read(&patch_path).unwrap();
    assert_eq!(
        actual, expected,
        "streaming writer bytes differ from safetensors::serialize"
    );
    assert_eq!(report.out_sha256.as_str(), file_hash(&patch_path));
    assert_eq!(report.out_bytes, actual.len() as u64);

    // And the byte-identical file still round-trips to the edited
    // tensor contents.
    let model = make_model(tmp.path(), &base_bytes);
    let out = tmp.path().join("patched");
    apply(&ApplyReplacement {
        patch: patch_path.clone(),
        model,
        out: out.clone(),
        expect_tensor_sha256: expected_digests(&patch_path, &base_bytes),
    })
    .unwrap();
    assert_eq!(
        tensor_hashes(&fs::read(out.join("model.safetensors")).unwrap()),
        tensor_hashes(&edited_bytes)
    );
}

#[test]
fn corrupted_replacement_refused_by_streaming_gate() {
    // M4 / 7.3 item 7: the relocated gate parses through the
    // official reader and streams the `#idx` checks; every
    // corruption the former hand parser refused is still refused,
    // and the valid control still passes.
    use safetensors::Dtype;
    let tmp = tempfile::tempdir().unwrap();
    let valid = st_bytes(&[
        ("w#idx", Dtype::I64, &[2], i64s(&[1, 3])),
        ("w#new", Dtype::BF16, &[2], u16s(&[7, 8])),
        ("w#old", Dtype::BF16, &[2], u16s(&[9, 10])),
    ]);
    let cases: Vec<(&str, Vec<u8>, &str)> = vec![
        ("valid-sparse", valid.clone(), ""),
        (
            "missing-old",
            st_bytes(&[
                ("w#idx", Dtype::I64, &[1], i64s(&[1])),
                ("w#new", Dtype::BF16, &[1], u16s(&[7])),
            ]),
            "sparse tensors need #idx, #new and #old",
        ),
        (
            "unsorted-idx",
            st_bytes(&[
                ("w#idx", Dtype::I64, &[2], i64s(&[3, 1])),
                ("w#new", Dtype::BF16, &[2], u16s(&[7, 8])),
                ("w#old", Dtype::BF16, &[2], u16s(&[9, 10])),
            ]),
            "indices must be sorted, unique and non-negative",
        ),
        (
            "duplicate-idx",
            st_bytes(&[
                ("w#idx", Dtype::I64, &[2], i64s(&[1, 1])),
                ("w#new", Dtype::BF16, &[2], u16s(&[7, 8])),
                ("w#old", Dtype::BF16, &[2], u16s(&[9, 10])),
            ]),
            "indices must be sorted, unique and non-negative",
        ),
        (
            "negative-idx",
            st_bytes(&[
                ("w#idx", Dtype::I64, &[2], i64s(&[-1, 3])),
                ("w#new", Dtype::BF16, &[2], u16s(&[7, 8])),
                ("w#old", Dtype::BF16, &[2], u16s(&[9, 10])),
            ]),
            "indices must be sorted, unique and non-negative",
        ),
        (
            "unknown-suffix",
            st_bytes(&[("w#bogus", Dtype::BF16, &[1], u16s(&[7]))]),
            "unknown suffix #bogus",
        ),
        (
            "full-and-sparse",
            st_bytes(&[
                ("w", Dtype::BF16, &[1], u16s(&[7])),
                ("w#idx", Dtype::I64, &[1], i64s(&[0])),
                ("w#new", Dtype::BF16, &[1], u16s(&[8])),
                ("w#old", Dtype::BF16, &[1], u16s(&[9])),
            ]),
            "both full and sparse",
        ),
        (
            "truncated-data",
            valid[..valid.len() - 1].to_vec(),
            "malformed safetensors",
        ),
    ];
    for (name, bytes, needle) in cases {
        let path = tmp.path().join(format!("{name}.safetensors"));
        fs::write(&path, &bytes).unwrap();
        let result = weightspace_tools::patch::check_replacement_file(&path);
        if needle.is_empty() {
            result.unwrap_or_else(|e| panic!("{name} must pass the gate: {e}"));
        } else {
            let err = result.expect_err(name);
            assert!(err.contains(needle), "{name}: {err} lacks {needle:?}");
        }
    }
}
