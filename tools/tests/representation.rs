mod common;

use std::collections::BTreeMap;
use std::fs;

use common::*;
use safetensors::Dtype;
use weightspace_tools::error::ToolError;
use weightspace_tools::ops::{ApplyReplacement, BuildReplacement};
use weightspace_tools::patch::{
    ChangedTensor, PatchTensor, Representation, choose_representations, parse_patch,
};

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
