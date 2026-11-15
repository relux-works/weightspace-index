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
