mod common;

use std::collections::BTreeMap;
use std::fs;

use common::*;
use serde_json::json;
use weightspace_tools::error::ToolError;
use weightspace_tools::files::Sha256Hex;
use weightspace_tools::ops::{ApplyReplacement, BuildReplacement};

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
