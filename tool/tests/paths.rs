mod common;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use common::*;
use weightspace_tools::error::ToolError;
use weightspace_tools::ops::ApplyReplacement;
#[cfg(unix)]
use weightspace_tools::ops::BuildReplacement;

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
    assert!(matches!(err, ToolError::OutputInsideInput { .. }), "{err}");
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
const ONE_SHARD: &str = "model.safetensors-00001-of-00001.safetensors";

fn one_shard_model(dir: &Path, mapped_to: &str) -> (PathBuf, PathBuf, String) {
    let (patch, model, _) = simple_patch(dir);
    fs::rename(model.join("model.safetensors"), model.join(ONE_SHARD)).unwrap();
    let index = format!("{{\"metadata\":{{}},\"weight_map\":{{\"w\":\"{mapped_to}\"}}}}\n");
    fs::write(model.join("model.safetensors.index.json"), &index).unwrap();
    (patch, model, index)
}

#[test]
fn one_shard_indexed_model_applies() {
    let tmp = tempfile::tempdir().unwrap();
    let (patch, model, index) = one_shard_model(tmp.path(), ONE_SHARD);
    let out = tmp.path().join("patched");
    apply(&ApplyReplacement {
        patch,
        model,
        out: out.clone(),
        expect_tensor_sha256: BTreeMap::new(),
    })
    .unwrap();
    assert!(out.join(ONE_SHARD).is_file());
    assert!(!out.join("model.safetensors").exists());
    assert_eq!(
        fs::read_to_string(out.join("model.safetensors.index.json")).unwrap(),
        index
    );
}

#[test]
fn one_shard_indexed_model_with_sidecar_applies() {
    let tmp = tempfile::tempdir().unwrap();
    let (patch, model, index) = one_shard_model(tmp.path(), ONE_SHARD);
    let sidecar_bytes = b"auxiliary-mtp-weights";
    fs::write(model.join("mtp.safetensors"), sidecar_bytes).unwrap();
    let out = tmp.path().join("patched");
    apply(&ApplyReplacement {
        patch,
        model,
        out: out.clone(),
        expect_tensor_sha256: BTreeMap::new(),
    })
    .unwrap();
    assert!(out.join(ONE_SHARD).is_file());
    assert_eq!(
        fs::read(out.join("mtp.safetensors")).unwrap(),
        sidecar_bytes
    );
    assert_eq!(
        fs::read_to_string(out.join("model.safetensors.index.json")).unwrap(),
        index
    );
}

#[test]
fn index_naming_another_shard_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let (patch, model, _) =
        one_shard_model(tmp.path(), "model.safetensors-00002-of-00002.safetensors");
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

    // SUPERSEDED (X2, consultation 3.3): leaf1b accepted an existing
    // dir-link as the output parent and resolved through it. The
    // OutputTarget contract refuses a symlinked output parent as
    // spelled; only links above the parent are resolved (see
    // `symlinked_ancestor_accepted`). Nothing is written through it.
    let real_dir = tmp.path().join("real");
    fs::create_dir(&real_dir).unwrap();
    let dir_link = tmp.path().join("dir-link");
    symlink(&real_dir, &dir_link).unwrap();
    let out = dir_link.join("patch.safetensors");
    let err = build(&BuildReplacement {
        base,
        edited,
        manifest: None,
        out,
    })
    .err()
    .unwrap();
    assert!(matches!(err, ToolError::SymlinkRefused { .. }), "{err}");
    assert!(!real_dir.join("patch.safetensors").exists());
}
// X5: a symlinked input directory is refused under every spelling of its final component.
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

// X1: output below a symlinked ancestor (var -> private/var shape) is accepted and lands in the real parent.
#[cfg(unix)]
#[test]
fn symlinked_ancestor_accepted() {
    use std::os::unix::fs::symlink;
    let tmp = tempfile::tempdir().unwrap();
    let (base, edited, _, _) = write_pair(tmp.path());
    let real_parent = tmp.path().join("private").join("var").join("parent");
    fs::create_dir_all(&real_parent).unwrap();
    symlink(
        tmp.path().join("private").join("var"),
        tmp.path().join("var"),
    )
    .unwrap();
    let out = tmp
        .path()
        .join("var")
        .join("parent")
        .join("patch.safetensors");
    build(&BuildReplacement {
        base,
        edited,
        manifest: None,
        out,
    })
    .unwrap();
    let landed = real_parent.join("patch.safetensors");
    assert!(landed.exists());
    assert!(
        !fs::symlink_metadata(&landed)
            .unwrap()
            .file_type()
            .is_symlink()
    );
}

// X6: output inside the model reached through a symlink-alias ancestor is refused.
#[cfg(unix)]
#[test]
fn output_inside_model_refused_any_case() {
    use std::os::unix::fs::symlink;
    let tmp = tempfile::tempdir().unwrap();
    let (patch, model, _) = simple_patch(tmp.path());
    let weights = model.join("model.safetensors");
    let inputs = snapshot(&[&weights]);
    // The alias is an ancestor above the output parent (the model
    // dir itself), so the output check passes and containment —
    // canonical parent against canonical model — must refuse it.
    let alias = tmp.path().join("alias");
    symlink(tmp.path(), &alias).unwrap();
    let out = alias
        .join(model.file_name().unwrap())
        .join("patched-inside");
    let err = assert_refusal(
        apply(&ApplyReplacement {
            patch,
            model: model.clone(),
            out: out.clone(),
            expect_tensor_sha256: BTreeMap::new(),
        }),
        &model.join("patched-inside"),
        &inputs,
    );
    assert!(matches!(err, ToolError::OutputInsideInput { .. }), "{err}");
}
