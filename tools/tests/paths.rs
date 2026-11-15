mod common;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use common::*;
use weightspace_tools::error::ToolError;
use weightspace_tools::ops::{ApplyReplacement, BuildReplacement};

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
