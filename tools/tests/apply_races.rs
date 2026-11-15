mod common;

use std::collections::BTreeMap;
use std::fs;
use std::sync::atomic::{AtomicBool, Ordering};

use common::*;
use weightspace_tools::error::ToolError;
use weightspace_tools::ops::{
    ApplyReplacement, BuildReplacement, Cx, Progress, apply_replacement, build_replacement,
};

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

// P4: an I/O fault mid-copy (unreadable model sibling) publishes nothing and leaves no temp entries.
#[cfg(unix)]
#[test]
fn io_fault_mid_copy_publishes_nothing() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let (patch, model, _) = simple_patch(tmp.path());
    // Positive control first: the readable model applies cleanly.
    let out_ok = tmp.path().join("patched-ok");
    apply(&ApplyReplacement {
        patch: patch.clone(),
        model: model.clone(),
        out: out_ok.clone(),
        expect_tensor_sha256: BTreeMap::new(),
    })
    .unwrap();
    assert!(out_ok.join("model.safetensors").exists());
    // chmod 000 faults are meaningless as root (CAP_DAC_OVERRIDE):
    // panic rather than silently skip (consultation 8.2).
    let note = model.join("assets/note.txt");
    fs::set_permissions(&note, fs::Permissions::from_mode(0o000)).unwrap();
    if fs::File::open(&note).is_ok() {
        fs::set_permissions(&note, fs::Permissions::from_mode(0o644)).unwrap();
        panic!("chmod-000 file is readable (running as root?); P4 cannot run here");
    }
    let out = tmp.path().join("patched-fault");
    let err = apply(&ApplyReplacement {
        patch,
        model,
        out: out.clone(),
        expect_tensor_sha256: BTreeMap::new(),
    })
    .err()
    .unwrap();
    fs::set_permissions(&note, fs::Permissions::from_mode(0o644)).unwrap();
    assert!(matches!(err, ToolError::Io { .. }), "{err}");
    assert!(!out.exists());
    let leftovers: Vec<String> = fs::read_dir(tmp.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name.contains(".weightspace-"))
        .collect();
    assert!(leftovers.is_empty(), "temp leftovers: {leftovers:?}");
}
