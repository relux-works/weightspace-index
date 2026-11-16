mod common;

use std::collections::BTreeMap;
use std::fs;
use std::sync::atomic::{AtomicBool, Ordering};

use common::*;
use weightspace_tools::error::ToolError;
use weightspace_tools::ops::{
    ApplyReplacement, BuildReplacement, Cx, Phase, Progress, apply_replacement, build_replacement,
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
    assert!(matches!(err, ToolError::NoChanges), "{err}");

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
fn apply_writes_weights_from_verified_snapshot() {
    // M3: verification runs on the base snapshot and the staged
    // weights are written from that same snapshot mapping, so
    // mutating the original base file after its snapshot was taken
    // (here: from the first Verify-phase progress report) cannot
    // change the published output. Renames and inverts
    // apply_snapshot_hash_gate_refuses_base_changed_during_verify:
    // there is no original-file hash gate to refuse at any more,
    // because the original is never re-read.
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
    let mut sink = move |progress: Progress| {
        if progress.phase == Phase::Verify && !mutated.get() {
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
    apply_replacement(
        &ApplyReplacement {
            patch: patch_path,
            model,
            out: out.clone(),
            expect_tensor_sha256: digests,
        },
        &mut cx,
    )
    .unwrap();
    assert_eq!(
        fs::read(out.join("model.safetensors")).unwrap(),
        fs::read(out_ok.join("model.safetensors")).unwrap(),
        "published weights must be the verified snapshot bytes, not the mutated original"
    );
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

#[test]
fn progress_reports_phases_and_bytes() {
    // M5: build and apply report their long phases (with byte
    // counters on the byte-moving ones), not only per-tensor counts.
    let tmp = tempfile::tempdir().unwrap();
    let (base, edited, base_bytes, _) = write_pair(tmp.path());
    let seen = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let seen_sink = std::rc::Rc::clone(&seen);
    let mut sink = move |progress: Progress| {
        seen_sink
            .borrow_mut()
            .push((progress.phase, progress.bytes_done, progress.bytes_total));
    };
    let flag = AtomicBool::new(false);
    let mut cx = Cx::new(&flag, &mut sink);
    let patch_path = tmp.path().join("patch.safetensors");
    build_replacement(
        &BuildReplacement {
            base,
            edited,
            manifest: None,
            out: patch_path.clone(),
        },
        &mut cx,
    )
    .unwrap();
    let phases: Vec<Phase> = seen.borrow().iter().map(|(phase, _, _)| *phase).collect();
    assert!(phases.contains(&Phase::Snapshot), "{phases:?}");
    assert!(phases.contains(&Phase::Plan), "{phases:?}");
    assert!(phases.contains(&Phase::Serialize), "{phases:?}");
    drop(cx);

    seen.borrow_mut().clear();
    let seen_sink = std::rc::Rc::clone(&seen);
    let mut sink = move |progress: Progress| {
        seen_sink
            .borrow_mut()
            .push((progress.phase, progress.bytes_done, progress.bytes_total));
    };
    let mut cx = Cx::new(&flag, &mut sink);
    let model = make_model(tmp.path(), &base_bytes);
    let out = tmp.path().join("patched");
    apply_replacement(
        &ApplyReplacement {
            patch: patch_path.clone(),
            model,
            out,
            expect_tensor_sha256: expected_digests(&patch_path, &base_bytes),
        },
        &mut cx,
    )
    .unwrap();
    let records = seen.borrow();
    let phases: Vec<Phase> = records.iter().map(|(phase, _, _)| *phase).collect();
    for expected in [Phase::Snapshot, Phase::Verify, Phase::Copy, Phase::Hash] {
        assert!(
            phases.contains(&expected),
            "missing {expected:?} in {phases:?}"
        );
    }
    assert!(
        records
            .iter()
            .any(|(phase, done, total)| *phase == Phase::Copy && *total > 0 && *done == *total),
        "copy phase must end at bytes_done == bytes_total: {records:?}"
    );
}

#[test]
fn cancellation_during_copy_phase_publishes_nothing() {
    // M5: cancellation is observed inside the staged-copy phase,
    // not only between tensors.
    let tmp = tempfile::tempdir().unwrap();
    let (patch, model, _base_bytes) = simple_patch(tmp.path());
    let weights = model.join("model.safetensors");
    let inputs = snapshot(&[&weights, &patch]);
    let out = tmp.path().join("patched");
    let flag = AtomicBool::new(false);
    let flag_ref = &flag;
    let mut sink = move |progress: Progress| {
        if progress.phase == Phase::Copy {
            flag_ref.store(true, Ordering::Relaxed);
        }
    };
    let mut cx = Cx::new(&flag, &mut sink);
    let result = apply_replacement(
        &ApplyReplacement {
            patch,
            model,
            out: out.clone(),
            expect_tensor_sha256: BTreeMap::new(),
        },
        &mut cx,
    );
    let err = assert_refusal(result, &out, &inputs);
    assert!(matches!(err, ToolError::Cancelled), "{err}");
}

#[test]
fn cancellation_during_patch_phase_publishes_nothing() {
    // M5 remainder: cancellation is observed inside the sparse
    // patch-write blocks (Phase::Patch), not only between tensors.
    let tmp = tempfile::tempdir().unwrap();
    let (patch, model, _base_bytes) = simple_patch(tmp.path());
    let weights = model.join("model.safetensors");
    let inputs = snapshot(&[&weights, &patch]);
    let out = tmp.path().join("patched");
    let flag = AtomicBool::new(false);
    let flag_ref = &flag;
    let mut sink = move |progress: Progress| {
        if progress.phase == Phase::Patch {
            flag_ref.store(true, Ordering::Relaxed);
        }
    };
    let mut cx = Cx::new(&flag, &mut sink);
    let result = apply_replacement(
        &ApplyReplacement {
            patch,
            model,
            out: out.clone(),
            expect_tensor_sha256: BTreeMap::new(),
        },
        &mut cx,
    );
    let err = assert_refusal(result, &out, &inputs);
    assert!(matches!(err, ToolError::Cancelled), "{err}");
}

#[test]
fn cancellation_during_hash_phase_publishes_nothing() {
    // M5 remainder: cancellation is observed inside the final
    // hash_file pass (Phase::Hash), chunk by chunk.
    let tmp = tempfile::tempdir().unwrap();
    let (patch, model, _base_bytes) = simple_patch(tmp.path());
    let weights = model.join("model.safetensors");
    let inputs = snapshot(&[&weights, &patch]);
    let out = tmp.path().join("patched");
    let flag = AtomicBool::new(false);
    let flag_ref = &flag;
    let mut sink = move |progress: Progress| {
        if progress.phase == Phase::Hash {
            flag_ref.store(true, Ordering::Relaxed);
        }
    };
    let mut cx = Cx::new(&flag, &mut sink);
    let result = apply_replacement(
        &ApplyReplacement {
            patch,
            model,
            out: out.clone(),
            expect_tensor_sha256: BTreeMap::new(),
        },
        &mut cx,
    );
    let err = assert_refusal(result, &out, &inputs);
    assert!(matches!(err, ToolError::Cancelled), "{err}");
}

#[test]
fn cancellation_during_snapshot_phase_publishes_nothing() {
    // M5 remainder: cancellation set from the first chunk-granular
    // snapshot-copy report aborts the snapshot copy itself.
    let tmp = tempfile::tempdir().unwrap();
    let (patch, model, _base_bytes) = simple_patch(tmp.path());
    let weights = model.join("model.safetensors");
    let inputs = snapshot(&[&weights, &patch]);
    let out = tmp.path().join("patched");
    let flag = AtomicBool::new(false);
    let flag_ref = &flag;
    let mut sink = move |progress: Progress| {
        if progress.phase == Phase::Snapshot && progress.bytes_done > 0 {
            flag_ref.store(true, Ordering::Relaxed);
        }
    };
    let mut cx = Cx::new(&flag, &mut sink);
    let result = apply_replacement(
        &ApplyReplacement {
            patch,
            model,
            out: out.clone(),
            expect_tensor_sha256: BTreeMap::new(),
        },
        &mut cx,
    );
    let err = assert_refusal(result, &out, &inputs);
    assert!(matches!(err, ToolError::Cancelled), "{err}");
}

#[test]
fn progress_reports_chunk_bytes_inside_long_phases() {
    // M5 remainder: with a multi-MiB tensor, the snapshot copy,
    // staged copy and final hash all report intermediate
    // (0 < bytes_done < bytes_total) chunk progress, throttled,
    // and the sparse patch phase ends at its byte total.
    use safetensors::Dtype;
    const LEN: usize = 3 * 1024 * 1024;
    let tmp = tempfile::tempdir().unwrap();
    let mut edited_data = vec![0u8; LEN];
    edited_data[17] = 1;
    edited_data[LEN - 3] = 2;
    let base_bytes = st_bytes(&[("big", Dtype::U8, &[LEN], vec![0u8; LEN])]);
    let edited_bytes = st_bytes(&[("big", Dtype::U8, &[LEN], edited_data)]);
    let base = tmp.path().join("base.safetensors");
    let edited = tmp.path().join("edited.safetensors");
    fs::write(&base, &base_bytes).unwrap();
    fs::write(&edited, &edited_bytes).unwrap();

    let seen = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let seen_sink = std::rc::Rc::clone(&seen);
    let mut sink = move |progress: Progress| {
        seen_sink
            .borrow_mut()
            .push((progress.phase, progress.bytes_done, progress.bytes_total));
    };
    let flag = AtomicBool::new(false);
    let mut cx = Cx::new(&flag, &mut sink);
    let patch_path = tmp.path().join("patch.safetensors");
    build_replacement(
        &BuildReplacement {
            base,
            edited,
            manifest: None,
            out: patch_path.clone(),
        },
        &mut cx,
    )
    .unwrap();
    drop(cx);
    let intermediate = |phase: Phase, records: &[(Phase, u64, u64)]| {
        records
            .iter()
            .any(|(p, done, total)| *p == phase && *done > 0 && *done < *total)
    };
    assert!(
        intermediate(Phase::Snapshot, seen.borrow().as_slice()),
        "no intermediate snapshot-copy progress: {:?}",
        seen.borrow()
    );

    seen.borrow_mut().clear();
    let seen_sink = std::rc::Rc::clone(&seen);
    let mut sink = move |progress: Progress| {
        seen_sink
            .borrow_mut()
            .push((progress.phase, progress.bytes_done, progress.bytes_total));
    };
    let mut cx = Cx::new(&flag, &mut sink);
    let model = make_model(tmp.path(), &base_bytes);
    let out = tmp.path().join("patched");
    apply_replacement(
        &ApplyReplacement {
            patch: patch_path,
            model,
            out,
            expect_tensor_sha256: BTreeMap::new(),
        },
        &mut cx,
    )
    .unwrap();
    let records = seen.borrow();
    for phase in [Phase::Snapshot, Phase::Copy, Phase::Hash] {
        assert!(
            intermediate(phase, records.as_slice()),
            "no intermediate {phase:?} progress: {records:?}"
        );
    }
    assert!(
        records
            .iter()
            .any(|(p, done, total)| *p == Phase::Patch && *total > 0 && *done == *total),
        "sparse patch phase must end at bytes_done == bytes_total: {records:?}"
    );
}

/// A 3 MiB single tensor rewritten in every element, so the patch
/// is full-form and apply must hash the whole base tensor during
/// wrong-base verification. Returns (patch, model, edited bytes,
/// expected digests).
fn full_form_fixture(
    dir: &std::path::Path,
) -> (
    std::path::PathBuf,
    std::path::PathBuf,
    Vec<u8>,
    BTreeMap<String, weightspace_tools::files::Sha256Hex>,
) {
    use safetensors::Dtype;
    const LEN: usize = 3 * 1024 * 1024;
    let base_bytes = st_bytes(&[("big", Dtype::U8, &[LEN], vec![0u8; LEN])]);
    let edited_bytes = st_bytes(&[("big", Dtype::U8, &[LEN], vec![0xabu8; LEN])]);
    let base = dir.join("base.safetensors");
    let edited = dir.join("edited.safetensors");
    fs::write(&base, &base_bytes).unwrap();
    fs::write(&edited, &edited_bytes).unwrap();
    let patch_path = dir.join("patch.safetensors");
    build(&BuildReplacement {
        base,
        edited,
        manifest: None,
        out: patch_path.clone(),
    })
    .unwrap();
    let digests = expected_digests(&patch_path, &base_bytes);
    assert!(
        digests.contains_key("big"),
        "fixture patch must be full-form: {digests:?}"
    );
    let model = make_model(dir, &base_bytes);
    (patch_path, model, edited_bytes, digests)
}

#[test]
fn progress_reports_snapshot_hash_and_verify_hash_bytes() {
    // Gap B at the operation level: during one full-form apply the
    // recording sink observes (a) a snapshot-hash byte ramp for the
    // base file — a second Snapshot ramp restarting at 0 for the
    // same file total, with intermediates after the restart (the
    // copy alone produces only the caller's opening 0) — and
    // (b) intermediate Verify byte progress from the wrong-base
    // tensor hash, before any Copy report.
    let tmp = tempfile::tempdir().unwrap();
    let (patch_path, model, edited_bytes, digests) = full_form_fixture(tmp.path());
    let weights = model.join("model.safetensors");
    let base_len = fs::metadata(&weights).unwrap().len();

    let seen = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let seen_sink = std::rc::Rc::clone(&seen);
    let mut sink = move |progress: Progress| {
        seen_sink
            .borrow_mut()
            .push((progress.phase, progress.bytes_done, progress.bytes_total));
    };
    let flag = AtomicBool::new(false);
    let mut cx = Cx::new(&flag, &mut sink);
    let out = tmp.path().join("patched");
    apply_replacement(
        &ApplyReplacement {
            patch: patch_path,
            model,
            out: out.clone(),
            expect_tensor_sha256: digests,
        },
        &mut cx,
    )
    .unwrap();
    assert_eq!(
        fs::read(out.join("model.safetensors")).unwrap(),
        edited_bytes,
        "full-form apply output must equal the edited checkpoint"
    );
    let records = seen.borrow();
    let snapshot_zeros: Vec<usize> = records
        .iter()
        .enumerate()
        .filter(|(_, (p, done, total))| *p == Phase::Snapshot && *done == 0 && *total == base_len)
        .map(|(i, _)| i)
        .collect();
    assert!(
        snapshot_zeros.len() >= 2,
        "snapshot hash must restart the Snapshot ramp for the base file: {records:?}"
    );
    assert!(
        records[snapshot_zeros[1]..].iter().any(|(p, done, total)| {
            *p == Phase::Snapshot && *total == base_len && *done > 0 && *done < *total
        }),
        "no intermediate snapshot-hash progress after the restart: {records:?}"
    );
    let first_copy = records
        .iter()
        .position(|(p, _, _)| *p == Phase::Copy)
        .expect("apply must report Copy");
    assert!(
        records[..first_copy]
            .iter()
            .any(|(p, done, total)| *p == Phase::Verify && *done > 0 && *done < *total),
        "no intermediate verify-hash progress before Copy: {records:?}"
    );
}

#[test]
fn cancellation_during_verify_hash_publishes_nothing() {
    // Gap B: cancellation set from the first intermediate
    // verify-hash report aborts inside the tensor hash; nothing
    // is published and every input stays byte-unchanged.
    let tmp = tempfile::tempdir().unwrap();
    let (patch_path, model, _edited_bytes, digests) = full_form_fixture(tmp.path());
    let weights = model.join("model.safetensors");
    let inputs = snapshot(&[&weights, &patch_path]);
    let out = tmp.path().join("patched");
    let flag = AtomicBool::new(false);
    let flag_ref = &flag;
    let mut sink = move |progress: Progress| {
        if progress.phase == Phase::Verify
            && progress.bytes_done > 0
            && progress.bytes_done < progress.bytes_total
        {
            flag_ref.store(true, Ordering::Relaxed);
        }
    };
    let mut cx = Cx::new(&flag, &mut sink);
    let result = apply_replacement(
        &ApplyReplacement {
            patch: patch_path,
            model,
            out: out.clone(),
            expect_tensor_sha256: digests,
        },
        &mut cx,
    );
    let err = assert_refusal(result, &out, &inputs);
    assert!(matches!(err, ToolError::Cancelled), "{err}");
}
