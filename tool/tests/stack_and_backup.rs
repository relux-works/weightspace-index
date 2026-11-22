//! Integration tests for multi-patch replacement stacking (`--on-conflict`),
//! `--in-place` SHA-256-verified backups, and `restore`.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::atomic::AtomicBool;

use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};
use sha2::{Digest, Sha256};

use weightspace_tools::error::ToolError;
use weightspace_tools::ops::stack::{BACKUP_DIR, STATE_FILE};
use weightspace_tools::ops::{
    ApplyStack, BuildReplacement, ConflictPolicy, Cx, PatchSpec, Progress, apply_stack,
    build_replacement, read_state, restore_backup,
};

fn cx_run<T>(f: impl FnOnce(&mut Cx) -> T) -> T {
    let cancelled = AtomicBool::new(false);
    let mut sink = |_: Progress| {};
    let mut cx = Cx::new(&cancelled, &mut sink);
    f(&mut cx)
}

fn bf16_bits(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bf16_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter()
        .flat_map(|v| bf16_bits(*v).to_le_bytes())
        .collect()
}

fn make_checkpoint(w_f32: &[f32], w_bf16: &[f32]) -> Vec<u8> {
    let f32_raw = f32_bytes(w_f32);
    let bf16_raw = bf16_bytes(w_bf16);
    safetensors::serialize(
        [
            (
                "w_f32",
                TensorView::new(Dtype::F32, vec![w_f32.len()], &f32_raw).unwrap(),
            ),
            (
                "w_bf16",
                TensorView::new(Dtype::BF16, vec![w_bf16.len()], &bf16_raw).unwrap(),
            ),
        ],
        None,
    )
    .unwrap()
}

fn read_f32_tensor(weights_file: &Path, name: &str) -> Vec<f32> {
    let bytes = fs::read(weights_file).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let data = st.tensor(name).unwrap().data();
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_bf16_tensor(weights_file: &Path, name: &str) -> Vec<f32> {
    let bytes = fs::read(weights_file).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let data = st.tensor(name).unwrap().data();
    data.chunks_exact(2)
        .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

fn sha256_file(path: &Path) -> String {
    let data = fs::read(path).unwrap();
    Sha256::digest(&data)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

struct Fixture {
    _tmp: tempfile::TempDir,
    model_dir: std::path::PathBuf,
    patch_a: std::path::PathBuf,
    patch_b_disjoint: std::path::PathBuf,
    patch_b_overlap: std::path::PathBuf,
    base_sha256: String,
}

/// Builds a 512-element F32 + 512-element BF16 checkpoint (large enough that
/// single-coordinate edits are stored sparse) plus three replacement patches:
/// - `patch_a`: modifies coord 10 (+2.0 in f32, +2.0 in bf16) and coord 20 (+4.0)
/// - `patch_b_disjoint`: modifies coord 30 (+6.0)
/// - `patch_b_overlap`: modifies coord 10 (+6.0 in f32, +6.0 in bf16) and coord 40 (+8.0)
fn setup_fixture() -> Fixture {
    let tmp = tempfile::tempdir().unwrap();
    let base_f32 = vec![10.0f32; 512];
    let base_bf16 = vec![10.0f32; 512];
    let base_bytes = make_checkpoint(&base_f32, &base_bf16);

    let model_dir = tmp.path().join("base-model");
    fs::create_dir(&model_dir).unwrap();
    let base_weights = model_dir.join("model.safetensors");
    fs::write(&base_weights, &base_bytes).unwrap();
    fs::write(model_dir.join("config.json"), b"{\"test\":true}\n").unwrap();
    let base_sha256 = sha256_file(&base_weights);

    let build_patch = |name: &str, f32_edits: &[(usize, f32)], bf16_edits: &[(usize, f32)]| {
        let mut ef32 = base_f32.clone();
        for &(idx, val) in f32_edits {
            ef32[idx] = val;
        }
        let mut ebf16 = base_bf16.clone();
        for &(idx, val) in bf16_edits {
            ebf16[idx] = val;
        }
        let edited_path = tmp.path().join(format!("{name}-edited.safetensors"));
        fs::write(&edited_path, make_checkpoint(&ef32, &ebf16)).unwrap();
        let out_path = tmp.path().join(format!("{name}.safetensors"));
        cx_run(|cx| {
            build_replacement(
                &BuildReplacement {
                    base: base_weights.clone(),
                    edited: edited_path,
                    manifest: None,
                    out: out_path.clone(),
                },
                cx,
            )
        })
        .unwrap();
        out_path
    };

    let patch_a = build_patch("patch-a", &[(10, 12.0), (20, 14.0)], &[(10, 12.0)]);
    let patch_b_disjoint = build_patch("patch-b-disjoint", &[(30, 16.0)], &[(30, 16.0)]);
    let patch_b_overlap = build_patch("patch-b-overlap", &[(10, 16.0), (40, 18.0)], &[(10, 16.0)]);

    Fixture {
        _tmp: tmp,
        model_dir,
        patch_a,
        patch_b_disjoint,
        patch_b_overlap,
        base_sha256,
    }
}

#[test]
fn non_overlapping_patches_stack_bit_exact() {
    let f = setup_fixture();
    let out = f._tmp.path().join("stacked-disjoint");
    let report = cx_run(|cx| {
        apply_stack(
            &ApplyStack {
                patches: vec![
                    PatchSpec {
                        path: f.patch_a.clone(),
                        weight: 1.0,
                    },
                    PatchSpec {
                        path: f.patch_b_disjoint.clone(),
                        weight: 1.0,
                    },
                ],
                model: f.model_dir.clone(),
                out: Some(out.clone()),
                in_place: false,
                on_conflict: ConflictPolicy::Error,
                expect_tensor_sha256: BTreeMap::new(),
            },
            cx,
        )
    })
    .unwrap();
    assert_eq!(report.tensors_changed, 2);

    let wf32 = read_f32_tensor(&out.join("model.safetensors"), "w_f32");
    assert_eq!(wf32[10], 12.0);
    assert_eq!(wf32[20], 14.0);
    assert_eq!(wf32[30], 16.0);
    assert_eq!(wf32[0], 10.0);

    let wbf16 = read_bf16_tensor(&out.join("model.safetensors"), "w_bf16");
    assert_eq!(wbf16[10], 12.0);
    assert_eq!(wbf16[30], 16.0);

    let state = read_state(&out).unwrap().expect("state recorded");
    assert_eq!(state.applied_patches.len(), 2);
    assert_eq!(state.base_sha256, f.base_sha256);
    assert!(state.backup_rel_path.is_none());
}

#[test]
fn overlapping_patches_error_by_default() {
    let f = setup_fixture();
    let out = f._tmp.path().join("stacked-conflict");
    let err = cx_run(|cx| {
        apply_stack(
            &ApplyStack {
                patches: vec![
                    PatchSpec {
                        path: f.patch_a.clone(),
                        weight: 1.0,
                    },
                    PatchSpec {
                        path: f.patch_b_overlap.clone(),
                        weight: 1.0,
                    },
                ],
                model: f.model_dir.clone(),
                out: Some(out.clone()),
                in_place: false,
                on_conflict: ConflictPolicy::Error,
                expect_tensor_sha256: BTreeMap::new(),
            },
            cx,
        )
    })
    .err()
    .expect("expected PatchConflict");
    assert!(
        matches!(
            err,
            ToolError::PatchConflict {
                overlapping_entries: 1,
                ..
            }
        ),
        "expected PatchConflict, got {err:?}"
    );
    assert!(!out.exists(), "refused stack must not publish output");
}

#[test]
fn overlapping_policies_produce_exact_math() {
    let f = setup_fixture();

    // Base W0 = 10.0, A = 12.0 (delta +2), B = 16.0 (delta +6)
    // - add:        10 + 2 + 6 = 18.0
    // - average:    (12 + 16) / 2 = 14.0
    // - weighted:   0.5*A + 0.25*B => 10 + 0.5*2 + 0.25*6 = 12.5
    // - last-wins:  16.0
    // - first-wins: 12.0
    let cases = [
        ("add", ConflictPolicy::Add, 1.0, 1.0, 18.0f32, 14.0f32),
        (
            "average",
            ConflictPolicy::Average,
            1.0,
            1.0,
            14.0f32,
            14.0f32,
        ),
        (
            "weighted",
            ConflictPolicy::Weighted,
            0.5,
            0.25,
            12.5f32,
            12.0f32, // non-overlapping coord 20: 10 + 0.5*(14 - 10) = 12.0
        ),
        (
            "last-wins",
            ConflictPolicy::LastWins,
            1.0,
            1.0,
            16.0f32,
            14.0f32,
        ),
        (
            "first-wins",
            ConflictPolicy::FirstWins,
            1.0,
            1.0,
            12.0f32,
            14.0f32,
        ),
    ];

    for (label, policy, wa, wb, expected_coord10, expected_coord20) in cases {
        let out = f._tmp.path().join(format!("out-{label}"));
        cx_run(|cx| {
            apply_stack(
                &ApplyStack {
                    patches: vec![
                        PatchSpec {
                            path: f.patch_a.clone(),
                            weight: wa,
                        },
                        PatchSpec {
                            path: f.patch_b_overlap.clone(),
                            weight: wb,
                        },
                    ],
                    model: f.model_dir.clone(),
                    out: Some(out.clone()),
                    in_place: false,
                    on_conflict: policy,
                    expect_tensor_sha256: BTreeMap::new(),
                },
                cx,
            )
        })
        .unwrap();

        let wf32 = read_f32_tensor(&out.join("model.safetensors"), "w_f32");
        assert_eq!(wf32[10], expected_coord10, "policy {label} f32 coord 10");
        assert_eq!(wf32[20], expected_coord20, "policy {label} f32 coord 20");
        let wbf16 = read_bf16_tensor(&out.join("model.safetensors"), "w_bf16");
        assert_eq!(wbf16[10], expected_coord10, "policy {label} bf16 coord 10");
    }
}

#[test]
fn in_place_backup_incremental_stack_and_restore() {
    let f = setup_fixture();

    // 1. Apply patch A in-place.
    cx_run(|cx| {
        apply_stack(
            &ApplyStack {
                patches: vec![PatchSpec {
                    path: f.patch_a.clone(),
                    weight: 1.0,
                }],
                model: f.model_dir.clone(),
                out: None,
                in_place: true,
                on_conflict: ConflictPolicy::Error,
                expect_tensor_sha256: BTreeMap::new(),
            },
            cx,
        )
    })
    .unwrap();

    assert!(f.model_dir.join(STATE_FILE).is_file());
    assert!(f.model_dir.join(BACKUP_DIR).is_dir());
    let wf32 = read_f32_tensor(&f.model_dir.join("model.safetensors"), "w_f32");
    assert_eq!(wf32[10], 12.0);
    assert_eq!(wf32[20], 14.0);

    // 2. Stack overlapping patch B on top of already-patched model in-place with --on-conflict=average.
    cx_run(|cx| {
        apply_stack(
            &ApplyStack {
                patches: vec![PatchSpec {
                    path: f.patch_b_overlap.clone(),
                    weight: 1.0,
                }],
                model: f.model_dir.clone(),
                out: None,
                in_place: true,
                on_conflict: ConflictPolicy::Average,
                expect_tensor_sha256: BTreeMap::new(),
            },
            cx,
        )
    })
    .unwrap();

    let wf32 = read_f32_tensor(&f.model_dir.join("model.safetensors"), "w_f32");
    assert_eq!(wf32[10], 14.0, "average of A=12 and B=16");
    assert_eq!(wf32[20], 14.0, "from patch A");
    assert_eq!(wf32[40], 18.0, "from patch B");
    let state = read_state(&f.model_dir).unwrap().unwrap();
    assert_eq!(state.applied_patches.len(), 2);

    // 3. Also test branching out to --out from an in-place patched model: must not carry .weightspace-backup.
    let branched_out = f._tmp.path().join("branched");
    cx_run(|cx| {
        apply_stack(
            &ApplyStack {
                patches: vec![PatchSpec {
                    path: f.patch_b_disjoint.clone(),
                    weight: 1.0,
                }],
                model: f.model_dir.clone(),
                out: Some(branched_out.clone()),
                in_place: false,
                on_conflict: ConflictPolicy::Average,
                expect_tensor_sha256: BTreeMap::new(),
            },
            cx,
        )
    })
    .unwrap();
    assert!(!branched_out.join(BACKUP_DIR).exists());
    let wf32_branched = read_f32_tensor(&branched_out.join("model.safetensors"), "w_f32");
    assert_eq!(wf32_branched[10], 14.0);
    assert_eq!(wf32_branched[30], 16.0);
    assert_eq!(wf32_branched[40], 18.0);

    // 4. Restore original base weights and verify exact SHA-256 + cleanup.
    let restored = cx_run(|cx| restore_backup(&f.model_dir, cx)).unwrap();
    assert_eq!(restored.restored_sha256.as_str(), f.base_sha256);
    assert_eq!(
        sha256_file(&f.model_dir.join("model.safetensors")),
        f.base_sha256
    );
    assert!(!f.model_dir.join(STATE_FILE).exists());
    assert!(!f.model_dir.join(BACKUP_DIR).exists());
}

#[test]
fn cli_stack_in_place_and_restore_round_trip() {
    let f = setup_fixture();
    let bin = env!("CARGO_BIN_EXE_weightspace");

    let out = Command::new(bin)
        .args([
            "apply",
            "--patch",
            f.patch_a.to_str().unwrap(),
            "--patch",
            f.patch_b_overlap.to_str().unwrap(),
            "--model",
            f.model_dir.to_str().unwrap(),
            "--in-place",
            "--on-conflict",
            "add",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let wf32 = read_f32_tensor(&f.model_dir.join("model.safetensors"), "w_f32");
    assert_eq!(wf32[10], 18.0);

    let restore = Command::new(bin)
        .args(["restore", "--model", f.model_dir.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        restore.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&restore.stderr)
    );
    assert!(
        String::from_utf8_lossy(&restore.stdout).contains("restored original weights"),
        "stdout: {}",
        String::from_utf8_lossy(&restore.stdout)
    );
    assert_eq!(
        sha256_file(&f.model_dir.join("model.safetensors")),
        f.base_sha256
    );
}

#[test]
fn restore_refuses_tampered_backup() {
    let f = setup_fixture();
    cx_run(|cx| {
        apply_stack(
            &ApplyStack {
                patches: vec![PatchSpec {
                    path: f.patch_a.clone(),
                    weight: 1.0,
                }],
                model: f.model_dir.clone(),
                out: None,
                in_place: true,
                on_conflict: ConflictPolicy::Error,
                expect_tensor_sha256: BTreeMap::new(),
            },
            cx,
        )
    })
    .unwrap();

    let state = read_state(&f.model_dir).unwrap().unwrap();
    let backup_file = f.model_dir.join(state.backup_rel_path.unwrap());
    let mut bytes = fs::read(&backup_file).unwrap();
    bytes[0] ^= 0xff;
    fs::write(&backup_file, bytes).unwrap();

    let err = cx_run(|cx| restore_backup(&f.model_dir, cx)).unwrap_err();
    assert!(
        matches!(err, ToolError::Backup { .. }),
        "expected Backup integrity refusal, got {err:?}"
    );
}
