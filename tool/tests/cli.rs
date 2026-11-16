//! CLI argument-to-operation mapping through the real binary, plus the
//! preserved `check` success output.

mod common;

use std::fs;
use std::process::{Command, Output};

use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};
use sha2::{Digest, Sha256};

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_weightspace"))
        .args(args)
        .output()
        .unwrap()
}

fn stdout(out: &Output) -> String {
    String::from_utf8(out.stdout.clone()).unwrap()
}

fn stderr(out: &Output) -> String {
    String::from_utf8(out.stderr.clone()).unwrap()
}

fn u16s(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn u32s(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn st_bytes(big: &[u16], small: &[u32]) -> Vec<u8> {
    let big_data = u16s(big);
    let small_data = u32s(small);
    safetensors::serialize(
        [
            (
                "big",
                TensorView::new(Dtype::BF16, vec![big.len()], &big_data).unwrap(),
            ),
            (
                "small",
                TensorView::new(Dtype::F32, vec![small.len()], &small_data).unwrap(),
            ),
        ],
        None,
    )
    .unwrap()
}

fn tensor_hash(bytes: &[u8], name: &str) -> String {
    let parsed = SafeTensors::deserialize(bytes).unwrap();
    Sha256::digest(parsed.tensor(name).unwrap().data())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[test]
fn check_success_output_preserved() {
    let out = run(&["check", common::repo().to_str().unwrap()]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(stdout(&out).trim(), "checks passed");
}

#[test]
fn replacement_and_apply_via_cli() {
    let tmp = tempfile::tempdir().unwrap();
    let big: Vec<u16> = (0..512).map(|i| (i * 3 + 2) as u16).collect();
    let mut big_edited = big.clone();
    big_edited[9] = 0x1234;
    let base_bytes = st_bytes(&big, &[1, 2]);
    let edited_bytes = st_bytes(&big_edited, &[3, 4]);
    let base = tmp.path().join("base.safetensors");
    let edited = tmp.path().join("edited.safetensors");
    fs::write(&base, &base_bytes).unwrap();
    fs::write(&edited, &edited_bytes).unwrap();

    let patch = tmp.path().join("patch.safetensors");
    let out = run(&[
        "replacement",
        "--base",
        base.to_str().unwrap(),
        "--edited",
        edited.to_str().unwrap(),
        "--out",
        patch.to_str().unwrap(),
    ]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(
        stdout(&out).contains("replacement written"),
        "{}",
        stdout(&out)
    );
    assert!(patch.exists());

    let model = tmp.path().join("model");
    fs::create_dir(&model).unwrap();
    fs::write(model.join("model.safetensors"), &base_bytes).unwrap();
    let destination = tmp.path().join("patched");
    // `small` changes in both elements and is stored full; `big` is sparse.
    let expect = format!("small={}", tensor_hash(&base_bytes, "small"));
    let out = run(&[
        "apply",
        "--patch",
        patch.to_str().unwrap(),
        "--model",
        model.to_str().unwrap(),
        "--out",
        destination.to_str().unwrap(),
        "--expect-tensor-sha256",
        &expect,
    ]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(
        stdout(&out).contains("patched model written"),
        "{}",
        stdout(&out)
    );
    let result = fs::read(destination.join("model.safetensors")).unwrap();
    for name in ["big", "small"] {
        assert_eq!(tensor_hash(&result, name), tensor_hash(&edited_bytes, name));
    }
}

// --temp-dir: snapshots are made in the given directory, which is empty again afterwards (Unix unnamed form).
#[test]
fn replacement_with_temp_dir_flag() {
    let tmp = tempfile::tempdir().unwrap();
    let snap = tmp.path().join("snap");
    fs::create_dir(&snap).unwrap();
    let base_bytes = st_bytes(&[1, 2, 3], &[1]);
    let edited_bytes = st_bytes(&[1, 9, 3], &[1]);
    let base = tmp.path().join("base.safetensors");
    let edited = tmp.path().join("edited.safetensors");
    fs::write(&base, &base_bytes).unwrap();
    fs::write(&edited, &edited_bytes).unwrap();

    let patch = tmp.path().join("patch.safetensors");
    let out = run(&[
        "replacement",
        "--base",
        base.to_str().unwrap(),
        "--edited",
        edited.to_str().unwrap(),
        "--out",
        patch.to_str().unwrap(),
        "--temp-dir",
        snap.to_str().unwrap(),
    ]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(patch.exists());
    assert_eq!(
        fs::read_dir(&snap).unwrap().count(),
        0,
        "no snapshot may remain in --temp-dir"
    );
    // Inputs unchanged.
    assert_eq!(fs::read(&base).unwrap(), base_bytes);
    assert_eq!(fs::read(&edited).unwrap(), edited_bytes);

    // Help documents the snapshot-space requirement.
    let help = run(&["replacement", "--help"]);
    assert!(help.status.success());
    assert!(stdout(&help).contains("--temp-dir"), "{}", stdout(&help));
    assert!(
        stdout(&help).contains("free space for a copy"),
        "{}",
        stdout(&help)
    );
}

#[test]
fn cli_errors() {
    let tmp = tempfile::tempdir().unwrap();
    let missing = tmp.path().join("nope.safetensors");
    let some = tmp.path().join("x.safetensors");
    fs::write(&some, st_bytes(&[1, 2, 3], &[1])).unwrap();
    let dir = tmp.path();

    let cases: Vec<(Vec<String>, &str)> = vec![
        // Unknown subcommand (clap usage error).
        (vec!["frobnicate".into()], ""),
        // Missing required --out (clap usage error).
        (
            vec![
                "replacement".into(),
                "--base".into(),
                some.to_str().unwrap().into(),
                "--edited".into(),
                some.to_str().unwrap().into(),
            ],
            "",
        ),
        // Malformed NAME=HEX.
        (
            vec![
                "apply".into(),
                "--patch".into(),
                some.to_str().unwrap().into(),
                "--model".into(),
                dir.to_str().unwrap().into(),
                "--out".into(),
                dir.join("o").to_str().unwrap().into(),
                "--expect-tensor-sha256".into(),
                "noequals".into(),
            ],
            "NAME=HEX",
        ),
        // Missing input file.
        (
            vec![
                "replacement".into(),
                "--base".into(),
                missing.to_str().unwrap().into(),
                "--edited".into(),
                some.to_str().unwrap().into(),
                "--out".into(),
                dir.join("p.safetensors").to_str().unwrap().into(),
            ],
            "",
        ),
    ];
    for (args, needle) in cases {
        let out = Command::new(env!("CARGO_BIN_EXE_weightspace"))
            .args(&args)
            .output()
            .unwrap();
        assert!(!out.status.success(), "{args:?} unexpectedly succeeded");
        if !needle.is_empty() {
            assert!(
                stderr(&out).contains(needle),
                "{args:?}: stderr {} lacks {needle:?}",
                stderr(&out)
            );
        }
    }
    assert!(!dir.join("o").exists());
    assert!(!dir.join("p.safetensors").exists());

    // Apply of a full-form patch without its expected digest explains the flag.
    let base_bytes = st_bytes(&[1, 2, 3], &[7]);
    let edited_bytes = st_bytes(&[9, 9, 9], &[8]);
    let base = tmp.path().join("b.safetensors");
    let edited = tmp.path().join("e.safetensors");
    fs::write(&base, &base_bytes).unwrap();
    fs::write(&edited, &edited_bytes).unwrap();
    let patch = tmp.path().join("full-patch.safetensors");
    let out = run(&[
        "replacement",
        "--base",
        base.to_str().unwrap(),
        "--edited",
        edited.to_str().unwrap(),
        "--out",
        patch.to_str().unwrap(),
    ]);
    assert!(out.status.success(), "{}", stderr(&out));
    let model = tmp.path().join("m");
    fs::create_dir(&model).unwrap();
    fs::write(model.join("model.safetensors"), &base_bytes).unwrap();
    let destination = tmp.path().join("dest");
    let out = run(&[
        "apply",
        "--patch",
        patch.to_str().unwrap(),
        "--model",
        model.to_str().unwrap(),
        "--out",
        destination.to_str().unwrap(),
    ]);
    assert!(!out.status.success());
    assert!(
        stderr(&out).contains("--expect-tensor-sha256"),
        "{}",
        stderr(&out)
    );
    assert!(!destination.exists());
}
