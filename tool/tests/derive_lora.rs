mod common;
use common::{st_bytes, u16s};
use safetensors::{Dtype, SafeTensors};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use tempfile::TempDir;

fn hex_digest(data: &[u8]) -> String {
    Sha256::digest(data)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn cli(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_weightspace"))
        .args(args)
        .output()
        .unwrap()
}
fn text(path: &Path) -> &str {
    path.to_str().unwrap()
}
fn success(output: Output) {
    assert!(
        output.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
fn refused(output: Output, expected: &str, out: &Path) {
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains(expected),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!out.exists());
}

struct Fixture {
    temp: TempDir,
    model: PathBuf,
    patch: PathBuf,
    expected: String,
}
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let model = temp.path().join("model");
        fs::create_dir(&model).unwrap();
        let base = vec![0u16; 1024];
        let mut edited = base.clone();
        for i in 0..32 {
            edited[i * 32 + if i < 16 { 1 } else { 10 }] = if i < 16 { 0x3f80 } else { 0x4000 };
        }
        let tiny = u16s(&[0x3f80; 4]);
        let expected = format!("tiny.weight={}", hex_digest(&tiny));
        fs::write(
            model.join("model.safetensors"),
            st_bytes(&[
                ("linear.weight", Dtype::BF16, &[32, 32], u16s(&base)),
                ("tiny.weight", Dtype::BF16, &[2, 2], tiny),
                ("untouched", Dtype::BF16, &[1], u16s(&[0x3f80])),
            ]),
        )
        .unwrap();
        let edit_path = temp.path().join("edited.safetensors");
        fs::write(
            &edit_path,
            st_bytes(&[
                ("linear.weight", Dtype::BF16, &[32, 32], u16s(&edited)),
                (
                    "tiny.weight",
                    Dtype::BF16,
                    &[2, 2],
                    u16s(&[0x4000, 0x3fc0, 0x3fa0, 0x3f40]),
                ),
                ("untouched", Dtype::BF16, &[1], u16s(&[0x3f80])),
            ]),
        )
        .unwrap();
        let patch = temp.path().join("replacements.safetensors");
        success(cli(&[
            "replacement",
            "--base",
            text(&model.join("model.safetensors")),
            "--edited",
            text(&edit_path),
            "--out",
            text(&patch),
            "--temp-dir",
            text(temp.path()),
        ]));
        Self {
            temp,
            model,
            patch,
            expected,
        }
    }
    fn derive(&self, method: &str, rank: &str, out: &Path) -> Output {
        cli(&[
            "derive-lora",
            "--model",
            text(&self.model),
            "--patch",
            text(&self.patch),
            "--rank",
            rank,
            "--factor-dtype",
            "f32",
            "--method",
            method,
            "--out",
            text(out),
            "--expect-tensor-sha256",
            &self.expected,
            "--temp-dir",
            text(self.temp.path()),
        ])
    }
}

// Proves both CLI methods derive only the full+sparse patch tensors, measured
// reports reproduce independently read stored factors, and two runs of each
// method are byte-identical. Retains synthetic artifacts only when requested.
#[test]
fn replacement_to_lora_end_to_end() {
    let f = Fixture::new();
    let patch_data = fs::read(&f.patch).unwrap();
    let patch = SafeTensors::deserialize(&patch_data).unwrap();
    assert!(patch.tensor("linear.weight#idx").is_ok());
    assert!(patch.tensor("tiny.weight").is_ok());
    let original = fs::read(f.model.join("model.safetensors")).unwrap();
    for method in ["svd", "selector"] {
        let first = f.temp.path().join(method);
        let second = f.temp.path().join(format!("{method}-repeat"));
        success(f.derive(method, "2", &first));
        success(f.derive(method, "2", &second));
        for name in ["adapter.safetensors", "lora-report.json"] {
            assert_eq!(
                fs::read(first.join(name)).unwrap(),
                fs::read(second.join(name)).unwrap()
            );
        }
        let report: Value =
            serde_json::from_slice(&fs::read(first.join("lora-report.json")).unwrap()).unwrap();
        assert_eq!(report["tensors"].as_object().unwrap().len(), 2);
        assert_eq!(report["effective_scale"], 1);
        assert!(
            report["notice"]
                .as_str()
                .unwrap()
                .contains("study claims NEVER transfer")
        );
        let data = fs::read(first.join("adapter.safetensors")).unwrap();
        let adapter = SafeTensors::deserialize(&data).unwrap();
        assert_eq!(adapter.names().len(), 4);
        for (name, shape, k) in [("linear.weight", [32, 32], 2), ("tiny.weight", [2, 2], 2)] {
            let item = &report["tensors"][name];
            assert_eq!(item["active_columns_k"], k);
            assert_eq!(item["exact_rank_evidence"]["proven_exact_rank"], 2);
            assert_eq!(item["bf16_mismatch_count"], 0);
            assert_eq!(item["max_ulp"], 0);
            assert_eq!(item["mean_ulp_all_elements"], 0.0);
            assert_eq!(item["mean_ulp_active_columns"], 0.0);
            assert_eq!(item["baked_frobenius_error"], 0.0);
            let a = adapter.tensor(&format!("{name}.lora_A")).unwrap();
            let b = adapter.tensor(&format!("{name}.lora_B")).unwrap();
            assert_eq!(a.shape(), &[2, shape[1]]);
            assert_eq!(b.shape(), &[shape[0], 2]);
            assert_eq!(a.dtype(), Dtype::F32);
            assert_eq!(
                item["raw_factor_bytes"]["total"],
                (a.data().len() + b.data().len()) as u64
            );
            let values = |data: &[u8]| {
                data.chunks_exact(4)
                    .map(|v| f32::from_le_bytes(v.try_into().unwrap()) as f64)
                    .collect::<Vec<_>>()
            };
            let av = values(a.data());
            let bv = values(b.data());
            let expected = if name == "linear.weight" {
                (0..1024)
                    .map(|p| {
                        if p / 32 < 16 && p % 32 == 1 {
                            1.0
                        } else if p / 32 >= 16 && p % 32 == 10 {
                            2.0
                        } else {
                            0.0
                        }
                    })
                    .collect::<Vec<_>>()
            } else {
                vec![1.0, 0.5, 0.25, -0.25]
            };
            let mut error = 0.0f64;
            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    let reconstructed: f64 =
                        (0..2).map(|t| bv[i * 2 + t] * av[t * shape[1] + j]).sum();
                    error = error.hypot(reconstructed - expected[i * shape[1] + j]);
                }
            }
            assert_eq!(item["delta_frobenius_error"].as_f64().unwrap(), error);
            if method == "selector" {
                assert_eq!(error, 0.0);
            }
        }
        if let Ok(root) = std::env::var("WEIGHTSPACE_LORA_EVIDENCE") {
            let root = PathBuf::from(root);
            fs::create_dir_all(&root).unwrap();
            for path in [&first, &second] {
                common::copy_dir(path, &root.join(path.file_name().unwrap()));
            }
            common::copy_dir(&f.model, &root.join("model"));
            fs::copy(&f.patch, root.join("replacements.safetensors")).unwrap();
            fs::copy(
                f.temp.path().join("edited.safetensors"),
                root.join("edited.safetensors"),
            )
            .unwrap();
        }
    }
    assert_eq!(
        original,
        fs::read(f.model.join("model.safetensors")).unwrap()
    );
    assert_eq!(patch_data, fs::read(&f.patch).unwrap());
}

// Proves invalid ranks refuse with specific diagnostics and publish no output,
// beside the valid rank-2 positive control exercised above.
#[test]
fn rank_refusals() {
    let f = Fixture::new();
    for (method, rank, message) in [
        ("svd", "0", "rank must be positive"),
        ("svd", "3", "exceeds active columns"),
        ("selector", "1", "selector requires rank=k"),
    ] {
        let out = f.temp.path().join(format!("{method}-{rank}"));
        refused(f.derive(method, rank, &out), message, &out);
    }
}

// Proves a pre-existing output directory/file is preserved byte-for-byte.
#[test]
fn existing_output_is_untouched() {
    let f = Fixture::new();
    let out = f.temp.path().join("out");
    fs::create_dir(&out).unwrap();
    fs::write(out.join("keep"), b"sentinel").unwrap();
    let result = f.derive("svd", "2", &out);
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("already exists"));
    assert_eq!(fs::read(out.join("keep")).unwrap(), b"sentinel");
    assert_eq!(fs::read_dir(&out).unwrap().count(), 1);
    let file = f.temp.path().join("file");
    fs::write(&file, b"sentinel").unwrap();
    assert!(!f.derive("svd", "2", &file).status.success());
    assert_eq!(fs::read(file).unwrap(), b"sentinel");
}

// Proves nonfinite base and patched tensors refuse without partial output;
// finite valid matrices are covered by the neighboring CLI positive control.
#[test]
fn nonfinite_refusals() {
    let f = Fixture::new();
    for (label, base, new) in [("base", 0x7f80, 0x4000), ("patch", 0x3f80, 0x7fc1)] {
        let path = f.model.join("model.safetensors");
        let old = u16s(&[base, 0, 0, 0]);
        fs::write(
            &path,
            st_bytes(&[("bad", Dtype::BF16, &[2, 2], old.clone())]),
        )
        .unwrap();
        fs::write(
            &f.patch,
            st_bytes(&[("bad", Dtype::BF16, &[2, 2], u16s(&[new, 0, 0, 0]))]),
        )
        .unwrap();
        let out = f.temp.path().join(label);
        let expected = format!("bad={}", hex_digest(&old));
        refused(
            cli(&[
                "derive-lora",
                "--model",
                text(&f.model),
                "--patch",
                text(&f.patch),
                "--rank",
                "1",
                "--factor-dtype",
                "f32",
                "--method",
                "svd",
                "--out",
                text(&out),
                "--expect-tensor-sha256",
                &expected,
            ]),
            "finite BF16",
            &out,
        );
    }
}

// Proves absent patch tensor, missing/wrong full-form hash and wrong sparse old
// bytes retain apply's refusal contracts; no outputs are published on failures.
#[test]
fn base_identity_refusals() {
    let f = Fixture::new();
    let out = f.temp.path().join("absent");
    fs::write(
        &f.patch,
        st_bytes(&[("absent", Dtype::BF16, &[2, 2], u16s(&[0; 4]))]),
    )
    .unwrap();
    refused(f.derive("svd", "2", &out), "not present in the base", &out);
    let f = Fixture::new();
    let out = f.temp.path().join("missing-hash");
    refused(
        cli(&[
            "derive-lora",
            "--model",
            text(&f.model),
            "--patch",
            text(&f.patch),
            "--rank",
            "2",
            "--factor-dtype",
            "f32",
            "--method",
            "svd",
            "--out",
            text(&out),
        ]),
        "expected base",
        &out,
    );
    let out = f.temp.path().join("wrong-hash");
    let wrong = format!("tiny.weight={}", "0".repeat(64));
    refused(
        cli(&[
            "derive-lora",
            "--model",
            text(&f.model),
            "--patch",
            text(&f.patch),
            "--rank",
            "2",
            "--factor-dtype",
            "f32",
            "--method",
            "svd",
            "--out",
            text(&out),
            "--expect-tensor-sha256",
            &wrong,
        ]),
        "expected base sha256",
        &out,
    );
    let data = fs::read(f.model.join("model.safetensors")).unwrap();
    let parsed = SafeTensors::deserialize(&data).unwrap();
    let mut changed = vec![0u16; 1024];
    changed[1] = 0x3f80;
    fs::write(
        f.model.join("model.safetensors"),
        st_bytes(&[
            ("linear.weight", Dtype::BF16, &[32, 32], u16s(&changed)),
            (
                "tiny.weight",
                Dtype::BF16,
                &[2, 2],
                parsed.tensor("tiny.weight").unwrap().data().to_vec(),
            ),
        ]),
    )
    .unwrap();
    let out = f.temp.path().join("wrong-old");
    refused(f.derive("svd", "2", &out), "#old value", &out);
}

// Proves explicit dtype/method/rank are mandatory and unsupported F16 or method
// values cannot silently select an alternative arithmetic path.
#[test]
fn cli_explicit_options() {
    for args in [
        vec!["derive-lora"],
        vec!["derive-lora", "--factor-dtype", "f16"],
        vec!["derive-lora", "--method", "unknown"],
    ] {
        assert!(!cli(&args).status.success());
    }
    let help = cli(&["derive-lora", "--help"]);
    assert!(help.status.success());
    let help = String::from_utf8(help.stdout).unwrap();
    assert!(
        help.contains("study claims NEVER transfer")
            || help.contains("Study claims NEVER transfer")
    );
    assert!(help.contains("UNQUALIFIED"));
}

// Proves the report checker detects a deliberately truncated rank-1 treatment,
// including mismatches in the omitted active column and nonzero bake error.
#[test]
fn truncated_svd_reports_loss() {
    let f = Fixture::new();
    let out = f.temp.path().join("truncated");
    success(f.derive("svd", "1", &out));
    let report: Value =
        serde_json::from_slice(&fs::read(out.join("lora-report.json")).unwrap()).unwrap();
    let linear = &report["tensors"]["linear.weight"];
    assert_eq!(linear["bf16_mismatch_count"], 16);
    assert_eq!(linear["max_ulp"], 16256);
    assert_eq!(linear["mean_ulp_all_elements"], 254.0);
    assert_eq!(linear["mean_ulp_active_columns"], 4064.0);
    assert!((linear["baked_frobenius_error"].as_f64().unwrap() - 4.0).abs() < 1e-12);
    assert!((linear["delta_frobenius_error"].as_f64().unwrap() - 4.0).abs() < 1e-12);
    assert!((linear["relative_delta_error"].as_f64().unwrap() - 1.0 / 5.0f64.sqrt()).abs() < 1e-12);
}

// Proves rank-deficient modular bounds remain lower bounds in the report,
// and a signed-zero-only column remains active instead of being discarded.
#[test]
fn deficient_and_signed_zero_report() {
    let f = Fixture::new();
    let old = u16s(&[0; 4]);
    fs::write(
        f.model.join("model.safetensors"),
        st_bytes(&[("matrix", Dtype::BF16, &[2, 2], old.clone())]),
    )
    .unwrap();
    fs::write(
        &f.patch,
        st_bytes(&[(
            "matrix",
            Dtype::BF16,
            &[2, 2],
            u16s(&[0x3f80, 0x4000, 0x4000, 0x4080]),
        )]),
    )
    .unwrap();
    let expected = format!("matrix={}", hex_digest(&old));
    let run = |out: &Path| {
        cli(&[
            "derive-lora",
            "--model",
            text(&f.model),
            "--patch",
            text(&f.patch),
            "--rank",
            "2",
            "--factor-dtype",
            "f32",
            "--method",
            "selector",
            "--out",
            text(out),
            "--expect-tensor-sha256",
            &expected,
        ])
    };
    let out = f.temp.path().join("deficient");
    success(run(&out));
    let report: Value =
        serde_json::from_slice(&fs::read(out.join("lora-report.json")).unwrap()).unwrap();
    let evidence = &report["tensors"]["matrix"]["exact_rank_evidence"];
    assert_eq!(evidence["modular_ranks"], serde_json::json!([1, 1]));
    assert!(evidence["proven_exact_rank"].is_null());
    fs::write(
        &f.patch,
        st_bytes(&[("matrix", Dtype::BF16, &[2, 2], u16s(&[0x8000; 4]))]),
    )
    .unwrap();
    let out = f.temp.path().join("signed-zero");
    success(run(&out));
    let report: Value =
        serde_json::from_slice(&fs::read(out.join("lora-report.json")).unwrap()).unwrap();
    let tensor = &report["tensors"]["matrix"];
    assert_eq!(tensor["active_columns_k"], 2);
    assert_eq!(tensor["delta_frobenius_error"], 0.0);
    assert!(tensor["relative_delta_error"].is_null());
    assert_eq!(tensor["bf16_mismatch_count"], 4);
    assert_eq!(tensor["max_ulp"], 1);
    assert_eq!(tensor["baked_frobenius_error"], 0.0);
}
