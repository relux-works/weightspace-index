//! Synthetic dual-form publication fixtures, independent of model runtimes.
mod common;
use common::*;
use safetensors::{Dtype, SafeTensors};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn digest(data: &[u8]) -> String {
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
fn p(path: &Path) -> &str {
    path.to_str().unwrap()
}
fn success(o: Output) {
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
}
struct Fixture {
    temp: tempfile::TempDir,
    patch: PathBuf,
    shapes: PathBuf,
    model: PathBuf,
}
impl Fixture {
    fn new(full: bool) -> Self {
        let temp = tempfile::tempdir().unwrap();
        let patch = temp.path().join("replacements.safetensors");
        let shapes = temp.path().join("shapes.json");
        let model = temp.path().join("model");
        fs::create_dir(&model).unwrap();
        fs::write(
            model.join("model.safetensors"),
            st_bytes(&[
                ("sparse", Dtype::BF16, &[3, 4], u16s(&[0x3f80; 12])),
                ("dense", Dtype::BF16, &[2, 3], u16s(&[0x3f80; 6])),
                ("single", Dtype::BF16, &[2, 2], u16s(&[0x3f80; 4])),
                ("zero", Dtype::BF16, &[1, 1], u16s(&[0x3f80])),
            ]),
        )
        .unwrap();
        let mut specs = vec![
            ("sparse#idx", Dtype::I64, vec![2], i64s(&[0, 6])),
            ("sparse#old", Dtype::BF16, vec![2], u16s(&[0x3f80; 2])),
            ("sparse#new", Dtype::BF16, vec![2], u16s(&[0x4000, 0x4040])),
        ];
        if full {
            specs.extend([
                ("dense", Dtype::BF16, vec![2, 3], u16s(&[0x4000; 6])),
                (
                    "single",
                    Dtype::BF16,
                    vec![2, 2],
                    u16s(&[0x4000, 0x3f80, 0x3f80, 0x3f80]),
                ),
                ("zero", Dtype::BF16, vec![1, 1], u16s(&[0x3f80])),
            ]);
        } else {
            specs.extend([
                ("single#idx", Dtype::I64, vec![1], i64s(&[0])),
                ("single#old", Dtype::BF16, vec![1], u16s(&[0x3f80])),
                ("single#new", Dtype::BF16, vec![1], u16s(&[0x4000])),
            ]);
        }
        fs::write(
            &patch,
            st_bytes(
                &specs
                    .iter()
                    .map(|(n, d, s, b)| (*n, *d, s.as_slice(), b.clone()))
                    .collect::<Vec<_>>(),
            ),
        )
        .unwrap();
        write(&shapes, &json!({"single":[2,2],"sparse":[3,4]}));
        Self {
            temp,
            patch,
            shapes,
            model,
        }
    }
    fn derive(&self, out: &Path, full: bool) -> Output {
        let mut args = vec![
            "derive-lora",
            "--method",
            "selector-exact",
            "--factor-dtype",
            "f32",
            "--patch",
            p(&self.patch),
            "--out",
            p(out),
        ];
        let expected: Vec<String> = [("dense", 6), ("single", 4), ("zero", 1)]
            .into_iter()
            .map(|(n, len)| format!("{n}={}", digest(&u16s(&vec![0x3f80; len]))))
            .collect();
        if full {
            args.extend(["--model", p(&self.model)]);
            for e in &expected {
                args.extend(["--expect-tensor-sha256", e]);
            }
        } else {
            args.extend(["--tensor-shapes", p(&self.shapes)]);
        }
        cli(&args)
    }
}

// Proves one invocation handles ranks 0/1/2/3, including dense whole tensors;
// two identical runs preserve both hashes and independently checked products.
#[test]
fn per_tensor_full_and_sparse_deterministic() {
    let f = Fixture::new(true);
    let one = f.temp.path().join("one");
    let two = f.temp.path().join("two");
    success(f.derive(&one, true));
    success(f.derive(&two, true));
    for name in ["adapter.safetensors", "lora-report.json"] {
        assert_eq!(
            digest(&fs::read(one.join(name)).unwrap()),
            digest(&fs::read(two.join(name)).unwrap())
        );
    }
    let r = read(&one.join("lora-report.json"));
    let bytes = fs::read(one.join("adapter.safetensors")).unwrap();
    let a = SafeTensors::deserialize(&bytes).unwrap();
    for (name, rows, width, rank, expected) in [
        ("dense", 2, 3, 3, vec![1.; 6]),
        ("single", 2, 2, 1, vec![1., 0., 0., 0.]),
        (
            "sparse",
            3,
            4,
            2,
            vec![1., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0.],
        ),
        ("zero", 1, 1, 0, vec![0.]),
    ] {
        assert_eq!(r["tensors"][name]["rank"], rank);
        assert_eq!(r["tensors"][name]["bf16_mismatch_count"], 0);
        assert_eq!(r["tensors"][name]["delta_frobenius_error"], 0.0);
        let av = a.tensor(&format!("{name}.lora_A")).unwrap();
        let bv = a.tensor(&format!("{name}.lora_B")).unwrap();
        assert_eq!(av.shape(), &[rank, width]);
        assert_eq!(bv.shape(), &[rows, rank]);
        let values = |data: &[u8]| {
            data.chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()) as f64)
                .collect::<Vec<_>>()
        };
        let av = values(av.data());
        let bv = values(bv.data());
        for i in 0..rows {
            for j in 0..width {
                let got: f64 = (0..rank)
                    .map(|k| bv[i * rank + k] * av[k * width + j])
                    .sum();
                assert_eq!(got, expected[i * width + j]);
            }
        }
    }
}

// Proves offline #old derivation matches model-backed factors but explicitly
// restricts bake evidence to saved coordinates; malformed shapes publish nothing.
#[test]
fn offline_sparse_and_shape_refusals() {
    let f = Fixture::new(false);
    let out = f.temp.path().join("offline");
    success(f.derive(&out, false));
    let r = read(&out.join("lora-report.json"));
    assert_eq!(r["tensors"]["sparse"]["rank"], 2);
    assert_eq!(r["tensors"]["single"]["rank"], 1);
    assert_eq!(r["tensors"]["sparse"]["measured_elements"], 2);
    assert_eq!(
        r["tensors"]["sparse"]["verification_scope"],
        "saved-coordinates-only"
    );
    let backed = f.temp.path().join("backed");
    success(cli(&[
        "derive-lora",
        "--method",
        "selector-exact",
        "--factor-dtype",
        "f32",
        "--patch",
        p(&f.patch),
        "--model",
        p(&f.model),
        "--out",
        p(&backed),
    ]));
    assert_eq!(
        fs::read(out.join("adapter.safetensors")).unwrap(),
        fs::read(backed.join("adapter.safetensors")).unwrap()
    );
    for (label, shapes, message, report) in [
        (
            "missing",
            json!({"sparse":[3,4]}),
            "exactly every patch tensor",
            false,
        ),
        (
            "extra",
            json!({"sparse":[3,4],"single":[2,2],"extra":[1,1]}),
            "exactly every patch tensor",
            false,
        ),
        (
            "bounds",
            json!({"sparse":[1,4],"single":[2,2]}),
            "unsupported tensor coverage",
            true,
        ),
    ] {
        write(&f.shapes, &shapes);
        let dest = f.temp.path().join(label);
        let o = f.derive(&dest, false);
        assert!(!o.status.success());
        assert!(String::from_utf8_lossy(&o.stderr).contains(message));
        assert!(!dest.join("adapter.safetensors").exists());
        assert_eq!(dest.join("lora-report.json").exists(), report);
    }
}

// Proves nonmatrix, non-BF16 and missing offline full-tensor base are named in
// failed coverage reports, with no adapter or silent omission of any tensor.
#[test]
fn unsupported_coverage_is_reported() {
    let f = Fixture::new(false);
    fs::write(
        &f.patch,
        st_bytes(&[
            ("vector#idx", Dtype::I64, &[1], i64s(&[0])),
            ("vector#new", Dtype::BF16, &[1], u16s(&[0x4000])),
            ("vector#old", Dtype::BF16, &[1], u16s(&[0x3f80])),
            ("float#idx", Dtype::I64, &[1], i64s(&[0])),
            ("float#new", Dtype::F32, &[1], 1f32.to_le_bytes().to_vec()),
            ("float#old", Dtype::F32, &[1], 0f32.to_le_bytes().to_vec()),
            ("full", Dtype::BF16, &[1, 1], u16s(&[0x4000])),
        ]),
    )
    .unwrap();
    write(&f.shapes, &json!({"vector":[2],"float":[1,1],"full":[1,1]}));
    let out = f.temp.path().join("unsupported");
    assert!(!f.derive(&out, false).status.success());
    let r = read(&out.join("lora-report.json"));
    assert_eq!(r["status"], "unsupported");
    assert_eq!(r["failures"].as_object().unwrap().len(), 3);
    assert!(
        r["failures"]["vector"]
            .as_str()
            .unwrap()
            .contains("2-D BF16")
    );
    assert!(
        r["failures"]["float"]
            .as_str()
            .unwrap()
            .contains("2-D BF16")
    );
    assert!(
        r["failures"]["full"]
            .as_str()
            .unwrap()
            .contains("requires --model")
    );
    assert!(!out.join("adapter.safetensors").exists());
}

fn host(root: &Path, f: &Fixture, out: &Path) -> PathBuf {
    copy_dir(&repo().join("docs/example"), root);
    let v = root.join("models").join(BASE).join("better-math/v1");
    fs::copy(&f.patch, v.join("files/replacements.safetensors")).unwrap();
    for name in ["adapter.safetensors", "lora-report.json"] {
        fs::copy(out.join(name), v.join("files").join(name)).unwrap();
    }
    let source = digest(&fs::read(&f.patch).unwrap());
    edit(&v.join("version.json"), |r| {
        r["files"][0]["sha256"] = json!(source);
        r["files"][0]["size_bytes"] = json!(fs::metadata(&f.patch).unwrap().len());
        r["files"].as_array_mut().unwrap().truncate(1);
        for name in ["adapter.safetensors", "lora-report.json"] {
            let bytes = fs::read(out.join(name)).unwrap();
            r["files"].as_array_mut().unwrap().push(json!({
                "name":name,"role":"derived","sha256":digest(&bytes),"size_bytes":bytes.len(),
                "locators":[{"kind":"index-path","path":format!("files/{name}")}],
                "derived_from":[source],"tool":{"name":"weightspace derive-lora","version":env!("CARGO_PKG_VERSION")}
            }));
        }
    });
    v
}
fn rehash_report(v: &Path) {
    let data = fs::read(v.join("files/lora-report.json")).unwrap();
    edit(&v.join("version.json"), |r| {
        r["files"][2]["sha256"] = json!(digest(&data));
        r["files"][2]["size_bytes"] = json!(data.len());
    });
}

// Proves a docs/example-style dual entry passes schema/layout/generation/check;
// each planted provenance, pair, digest, rank, metric or method violation fails.
#[test]
fn dual_records_and_broken_provenance_controls() {
    let f = Fixture::new(false);
    let out = f.temp.path().join("derived");
    success(f.derive(&out, false));
    type Mutate = fn(&Path);
    let cases: Vec<(&str, Mutate, &str)> = vec![
        ("valid", |_| {}, ""),
        (
            "missing-report",
            |v| {
                edit(&v.join("version.json"), |r| {
                    r["files"].as_array_mut().unwrap().pop();
                });
                fs::remove_file(v.join("files/lora-report.json")).unwrap();
            },
            "requires both",
        ),
        (
            "multiple-sources",
            |v| {
                fs::copy(
                    v.join("files/replacements.safetensors"),
                    v.join("files/second.safetensors"),
                )
                .unwrap();
                edit(&v.join("version.json"), |r| {
                    let mut source = r["files"][0].clone();
                    source["name"] = json!("second.safetensors");
                    source["locators"][0]["path"] = json!("files/second.safetensors");
                    r["files"].as_array_mut().unwrap().push(source);
                });
            },
            "exactly one replacement source",
        ),
        (
            "foreign-source",
            |v| {
                edit(&v.join("version.json"), |r| {
                    r["files"][1]["derived_from"] = json!(["9".repeat(64)])
                })
            },
            "not a source",
        ),
        (
            "no-provenance",
            |v| {
                edit(&v.join("version.json"), |r| {
                    remove(&mut r["files"][1], "derived_from")
                })
            },
            "missing or invalid version.json",
        ),
        (
            "no-tool",
            |v| {
                edit(&v.join("version.json"), |r| {
                    remove(&mut r["files"][1], "tool")
                })
            },
            "missing or invalid version.json",
        ),
        (
            "wrong-digest",
            |v| {
                edit(&v.join("version.json"), |r| {
                    r["files"][1]["sha256"] = json!("9".repeat(64))
                })
            },
            "SHA-256 differs",
        ),
        (
            "forged-factor",
            |v| {
                let path = v.join("files/adapter.safetensors");
                let mut data = fs::read(&path).unwrap();
                let last = data.len() - 1;
                data[last] ^= 1;
                fs::write(&path, &data).unwrap();
                let hash = digest(&data);
                edit(&v.join("version.json"), |r| {
                    r["files"][1]["sha256"] = json!(hash)
                });
                edit(&v.join("files/lora-report.json"), |r| {
                    r["adapter_safetensors_sha256"] = json!(hash)
                });
                rehash_report(v);
            },
            "rebuilt source selectors",
        ),
        (
            "wrong-method",
            |v| {
                edit(&v.join("files/lora-report.json"), |r| {
                    r["method"] = json!("svd")
                });
                rehash_report(v);
            },
            "complete selector-exact",
        ),
        (
            "wrong-rank",
            |v| {
                edit(&v.join("files/lora-report.json"), |r| {
                    r["tensors"]["sparse"]["rank"] = json!(1)
                });
                rehash_report(v);
            },
            "rank/active columns",
        ),
        (
            "wrong-metric",
            |v| {
                edit(&v.join("files/lora-report.json"), |r| {
                    r["tensors"]["sparse"]["bf16_mismatch_count"] = json!(1)
                });
                rehash_report(v);
            },
            "rebuilt source evidence",
        ),
        (
            "wrong-source-report",
            |v| {
                edit(&v.join("files/lora-report.json"), |r| {
                    r["patch_safetensors_sha256"] = json!("9".repeat(64))
                });
                rehash_report(v);
            },
            "not a source",
        ),
    ];
    for (label, mutate, message) in cases {
        let root = f.temp.path().join(label);
        let v = host(&root, &f, &out);
        mutate(&v);
        let errors = weightspace_tools::layout::check_models(records(), &root.join("models"));
        if label == "valid" {
            assert!(errors.is_empty(), "{errors:#?}");
            copy_dir(&repo().join("schema"), &root.join("schema"));
            copy_dir(&repo().join("docs/example"), &root.join("docs/example"));
            success(cli(&["generate", p(&root)]));
            success(cli(&["check", p(&root)]));
            assert!(
                weightspace_tools::generate::diff(
                    &root,
                    &weightspace_tools::generate::build(&root).unwrap()
                )
                .is_empty()
            );
        } else {
            assert!(
                errors.iter().any(|e| e.contains(message)),
                "{label}: {errors:#?}"
            );
        }
    }
}

// Proves F32 delta loss and signed-zero bake mismatches are exported as metrics,
// not worthiness refusals; an unchanged tensor retains its zero-rank factors.
#[test]
fn numerical_loss_is_reported_without_a_gate() {
    let f = Fixture::new(false);
    let mut specs = Vec::new();
    for (name, old, new) in [
        ("loss", 0x3080, 0x3f80),
        ("signed-zero", 0, 0x8000),
        ("unchanged", 0x3f80, 0x3f80),
    ] {
        specs.push((format!("{name}#idx"), Dtype::I64, i64s(&[0])));
        specs.push((format!("{name}#old"), Dtype::BF16, u16s(&[old])));
        specs.push((format!("{name}#new"), Dtype::BF16, u16s(&[new])));
    }
    fs::write(
        &f.patch,
        st_bytes(
            &specs
                .iter()
                .map(|(n, d, b)| (n.as_str(), *d, &[1][..], b.clone()))
                .collect::<Vec<_>>(),
        ),
    )
    .unwrap();
    write(
        &f.shapes,
        &json!({"loss":[1,1],"signed-zero":[1,1],"unchanged":[1,1]}),
    );
    let out = f.temp.path().join("loss");
    success(f.derive(&out, false));
    let r = read(&out.join("lora-report.json"));
    assert_eq!(
        r["tensors"]["loss"]["delta_frobenius_error"],
        2.0f64.powi(-30)
    );
    assert_eq!(r["tensors"]["loss"]["stored_delta_exact"], false);
    assert_eq!(r["tensors"]["signed-zero"]["bf16_mismatch_count"], 1);
    assert_eq!(r["tensors"]["signed-zero"]["measured_bake_exact"], false);
    assert_eq!(r["tensors"]["unchanged"]["rank"], 0);
    assert!(out.join("adapter.safetensors").is_file());
}

// Proves per-tensor mode preserves existing outputs and rejects a global rank
// rather than silently mixing explicit and automatic rank policies.
#[test]
fn per_tensor_option_and_output_refusals() {
    let f = Fixture::new(false);
    let out = f.temp.path().join("existing");
    fs::create_dir(&out).unwrap();
    fs::write(out.join("keep"), b"sentinel").unwrap();
    assert!(!f.derive(&out, false).status.success());
    assert_eq!(fs::read(out.join("keep")).unwrap(), b"sentinel");
    assert_eq!(fs::read_dir(&out).unwrap().count(), 1);
    let dest = f.temp.path().join("invalid-rank");
    let o = cli(&[
        "derive-lora",
        "--method",
        "selector-exact",
        "--factor-dtype",
        "f32",
        "--patch",
        p(&f.patch),
        "--tensor-shapes",
        p(&f.shapes),
        "--rank",
        "1",
        "--out",
        p(&dest),
    ]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("omit --rank"));
    assert!(!dest.exists());
}

// Proves a base-verified mixed full/sparse publication passes the same profile;
// full bake arithmetic still requires the base for independent revalidation.
#[test]
fn verified_base_publication_profile() {
    let f = Fixture::new(true);
    let out = f.temp.path().join("derived");
    success(f.derive(&out, true));
    let root = f.temp.path().join("catalog");
    host(&root, &f, &out);
    let errors = weightspace_tools::layout::check_models(records(), &root.join("models"));
    assert!(errors.is_empty(), "{errors:#?}");
}

// Pins a disclosed full-form offline limitation, not a security guarantee.
#[test]
fn verified_base_full_form_offline_check_boundary() {
    let f = Fixture::new(true);
    let out = f.temp.path().join("derived");
    success(f.derive(&out, true));
    assert_eq!(
        read(&out.join("lora-report.json"))["input_mode"],
        "verified-base"
    );
    fs::remove_dir_all(&f.model).unwrap();
    for label in ["valid", "forged-full-b", "forged-full-metric"] {
        let root = f.temp.path().join(label);
        let v = host(&root, &f, &out);
        if label == "forged-full-b" {
            let path = v.join("files/adapter.safetensors");
            let original = fs::read(&path).unwrap();
            let tensors = SafeTensors::deserialize(&original).unwrap();
            let b = tensors.tensor("dense.lora_B").unwrap();
            let offset = b.data().as_ptr() as usize - original.as_ptr() as usize;
            let mut forged = original.clone();
            forged[offset..offset + 4].copy_from_slice(&2f32.to_le_bytes());
            assert_ne!(original, forged);
            let changed = SafeTensors::deserialize(&forged).unwrap();
            assert_eq!(
                &changed.tensor("dense.lora_B").unwrap().data()[..4],
                &2f32.to_le_bytes()
            );
            fs::write(&path, &forged).unwrap();
            let hash = digest(&forged);
            edit(&v.join("version.json"), |r| {
                r["files"][1]["sha256"] = json!(hash)
            });
            edit(&v.join("files/lora-report.json"), |r| {
                r["adapter_safetensors_sha256"] = json!(hash)
            });
            rehash_report(&v);
        } else if label == "forged-full-metric" {
            let path = v.join("files/lora-report.json");
            assert_eq!(read(&path)["tensors"]["dense"]["bf16_mismatch_count"], 0);
            edit(&path, |r| {
                r["tensors"]["dense"]["bf16_mismatch_count"] = json!(1)
            });
            assert_eq!(read(&path)["tensors"]["dense"]["bf16_mismatch_count"], 1);
            rehash_report(&v);
        }
        copy_dir(&repo().join("schema"), &root.join("schema"));
        copy_dir(&repo().join("docs/example"), &root.join("docs/example"));
        success(cli(&["generate", p(&root)]));
        success(cli(&["check", p(&root)]));
    }
}

// Proves numeric representation refusals name every affected tensor and emit
// no partial adapter; valid finite but lossy differences are accepted above.
#[test]
fn numeric_coverage_limits_are_explicit() {
    let f = Fixture::new(false);
    let mut specs = Vec::new();
    for (name, old, new) in [
        ("f64-gap", 1, 0x3f80),
        ("f32-overflow", 0xff7f, 0x7f7f),
        ("nonfinite", 0, 0x7f80),
    ] {
        specs.push((format!("{name}#idx"), Dtype::I64, i64s(&[0])));
        specs.push((format!("{name}#old"), Dtype::BF16, u16s(&[old])));
        specs.push((format!("{name}#new"), Dtype::BF16, u16s(&[new])));
    }
    fs::write(
        &f.patch,
        st_bytes(
            &specs
                .iter()
                .map(|(n, d, b)| (n.as_str(), *d, &[1][..], b.clone()))
                .collect::<Vec<_>>(),
        ),
    )
    .unwrap();
    write(
        &f.shapes,
        &json!({"f64-gap":[1,1],"f32-overflow":[1,1],"nonfinite":[1,1]}),
    );
    let out = f.temp.path().join("numeric");
    assert!(!f.derive(&out, false).status.success());
    let r = read(&out.join("lora-report.json"));
    assert!(
        r["failures"]["f64-gap"]
            .as_str()
            .unwrap()
            .contains("not representable in F64")
    );
    assert!(
        r["failures"]["f32-overflow"]
            .as_str()
            .unwrap()
            .contains("finite F32")
    );
    assert!(
        r["failures"]["nonfinite"]
            .as_str()
            .unwrap()
            .contains("finite BF16")
    );
    assert_eq!(r["failures"].as_object().unwrap().len(), 3);
    assert!(!out.join("adapter.safetensors").exists());
}
