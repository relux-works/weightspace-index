//! Derived experimental LoRA factors. Study claims NEVER transfer to LoRA.
//! Runtime loading in PEFT/llama.cpp/vLLM is UNQUALIFIED.
mod math;

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

use safetensors::Dtype;
use safetensors::tensor::TensorView;
use serde_json::{Value, json};

use crate::error::ToolError;
use crate::files::{OutputTarget, Sha256Hex, TempDir, check_input_dir, hash_file, is_within};
use crate::model_dir::single_weights_file;
use crate::ops::{ApplyReplacement, Cx, Report, verify_patch_against_base};
use crate::patch::{PatchFile, PatchTensor, for_each_index_block};
use math::{bf16, exact_difference, invalid};

pub const DISCLAIMER: &str = "LoRA is a derived form for experimentation/combination; study claims NEVER transfer to it; it needs its own evaluation. PEFT/llama.cpp/vLLM runtime loading is UNQUALIFIED.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Svd,
    Selector,
}

impl Method {
    fn name(self) -> &'static str {
        match self {
            Self::Svd => "svd",
            Self::Selector => "selector",
        }
    }
}

/// F32 factors only; no implicit rank, storage dtype or runtime scale.
pub struct DeriveLora {
    pub model: PathBuf,
    pub patch: PathBuf,
    pub rank: usize,
    pub method: Method,
    pub out: PathBuf,
    /// Same full-tensor wrong-base gate as `apply`.
    pub expect_tensor_sha256: BTreeMap<String, Sha256Hex>,
}

/// Derive only the patch payload's named BF16 matrices. Input snapshots,
/// sparse old-value verification and full-form digest gates are shared with apply.
/// All analysis completes before the output directory is published.
pub fn derive_lora(req: &DeriveLora, cx: &mut Cx) -> Result<Report, ToolError> {
    let out = OutputTarget::check(&req.out)?;
    check_input_dir(&req.model)?;
    let model = fs::canonicalize(&req.model).map_err(|e| ToolError::io(&req.model, e))?;
    if is_within(out.parent(), &model) {
        return Err(ToolError::OutputInsideInput {
            output: req.out.clone(),
            model: req.model.clone(),
        });
    }
    if req.rank == 0 {
        return Err(invalid("derive-lora rank must be positive"));
    }
    let weights = req.model.join(single_weights_file(&req.model)?);
    let apply = ApplyReplacement {
        model: req.model.clone(),
        patch: req.patch.clone(),
        out: req.out.clone(),
        expect_tensor_sha256: req.expect_tensor_sha256.clone(),
    };
    let (patch, base, entries_changed) = verify_patch_against_base(&apply, &weights, cx)?;
    let mut factors = BTreeMap::new();
    let mut reports = BTreeMap::new();
    for (name, tensor) in &patch.tensors {
        cx.check()?;
        let meta = base.tensor(name)?;
        if meta.dtype != Dtype::BF16 || meta.shape.len() != 2 || meta.shape.contains(&0) {
            return Err(invalid(format!(
                "{name}: derive-lora requires a nonempty 2-D BF16 matrix"
            )));
        }
        let rows =
            usize::try_from(meta.shape[0]).map_err(|_| invalid("matrix rows exceed usize"))?;
        let width =
            usize::try_from(meta.shape[1]).map_err(|_| invalid("matrix width exceeds usize"))?;
        let old = decode(base.tensor_bytes(meta)?);
        let new = reconstruct(&patch, tensor, name, &old, cx)?;
        let mut active = vec![false; width];
        let mut delta = Vec::with_capacity(old.len());
        for (i, (&a, &b)) in new.iter().zip(&old).enumerate() {
            if i % 65536 == 0 {
                cx.check()?;
            }
            delta.push(exact_difference(a, b)?);
            active[i % width] |= a != b;
        }
        let columns: Vec<usize> = (0..width).filter(|&j| active[j]).collect();
        let k = columns.len();
        if req.rank > k {
            return Err(invalid(format!(
                "{name}: rank {} exceeds active columns k={k}",
                req.rank
            )));
        }
        if req.method == Method::Selector && req.rank != k {
            return Err(invalid(format!("{name}: selector requires rank=k={k}")));
        }
        let thin = rows.min(k);
        if req.method == Method::Svd && req.rank > thin {
            return Err(invalid(format!(
                "{name}: SVD rank exceeds min(rows,k)={thin}"
            )));
        }
        let compact: Vec<f64> = (0..rows)
            .flat_map(|i| {
                columns.iter().map({
                    let delta = &delta;
                    move |&j| delta[i * width + j]
                })
            })
            .collect();
        let spectrum = math::svd(&compact, rows, k, cx)?;
        let bounds = math::modular_ranks(&old, &new, width, &columns, cx)?;
        let lower = *bounds.iter().max().expect("two primes");
        let exact_rank = if lower == thin { Some(thin) } else { None };
        let (a, b) = factors_for(req, rows, width, &columns, &compact, &spectrum);
        let a = stored_f32(a)?;
        let b = stored_f32(b)?;
        let mut report = measure(&old, &new, &delta, &active, &a, &b, req.rank, cx)?;
        report["shape"] = json!([rows, width]);
        report["active_columns_k"] = json!(k);
        report["active_column_indices"] = json!(columns);
        report["singular_values_used"] = if req.method == Method::Svd {
            json!(&spectrum.singular[..req.rank])
        } else {
            json!([])
        };
        report["computed_active_singular_values"] = json!(spectrum.singular);
        report["structural_zero_singular_values"] = json!(rows.min(width) - thin);
        report["jacobi_sweeps"] = json!(spectrum.sweeps);
        report["exact_rank_evidence"] = json!({
            "integer_scale_power_of_two": 133, "primes": math::PRIMES,
            "modular_ranks": bounds, "lower_bound": lower, "upper_bound": thin,
            "proven_exact_rank": exact_rank,
            "interpretation": "Modular ranks are lower bounds; equality to the structural upper bound proves exact rank. A smaller bound does not prove deficiency."
        });
        let a_name = format!("{name}.lora_A");
        let b_name = format!("{name}.lora_B");
        report["factor_names"] = json!({"A": a_name, "B": b_name});
        report["factor_shapes"] = json!({"A": [req.rank, width], "B": [rows, req.rank]});
        report["raw_factor_bytes"] =
            json!({"A": a.len() * 4, "B": b.len() * 4, "total": (a.len() + b.len()) * 4});
        factors.insert(a_name, (vec![req.rank, width], bytes(&a)));
        factors.insert(b_name, (vec![rows, req.rank], bytes(&b)));
        reports.insert(name.clone(), report);
    }
    let report = json!({
        "format_version": 1, "notice": DISCLAIMER, "method": req.method.name(),
        "factor_dtype": "F32", "rank": req.rank, "effective_scale": 1,
        "tensor_name_mapping": "Append literal .lora_A and .lora_B to each original UTF-8 tensor name; A=[rank,width], B=[rows,rank]; row-major, no name rewriting or transpose.",
        "runtime_loading": {"PEFT": "UNQUALIFIED", "llama.cpp": "UNQUALIFIED", "vLLM": "UNQUALIFIED"},
        "frozen_bake": "Stored F32 factors widened to F64; rank-ordered F64 multiply/accumulate; add exact BF16 base in F64; cast to F32; BF16 round-to-nearest-ties-even. Not a runtime kernel simulation.",
        "base_safetensors_sha256": base.snapshot_sha256().to_string(),
        "patch_safetensors_sha256": patch.checkpoint.snapshot_sha256().to_string(),
        "tensors": reports
    });
    let temp = TempDir::create_in(out.parent())?;
    let adapter = temp.path().join("adapter.safetensors");
    let views = factors
        .iter()
        .map(|(name, (shape, data))| {
            TensorView::new(Dtype::F32, shape.clone(), data)
                .map(|view| (name.as_str(), view))
                .map_err(|e| invalid(e.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    safetensors::tensor::serialize_to_file(views, None, &adapter)
        .map_err(|e| invalid(e.to_string()))?;
    OpenOptions::new()
        .write(true)
        .open(&adapter)
        .and_then(|f| f.sync_all())
        .map_err(|e| ToolError::io(&adapter, e))?;
    let report_path = temp.path().join("lora-report.json");
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&report_path)
        .map_err(|e| ToolError::io(&report_path, e))?;
    let mut report_bytes =
        serde_json::to_vec_pretty(&report).map_err(|e| invalid(e.to_string()))?;
    report_bytes.push(b'\n');
    file.write_all(&report_bytes)
        .and_then(|()| file.sync_all())
        .map_err(|e| ToolError::io(&report_path, e))?;
    drop(file);
    let out_sha256 = hash_file(&adapter, cx.cancelled)?;
    let out_bytes = fs::metadata(&adapter)
        .map_err(|e| ToolError::io(&adapter, e))?
        .len();
    cx.check()?;
    temp.publish(&out)?;
    Ok(Report {
        tensors_changed: reports.len() as u64,
        entries_changed,
        out_sha256,
        out_bytes,
    })
}

fn decode(data: &[u8]) -> Vec<u16> {
    data.chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}

fn reconstruct(
    patch: &PatchFile,
    tensor: &PatchTensor,
    name: &str,
    old: &[u16],
    cx: &Cx,
) -> Result<Vec<u16>, ToolError> {
    match tensor {
        PatchTensor::Full { meta } => Ok(decode(patch.checkpoint.tensor_bytes(meta)?)),
        PatchTensor::Sparse { idx, new, .. } => {
            let mut result = old.to_vec();
            let values = decode(patch.checkpoint.tensor_bytes(new)?);
            for_each_index_block(
                &patch.checkpoint,
                name,
                idx,
                cx.cancelled,
                |block, ordinal| {
                    cx.check()?;
                    for (i, &index) in block.iter().enumerate() {
                        result[index as usize] = values[ordinal as usize + i];
                    }
                    Ok(())
                },
            )?;
            Ok(result)
        }
    }
}

fn factors_for(
    req: &DeriveLora,
    rows: usize,
    width: usize,
    columns: &[usize],
    compact: &[f64],
    spectrum: &math::Svd,
) -> (Vec<f64>, Vec<f64>) {
    let rank = req.rank;
    let k = columns.len();
    let thin = spectrum.singular.len();
    let mut a = vec![0.0; rank * width];
    let mut b = vec![0.0; rows * rank];
    for j in 0..rank {
        match req.method {
            Method::Selector => {
                a[j * width + columns[j]] = 1.0;
                for i in 0..rows {
                    b[i * rank + j] = compact[i * k + j];
                }
            }
            Method::Svd => {
                let root = spectrum.singular[j].sqrt();
                for (i, &col) in columns.iter().enumerate() {
                    a[j * width + col] = root * spectrum.v[i * thin + j];
                }
                for i in 0..rows {
                    b[i * rank + j] = spectrum.u[i * thin + j] * root;
                }
            }
        }
    }
    (a, b)
}

fn stored_f32(values: Vec<f64>) -> Result<Vec<f32>, ToolError> {
    values
        .into_iter()
        .map(|x| {
            let stored = x as f32;
            if stored.is_finite() {
                Ok(stored)
            } else {
                Err(invalid("factor is not representable as finite F32"))
            }
        })
        .collect()
}
fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[allow(clippy::too_many_arguments)]
fn measure(
    old: &[u16],
    new: &[u16],
    delta: &[f64],
    active: &[bool],
    a: &[f32],
    b: &[f32],
    rank: usize,
    cx: &Cx,
) -> Result<Value, ToolError> {
    let width = active.len();
    let mut delta_error = 0.0f64;
    let mut delta_norm = 0.0f64;
    let mut baked_error = 0.0f64;
    let mut mismatches = 0u64;
    let mut max_ulp = 0u16;
    let mut ulp_all = 0u64;
    let mut ulp_active = 0u64;
    for index in 0..old.len() {
        if index % 65536 == 0 {
            cx.check()?;
        }
        let i = index / width;
        let j = index % width;
        let mut reconstructed = 0.0;
        for t in 0..rank {
            reconstructed += f64::from(b[i * rank + t]) * f64::from(a[t * width + j]);
        }
        delta_error = delta_error.hypot(reconstructed - delta[index]);
        delta_norm = delta_norm.hypot(delta[index]);
        let baked = math::bake(bf16(old[index]) + reconstructed)?;
        mismatches += u64::from(baked != new[index]);
        let ulp = math::ordered(baked).abs_diff(math::ordered(new[index]));
        max_ulp = max_ulp.max(ulp);
        ulp_all += u64::from(ulp);
        if active[j] {
            ulp_active += u64::from(ulp);
        }
        baked_error = baked_error.hypot(bf16(baked) - bf16(new[index]));
    }
    let active_elements = (old.len() / width) * active.iter().filter(|&&v| v).count();
    Ok(json!({
        "delta_frobenius_error": delta_error,
        "exact_delta_frobenius_norm": delta_norm,
        "relative_delta_error": if delta_norm == 0.0 { None } else { Some(delta_error / delta_norm) },
        "bf16_mismatch_count": mismatches, "max_ulp": max_ulp,
        "mean_ulp_all_elements": ulp_all as f64 / old.len() as f64,
        "mean_ulp_active_columns": ulp_active as f64 / active_elements as f64,
        "all_elements": old.len(), "active_elements": active_elements,
        "baked_frobenius_error": baked_error
    }))
}

#[cfg(test)]
mod tests;
