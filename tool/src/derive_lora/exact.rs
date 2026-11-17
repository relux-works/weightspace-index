//! Per-tensor algebraic selectors, with stored-factor and frozen-bake evidence.
//! Offline sparse inputs prove only saved coordinates, never unseen base bytes.
use super::*;
use crate::checkpoint::Checkpoint;
use crate::patch::parse_patch_in;
use sha2::{Digest, Sha256};
use std::path::Path;

pub const METHOD: &str = "selector-exact";
pub const TOOL: &str = "weightspace derive-lora";

pub struct DeriveExactLora {
    pub model: Option<PathBuf>,
    /// Strict JSON object mapping every patch tensor name to [rows, width].
    /// Only sparse inputs can be derived without the base.
    pub tensor_shapes: Option<PathBuf>,
    pub patch: PathBuf,
    pub out: PathBuf,
    pub expect_tensor_sha256: BTreeMap<String, Sha256Hex>,
}

type Factors = BTreeMap<String, (Vec<usize>, Vec<u8>)>;

pub fn derive_exact_lora(req: &DeriveExactLora, cx: &mut Cx) -> Result<Report, ToolError> {
    let out = OutputTarget::check(&req.out)?;
    let (patch, base, shapes, shape_digest) = match (&req.model, &req.tensor_shapes) {
        (Some(model), None) => {
            check_input_dir(model)?;
            let canonical = fs::canonicalize(model).map_err(|e| ToolError::io(model, e))?;
            if is_within(out.parent(), &canonical) {
                return Err(ToolError::OutputInsideInput {
                    output: req.out.clone(),
                    model: model.clone(),
                });
            }
            let apply = ApplyReplacement {
                model: model.clone(),
                patch: req.patch.clone(),
                out: req.out.clone(),
                expect_tensor_sha256: req.expect_tensor_sha256.clone(),
            };
            let (patch, base, _) =
                verify_patch_against_base(&apply, &model.join(single_weights_file(model)?), cx)?;
            let shapes = patch
                .tensors
                .keys()
                .map(|name| Ok((name.clone(), base.tensor(name)?.shape.clone())))
                .collect::<Result<BTreeMap<_, _>, ToolError>>()?;
            (patch, Some(base), shapes, None)
        }
        (None, Some(path)) => {
            if !req.expect_tensor_sha256.is_empty() {
                return Err(invalid(
                    "offline shapes cannot verify expected base digests",
                ));
            }
            let patch = parse_patch_in(&req.patch, cx.temp_dir(), cx.cancelled)?;
            let data = fs::read(path).map_err(|e| ToolError::io(path, e))?;
            let shapes: BTreeMap<String, Vec<u64>> = serde_json::from_slice(&data)
                .map_err(|e| invalid(format!("tensor-shapes: {e}")))?;
            if shapes.keys().ne(patch.tensors.keys()) {
                return Err(invalid(
                    "tensor-shapes must name exactly every patch tensor",
                ));
            }
            let digest = Sha256Hex::from_digest(&Sha256::digest(&data)).to_string();
            (patch, None, shapes, Some(digest))
        }
        _ => {
            return Err(invalid(
                "selector-exact needs exactly one of --model or --tensor-shapes",
            ));
        }
    };
    let mut factors = Factors::new();
    let mut tensors = BTreeMap::new();
    let mut failures = BTreeMap::new();
    let mut entries = 0;
    for (name, tensor) in &patch.tensors {
        cx.check()?;
        match analyze(&patch, tensor, name, base.as_ref(), &shapes[name], cx) {
            Ok((values, report, count)) => {
                factors.extend(values);
                tensors.insert(name.clone(), report);
                entries += count;
            }
            Err(ToolError::InvalidArgument { detail }) => {
                failures.insert(name.clone(), detail);
            }
            Err(e) => return Err(e),
        }
    }
    let mut report = json!({
        "format_version": 2, "status": if failures.is_empty() {"complete"} else {"unsupported"},
        "notice": DISCLAIMER, "method": METHOD, "rank": null,
        "rank_policy": "per-tensor active columns k_t (zero permitted)",
        "factor_dtype": "F32", "effective_scale": 1,
        "tool": {"name": TOOL, "version": env!("CARGO_PKG_VERSION")},
        "input_mode": if base.is_some() {"verified-base"} else {"sparse-old-values"},
        "tensor_shapes": shapes, "tensor_shapes_sha256": shape_digest,
        "base_safetensors_sha256": base.as_ref().map(|b| b.snapshot_sha256().to_string()),
        "patch_safetensors_sha256": patch.checkpoint.snapshot_sha256().to_string(),
        "frozen_bake": "Stored F32 selector factors widened to F64; rank-ordered F64 multiply/accumulate; add BF16 old value in F64; cast to F32; BF16 round-to-nearest-ties-even. Not a runtime kernel simulation.",
        "exactness": "Exact refers to the algebraic selector before F32 storage; stored delta error and BF16 bake mismatches are measured, never a usefulness or sparsity gate. Offline metrics cover saved coordinates only; unseen base signed zeros/nonfinite values are unverified.",
        "runtime_loading": {"PEFT": "UNQUALIFIED", "llama.cpp": "UNQUALIFIED", "vLLM": "UNQUALIFIED"},
        "tensors": tensors, "failures": failures,
    });
    let temp = TempDir::create_in(out.parent())?;
    let adapter = temp.path().join("adapter.safetensors");
    let result = if failures.is_empty() {
        let views = factors
            .iter()
            .map(|(name, (shape, data))| {
                TensorView::new(Dtype::F32, shape.clone(), data)
                    .map(|v| (name.as_str(), v))
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
        let digest = hash_file(&adapter, cx.cancelled)?;
        let size = fs::metadata(&adapter)
            .map_err(|e| ToolError::io(&adapter, e))?
            .len();
        report["adapter_safetensors_sha256"] = json!(digest.to_string());
        report["adapter_size_bytes"] = json!(size);
        Some((digest, size))
    } else {
        None
    };
    write_report(&temp.path().join("lora-report.json"), &report)?;
    cx.check()?;
    temp.publish(&out)?;
    match result {
        Some((out_sha256, out_bytes)) => Ok(Report {
            tensors_changed: tensors.len() as u64,
            entries_changed: entries,
            out_sha256,
            out_bytes,
        }),
        None => Err(invalid(
            "unsupported tensor coverage; lora-report.json published without an adapter; see failures",
        )),
    }
}

fn write_report(path: &Path, report: &Value) -> Result<(), ToolError> {
    let mut data = serde_json::to_vec_pretty(report).map_err(|e| invalid(e.to_string()))?;
    data.push(b'\n');
    let mut f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| ToolError::io(path, e))?;
    f.write_all(&data)
        .and_then(|()| f.sync_all())
        .map_err(|e| ToolError::io(path, e))
}

/// Reused by offline publication validation; no numerical decomposition or gate.
fn analyze(
    patch: &PatchFile,
    tensor: &PatchTensor,
    name: &str,
    base: Option<&Checkpoint>,
    shape: &[u64],
    cx: &Cx,
) -> Result<(Factors, Value, u64), ToolError> {
    let dtype = match tensor {
        PatchTensor::Full { meta } => meta.dtype,
        PatchTensor::Sparse { new, .. } => new.dtype,
    };
    if dtype != Dtype::BF16 || shape.len() != 2 || shape.contains(&0) {
        return Err(invalid("requires a nonempty 2-D BF16 matrix"));
    }
    let rows = usize::try_from(shape[0]).map_err(|_| invalid("rows exceed usize"))?;
    let width = usize::try_from(shape[1]).map_err(|_| invalid("width exceeds usize"))?;
    let numel = rows
        .checked_mul(width)
        .ok_or_else(|| invalid("shape product exceeds usize"))?;
    // (flat coordinate, old bits, new bits). Offline never invents unseen old bytes.
    let pairs: Vec<(usize, u16, u16)> = if let Some(base) = base {
        let meta = base.tensor(name)?;
        let old = decode(base.tensor_bytes(meta)?);
        let new = reconstruct(patch, tensor, name, &old, cx)?;
        old.into_iter()
            .zip(new)
            .enumerate()
            .map(|(i, (o, n))| (i, o, n))
            .collect()
    } else {
        let PatchTensor::Sparse { idx, new, old, .. } = tensor else {
            return Err(invalid(
                "full-form replacement requires --model: no old values are stored",
            ));
        };
        let old = decode(patch.checkpoint.tensor_bytes(old)?);
        let new = decode(patch.checkpoint.tensor_bytes(new)?);
        let mut pairs = Vec::with_capacity(old.len());
        for_each_index_block(
            &patch.checkpoint,
            name,
            idx,
            cx.cancelled,
            |block, ordinal| {
                for (i, &index) in block.iter().enumerate() {
                    let index =
                        usize::try_from(index).map_err(|_| invalid("index exceeds usize"))?;
                    if index >= numel {
                        return Err(invalid("sparse index outside declared tensor shape"));
                    }
                    let p = ordinal as usize + i;
                    pairs.push((index, old[p], new[p]));
                }
                Ok(())
            },
        )?;
        pairs
    };
    let mut active = vec![false; width];
    for &(index, old, new) in &pairs {
        active[index % width] |= old != new;
    }
    let columns: Vec<usize> = (0..width).filter(|&j| active[j]).collect();
    let rank = columns.len();
    let mut slots = vec![None; width];
    for (t, &j) in columns.iter().enumerate() {
        slots[j] = Some(t);
    }
    let a_len = rank
        .checked_mul(width)
        .ok_or_else(|| invalid("A length overflow"))?;
    let b_len = rows
        .checked_mul(rank)
        .ok_or_else(|| invalid("B length overflow"))?;
    let mut a = vec![0f32; a_len];
    let mut b = vec![0f32; b_len];
    for (t, &j) in columns.iter().enumerate() {
        a[t * width + j] = 1.0;
    }
    let mut delta_error = 0f64;
    let mut delta_norm = 0f64;
    let mut baked_error = 0f64;
    let mut mismatches = 0u64;
    let mut max_ulp = 0u16;
    let mut ulp_all = 0u64;
    let mut ulp_active = 0u64;
    let mut measured_active = 0usize;
    let mut bake_failures = 0usize;
    for (p, &(index, old, new)) in pairs.iter().enumerate() {
        if p % 65536 == 0 {
            cx.check()?;
        }
        let delta = exact_difference(new, old)?;
        let stored = delta as f32;
        if !stored.is_finite() {
            return Err(invalid("difference cannot be stored in finite F32"));
        }
        if let Some(t) = slots[index % width] {
            b[(index / width) * rank + t] = stored;
        }
        // The one-hot product has one nonzero term. Include the rank-ordered
        // accumulator's +0 convention, including signed-zero-only changes.
        let reconstructed = 0.0 + f64::from(stored);
        delta_error = delta_error.hypot(reconstructed - delta);
        delta_norm = delta_norm.hypot(delta);
        match math::bake(bf16(old) + reconstructed) {
            Ok(baked) => {
                mismatches += u64::from(baked != new);
                let ulp = math::ordered(baked).abs_diff(math::ordered(new));
                max_ulp = max_ulp.max(ulp);
                ulp_all += u64::from(ulp);
                if active[index % width] {
                    ulp_active += u64::from(ulp);
                }
                baked_error = baked_error.hypot(bf16(baked) - bf16(new));
            }
            Err(_) => {
                bake_failures += 1;
            }
        }
        measured_active += usize::from(active[index % width]);
    }
    let a_name = format!("{name}.lora_A");
    let b_name = format!("{name}.lora_B");
    let report = json!({
        "shape": shape, "rank": rank, "active_columns_k": rank,
        "active_column_indices": columns,
        "factor_names": {"A": a_name, "B": b_name},
        "factor_shapes": {"A": [rank,width], "B": [rows,rank]},
        "raw_factor_bytes": {"A": a_len * 4, "B": b_len * 4, "total": (a_len+b_len)*4},
        "verification_scope": if base.is_some() {"full-tensor"} else {"saved-coordinates-only"},
        "all_elements": numel, "measured_elements": pairs.len(), "measured_active_elements": measured_active,
        "delta_frobenius_error": delta_error, "exact_delta_frobenius_norm": delta_norm,
        "relative_delta_error": if delta_norm == 0.0 {None} else {Some(delta_error/delta_norm)},
        "bf16_mismatch_count": mismatches, "bake_failure_count": bake_failures,
        "max_ulp": max_ulp, "mean_ulp_measured_elements": ulp_all as f64 / pairs.len() as f64,
        "mean_ulp_measured_active_elements": if measured_active == 0 {None} else {Some(ulp_active as f64 / measured_active as f64)},
        "baked_frobenius_error": baked_error,
        "stored_delta_exact": delta_error == 0.0,
        "measured_bake_exact": mismatches == 0 && bake_failures == 0,
    });
    let factors = BTreeMap::from([
        (a_name, (vec![rank, width], bytes(&a))),
        (b_name, (vec![rows, rank], bytes(&b))),
    ]);
    Ok((
        factors,
        report,
        pairs.iter().filter(|(_, o, n)| o != n).count() as u64,
    ))
}

/// Checks format/method/ranks, source coverage, factor descriptors and bytes.
/// Sparse-old mode also independently rebuilds the selectors and measured
/// tensor reports from the snapshot source plus the report's explicit shapes.
pub fn check_exact_publication(
    patch_path: &Path,
    adapter: &Checkpoint,
    r: &Value,
) -> Result<(), ToolError> {
    if r["format_version"] != 2
        || r["status"] != "complete"
        || r["method"] != METHOD
        || r["factor_dtype"] != "F32"
        || r["effective_scale"] != 1
        || !r["rank"].is_null()
        || r["failures"].as_object().is_none_or(|f| !f.is_empty())
    {
        return Err(invalid(
            "report needs complete selector-exact format 2, F32, scale 1 and per-tensor ranks",
        ));
    }
    let offline = match r["input_mode"].as_str() {
        Some("sparse-old-values") if r["base_safetensors_sha256"].is_null() => true,
        Some("verified-base")
            if r["base_safetensors_sha256"]
                .as_str()
                .is_some_and(|s| Sha256Hex::parse(s).is_ok()) =>
        {
            false
        }
        _ => return Err(invalid("invalid report input_mode/base digest")),
    };
    if offline
        && !r["tensor_shapes_sha256"]
            .as_str()
            .is_some_and(|s| Sha256Hex::parse(s).is_ok())
    {
        return Err(invalid("offline report requires shape-map digest"));
    }
    let patch = crate::patch::parse_patch(patch_path)?;
    if r["patch_safetensors_sha256"] != patch.checkpoint.snapshot_sha256().to_string() {
        return Err(invalid(
            "report source digest differs from replacement bytes",
        ));
    }
    let shapes: BTreeMap<String, Vec<u64>> = serde_json::from_value(r["tensor_shapes"].clone())
        .map_err(|_| invalid("invalid report tensor_shapes"))?;
    let tensors = r["tensors"]
        .as_object()
        .ok_or_else(|| invalid("missing tensor reports"))?;
    if shapes.keys().ne(patch.tensors.keys()) || tensors.keys().ne(patch.tensors.keys()) {
        return Err(invalid(
            "report must cover every source tensor exactly once",
        ));
    }
    if adapter.tensors.len() != patch.tensors.len() * 2 {
        return Err(invalid("extra/missing adapter factors"));
    }
    let cancelled = std::sync::atomic::AtomicBool::new(false);
    let mut sink = |_| {};
    let cx = Cx::new(&cancelled, &mut sink);
    for (name, tensor) in &patch.tensors {
        let t = &tensors[name];
        let shape = &shapes[name];
        if shape.len() != 2 || shape.contains(&0) || t["shape"] != json!(shape) {
            return Err(invalid("invalid tensor shape in report"));
        }
        let columns: Vec<usize> = serde_json::from_value(t["active_column_indices"].clone())
            .map_err(|_| invalid("invalid active columns"))?;
        let rank = columns.len();
        if columns.windows(2).any(|w| w[0] >= w[1])
            || columns.iter().any(|&j| j as u64 >= shape[1])
            || t["rank"] != rank
            || t["active_columns_k"] != rank
        {
            return Err(invalid("report rank/active columns disagree"));
        }
        let mut raw_total = 0;
        for (suffix, dims) in [
            ("A", vec![rank as u64, shape[1]]),
            ("B", vec![shape[0], rank as u64]),
        ] {
            let factor_name = format!("{name}.lora_{suffix}");
            let meta = adapter.tensor(&factor_name)?;
            if meta.dtype != Dtype::F32
                || meta.shape != dims
                || t["factor_names"][suffix] != factor_name
                || t["factor_shapes"][suffix] != json!(dims)
                || t["raw_factor_bytes"][suffix] != meta.byte_len()
            {
                return Err(invalid("factor descriptor disagrees with report"));
            }
            if adapter
                .tensor_bytes(meta)?
                .chunks_exact(4)
                .any(|b| !f32::from_le_bytes(b.try_into().expect("four bytes")).is_finite())
            {
                return Err(invalid("nonfinite stored factor"));
            }
            raw_total += meta.byte_len();
        }
        if t["raw_factor_bytes"]["total"] != raw_total {
            return Err(invalid("factor byte total disagrees"));
        }
        if offline {
            let (expected, metrics, _) = analyze(&patch, tensor, name, None, shape, &cx)?;
            if *t != metrics {
                return Err(invalid(
                    "offline tensor report differs from rebuilt source evidence",
                ));
            }
            for (name, (_, bytes)) in expected {
                if adapter.tensor_bytes(adapter.tensor(&name)?)? != bytes {
                    return Err(invalid(
                        "stored factors differ from rebuilt source selectors",
                    ));
                }
            }
        } else {
            if t["verification_scope"] != "full-tensor" {
                return Err(invalid("verified-base report requires full-tensor scope"));
            }
            let a_name = format!("{name}.lora_A");
            let a = adapter.tensor_bytes(adapter.tensor(&a_name)?)?;
            let width = usize::try_from(shape[1]).map_err(|_| invalid("width exceeds usize"))?;
            for (index, bytes) in a.chunks_exact(4).enumerate() {
                let expected = if index % width == columns[index / width] {
                    1f32
                } else {
                    0f32
                };
                if bytes != expected.to_le_bytes() {
                    return Err(invalid("A is not the reported column selector"));
                }
            }
            match tensor {
                PatchTensor::Sparse { .. } => {
                    let (expected, _, _) = analyze(&patch, tensor, name, None, shape, &cx)?;
                    for (name, (_, bytes)) in expected {
                        if adapter.tensor_bytes(adapter.tensor(&name)?)? != bytes {
                            return Err(invalid(
                                "stored factors differ from rebuilt source selectors",
                            ));
                        }
                    }
                }
                PatchTensor::Full { meta } => {
                    if meta.dtype != Dtype::BF16 || meta.shape != *shape {
                        return Err(invalid("full tensor shape/dtype disagrees with report"));
                    }
                }
            }
        }
    }
    Ok(())
}
