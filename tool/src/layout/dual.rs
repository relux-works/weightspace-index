//! Additive v1 profile: the canonical derived adapter/report pair claims dual
//! publication. Historical v1 versions without the pair remain valid.
use super::*;
use crate::checkpoint::Checkpoint;
use crate::derive_lora::check_exact_publication;

pub(super) fn check(dir: &Path, kind: &str, files: &[Value]) -> Result<(), String> {
    let find = |name: &str| files.iter().find(|f| text(f, "name") == name);
    let adapter = find("adapter.safetensors");
    let report = find("lora-report.json");
    if adapter.is_none() && report.is_none() {
        return Ok(());
    }
    let error = |detail: &str| format!("{}: dual publication: {detail}", dir.display());
    let (Some(adapter), Some(report)) = (adapter, report) else {
        return Err(error(
            "requires both adapter.safetensors and lora-report.json",
        ));
    };
    if kind != "replacement" {
        return Err(error("canonical pair requires replacement sources"));
    }
    if files.iter().filter(|f| text(f, "role") == "source").count() != 1 {
        return Err(error(
            "canonical pair requires exactly one replacement source; multi-source coverage is unimplemented",
        ));
    }
    for f in [adapter, report] {
        if text(f, "role") != "derived" || text(&f["tool"], "name") != "weightspace derive-lora" {
            return Err(error(
                "adapter and report must be derived with weightspace derive-lora provenance",
            ));
        }
    }
    if adapter["derived_from"] != report["derived_from"] || adapter["tool"] != report["tool"] {
        return Err(error("adapter/report provenance differs"));
    }
    let local = |f: &Value| -> Result<PathBuf, String> {
        let paths: Vec<_> = items(f, "locators")
            .iter()
            .filter(|l| text(l, "kind") == "index-path")
            .collect();
        if paths.len() != 1 {
            return Err(error(
                "canonical pair and its source need one hosted index-path each for offline verification",
            ));
        }
        Ok(dir.join(text(paths[0], "path")))
    };
    let r = read_json(&local(report)?)?;
    let source = files
        .iter()
        .find(|f| text(f, "role") == "source" && f["sha256"] == r["patch_safetensors_sha256"])
        .ok_or_else(|| error("report patch digest is not a source of this version"))?;
    if adapter["derived_from"] != serde_json::json!([source["sha256"]]) {
        return Err(error(
            "pair must cite exactly its report's replacement source digest",
        ));
    }
    if r["tool"] != adapter["tool"]
        || r["adapter_safetensors_sha256"] != adapter["sha256"]
        || r["adapter_size_bytes"] != adapter["size_bytes"]
    {
        return Err(error(
            "report tool or adapter digest/size disagrees with records",
        ));
    }
    let a = Checkpoint::open(&local(adapter)?).map_err(|e| error(&e.to_string()))?;
    if a.snapshot_sha256().to_string() != text(adapter, "sha256") {
        return Err(error("adapter digest differs from report/record"));
    }
    check_exact_publication(&local(source)?, &a, &r).map_err(|e| error(&e.to_string()))
}
