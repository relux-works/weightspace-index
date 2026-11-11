//! Schema controls: the example records pass, and each defect fails.

mod common;

use common::*;
use serde_json::{Value, json};

fn expect(def: &str, v: &Value, ok: bool) {
    let result = records().validate(def, v);
    assert_eq!(result.is_ok(), ok, "{def}: {result:?}\n{v:#}");
}

/// Keywords this schema may use. JSON Schema ignores unknown keywords, so a
/// misspelt one would silently weaken a rule; this list catches that.
const KEYWORDS: &[&str] = &[
    "$schema",
    "$defs",
    "$ref",
    "title",
    "description",
    "type",
    "const",
    "enum",
    "oneOf",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "minItems",
    "maxItems",
    "minimum",
    "pattern",
    "minLength",
];

#[test]
fn schema_uses_only_known_keywords() {
    fn walk(schema: &Value, at: &str) {
        let Value::Object(map) = schema else {
            panic!("{at}: schema is not an object")
        };
        for (k, v) in map {
            assert!(KEYWORDS.contains(&k.as_str()), "{at}: unknown keyword {k}");
            match k.as_str() {
                "$defs" | "properties" => v
                    .as_object()
                    .unwrap()
                    .iter()
                    .for_each(|(n, s)| walk(s, &format!("{at}/{n}"))),
                "items" => walk(v, &format!("{at}/items")),
                "oneOf" => v
                    .as_array()
                    .unwrap()
                    .iter()
                    .for_each(|s| walk(s, &format!("{at}/oneOf"))),
                _ => {}
            }
        }
    }
    walk(&read(&repo().join("schema/records.schema.json")), "#");
}

#[test]
fn shipped_generated_files() {
    expect("bases", &read(&repo().join("bases.json")), true);
}

#[test]
fn bad_record_controls() {
    type Edit = fn(&mut Value);
    let cases: &[(&str, &str, &str, Edit)] = &[
        (
            "missing-digest",
            "version",
            "better-math/v1/version.json",
            |r| remove(&mut r["files"][0], "sha256"),
        ),
        (
            "word-for-digest",
            "version",
            "better-math/v1/version.json",
            |r| r["files"][0]["sha256"] = json!("advertised"),
        ),
        (
            "uppercase-digest",
            "version",
            "better-math/v1/version.json",
            |r| r["files"][0]["sha256"] = json!("A".repeat(64)),
        ),
        (
            "missing-size",
            "version",
            "better-math/v1/version.json",
            |r| remove(&mut r["files"][0], "size_bytes"),
        ),
        ("zero-size", "version", "better-math/v1/version.json", |r| {
            r["files"][0]["size_bytes"] = json!(0)
        }),
        (
            "fractional-size",
            "version",
            "better-math/v1/version.json",
            |r| r["files"][0]["size_bytes"] = json!(0.5),
        ),
        (
            "no-locator",
            "version",
            "better-math/v1/version.json",
            |r| r["files"][0]["locators"] = json!([]),
        ),
        ("no-files", "version", "better-math/v1/version.json", |r| {
            r["files"] = json!([])
        }),
        (
            "unknown-role",
            "version",
            "better-math/v1/version.json",
            |r| r["files"][0]["role"] = json!("submitted"),
        ),
        (
            "derived-without-inputs",
            "version",
            "better-math/v1/version.json",
            |r| remove(&mut r["files"][1], "derived_from"),
        ),
        (
            "derived-without-tool",
            "version",
            "better-math/v1/version.json",
            |r| remove(&mut r["files"][1], "tool"),
        ),
        (
            "source-with-inputs",
            "version",
            "better-math/v1/version.json",
            |r| r["files"][0]["derived_from"] = r["files"][1]["derived_from"].clone(),
        ),
        (
            "file-name-with-path",
            "version",
            "better-math/v1/version.json",
            |r| r["files"][0]["name"] = json!("files/replacements.safetensors"),
        ),
        (
            "hidden-operational-field",
            "version",
            "better-math/v1/version.json",
            |r| r["ignore_mismatch"] = json!(true),
        ),
        (
            "unsupported-contract",
            "version",
            "better-math/v1/version.json",
            |r| r["contract_version"] = json!(2),
        ),
        (
            "authored-targets",
            "version",
            "better-math/v1/version.json",
            |r| r["targets"] = json!(["x"]),
        ),
        ("old-kind", "patch", "better-math/patch.json", |r| {
            r["kind"] = json!("sparse-exact")
        }),
        ("whole-model-kind", "patch", "better-math/patch.json", |r| {
            r["kind"] = json!("whole")
        }),
        ("no-authors", "patch", "better-math/patch.json", |r| {
            r["authors"] = json!([])
        }),
        ("bad-submitter", "patch", "better-math/patch.json", |r| {
            r["submitter"] = json!("not a handle")
        }),
        ("uppercase-tag", "patch", "better-math/patch.json", |r| {
            r["tags"] = json!(["Math"])
        }),
        ("no-versions", "patch", "better-math/patch.json", |r| {
            r["versions"] = json!([])
        }),
        ("latest-as-status", "patch", "better-math/patch.json", |r| {
            r["versions"][0]["status"] = json!("latest")
        }),
        (
            "maintainers-field",
            "patch",
            "better-math/patch.json",
            |r| r["maintainers"] = json!(["someone"]),
        ),
        ("no-profiles", "base", "base.json", |r| {
            r["profiles"] = json!([])
        }),
        ("branch-revision", "base", "base.json", |r| {
            r["profiles"][0]["source"]["revision"] = json!("main")
        }),
        ("repo-type-in-source", "base", "base.json", |r| {
            r["profiles"][0]["source"]["repo_type"] = json!("model")
        }),
        ("uppercase-profile-id", "base", "base.json", |r| {
            r["profiles"][0]["id"] = json!("BF16")
        }),
        ("profile-without-files", "base", "base.json", |r| {
            r["profiles"][0]["files"] = json!([])
        }),
        (
            "run-without-outcomes",
            "run",
            "runs/baseline-bf16/run.json",
            |r| remove(r, "outcomes"),
        ),
        (
            "run-zero-denominator",
            "run",
            "runs/baseline-bf16/run.json",
            |r| r["score"]["denominator"] = json!(0),
        ),
        (
            "run-unknown-performer",
            "run",
            "runs/baseline-bf16/run.json",
            |r| r["performed_by"] = json!("someone"),
        ),
        (
            "run-unknown-score-without-owner",
            "run",
            "runs/baseline-bf16/run.json",
            |r| r["score"] = json!({"state": "UNKNOWN", "reason": "not scored"}),
        ),
        (
            "matched-without-denominator",
            "comparison",
            "better-math/v1/comparisons/bf16-accuracy.json",
            |r| remove(r, "denominator"),
        ),
        (
            "matched-without-outcomes",
            "comparison",
            "better-math/v1/comparisons/bf16-accuracy.json",
            |r| remove(r, "matched_outcomes"),
        ),
        (
            "score-in-comparison",
            "comparison",
            "better-math/v1/comparisons/bf16-accuracy.json",
            |r| r["score"] = json!(1),
        ),
        (
            "refused-without-reason",
            "comparison",
            "better-math/v1/comparisons/bf16-accuracy.json",
            |r| {
                r["state"] = json!("REFUSED");
                for k in ["denominator", "matched_outcomes", "uncertainty"] {
                    remove(r, k);
                }
            },
        ),
        (
            "unknown-without-owner",
            "assessment",
            "better-math/v1/assessments/arithmetic-help.json",
            |r| remove(r, "owner"),
        ),
        (
            "unjustified-support",
            "assessment",
            "better-math/v1/assessments/arithmetic-help.json",
            |r| r["decision"] = json!("SUPPORTED"),
        ),
    ];
    for (name, def, path, edit) in cases {
        let mut r = example(path);
        let before = records().validate(def, &r);
        assert!(before.is_ok(), "{name}: unedited record fails: {before:?}");
        edit(&mut r);
        assert!(
            records().validate(def, &r).is_err(),
            "{name}: defect accepted"
        );
    }
}

#[test]
fn decided_assessment() {
    let policy =
        example("better-math/v1/comparisons/bf16-accuracy.json")["matched_outcomes"].clone();
    let mut a = json!({
        "contract_version": 1, "intended_use": "SYNTHETIC", "decision": "INCONCLUSIVE",
        "policy": policy, "comparisons": ["bf16-accuracy"], "reason": "SYNTHETIC control",
    });
    for d in ["SUPPORTED", "REJECT", "INCONCLUSIVE"] {
        a["decision"] = json!(d);
        expect("assessment", &a, true);
    }
    a["comparisons"] = json!([]);
    expect("assessment", &a, false);
    a["comparisons"] = json!(["bf16-accuracy"]);
    remove(&mut a, "policy");
    expect("assessment", &a, false);
}

#[test]
fn generated_shapes() {
    let mut file = example("better-math/v1/version.json")["files"][0].clone();
    file["format"] = json!("safetensors");
    let run = json!({"id": "baseline-bf16", "profile": "bf16", "performed_by": "index",
        "protocol": {"name": "example-protocol", "sha256": "d".repeat(64)}, "state": "COMPLETE",
        "score": {"metric": "accuracy", "value": 0.5, "unit": "fraction", "denominator": 200}});
    let index = json!({
        "contract_version": 1, "base": "example-org/Example-Model-1B",
        "profiles": example("base.json")["profiles"], "baseline_runs": [run],
        "patches": [{
            "slug": "better-math", "title": "Better arithmetic", "summary": "SYNTHETIC", "kind": "replacement",
            "authors": [{"name": "Example Author"}], "submitter": "example-user", "license": "Apache-2.0",
            "tags": [], "latest": 1,
            "versions": [{
                "version": 1, "status": "active", "changelog": "First release.", "target_profile": "bf16",
                "files": [file], "runs": [], "assessments": [],
            }],
        }],
    });
    expect("index", &index, true);
    let mut bad = index.clone();
    remove(&mut bad["patches"][0]["versions"][0], "files");
    expect("index", &bad, false);
    let mut bad = index.clone();
    bad["patches"][0]["versions"][0]["files"][0]["format"] = json!("pickle");
    expect("index", &bad, false);

    expect(
        "digestLookup",
        &json!({"contract_version": 1, "matches": [{"base": "a/b", "profile": "bf16"}]}),
        true,
    );
    expect(
        "digestLookup",
        &json!({"contract_version": 1, "matches": []}),
        false,
    );
    expect(
        "repoLookup",
        &json!({"contract_version": 1, "matches": [{"base": "a/b", "profiles": ["q4"]}]}),
        true,
    );
    expect(
        "repoLookup",
        &json!({"contract_version": 1, "matches": [{"base": "a/b", "profiles": []}]}),
        false,
    );
    expect(
        "bases",
        &json!({"contract_version": 1, "bases": ["a/b/c"]}),
        false,
    );
}

#[test]
fn locator_controls() {
    let local = json!({"kind": "index-path", "path": "files/nested/adapter_model.safetensors"});
    let hf = json!({"kind": "hf", "repo_type": "model", "repo_id": "Example-Publisher/Example.Model_1",
        "revision": "a".repeat(40), "path": "nested/model.part-00001.safetensors"});
    expect("locator", &local, true);
    for (repo_type, ok) in [("model", true), ("dataset", true), ("space", false)] {
        let mut v = hf.clone();
        v["repo_type"] = json!(repo_type);
        expect("locator", &v, ok);
    }
    let paths = [
        "../escape",
        "/absolute",
        "a/../escape",
        "a/./b",
        "a//b",
        "a\\b",
        "a/%2e%2e/b",
        "a/b?x",
        "a/b#x",
        "a/\0b",
        ".hidden/file",
        "a/b/",
        "a/b\n",
        "a/b\r\n",
    ];
    for path in paths {
        for source in [&local, &hf] {
            let mut v = source.clone();
            v["path"] = json!(path);
            expect("locator", &v, false);
        }
    }
    let revisions = [
        "main".to_owned(),
        "v1".into(),
        "abc1234".into(),
        "a".repeat(39),
        "g".repeat(40),
        "a".repeat(40) + "\n",
    ];
    for revision in revisions {
        let mut v = hf.clone();
        v["revision"] = json!(revision);
        expect("locator", &v, false);
    }
    for repo in [
        "name-only",
        "namespace/name/extra",
        "../name",
        "https://example.invalid/model",
        "name/other?x",
    ] {
        let mut v = hf.clone();
        v["repo_id"] = json!(repo);
        expect("locator", &v, false);
    }
    let mut v = local.clone();
    v["storage"] = json!("git-lfs");
    expect("locator", &v, false);
    let mut v = hf.clone();
    v["url"] = json!("https://example.invalid/payload");
    expect("locator", &v, false);
}
