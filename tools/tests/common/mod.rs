#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use serde_json::Value;
use weightspace_tools::records::{Records, read_json};

pub fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

static RECORDS: LazyLock<Records> =
    LazyLock::new(|| Records::load(&repo().join("schema/records.schema.json")).unwrap());

/// The compiled schema, built once per test binary.
pub fn records() -> &'static Records {
    &RECORDS
}

pub fn example_models() -> PathBuf {
    repo().join("examples/models")
}

/// The synthetic example base, relative to a models/ root.
pub const BASE: &str = "example-org/Example-Model-1B";

pub fn example(rel: &str) -> Value {
    read(&example_models().join(BASE).join(rel))
}

pub fn read(path: &Path) -> Value {
    read_json(path).unwrap()
}

pub fn write(path: &Path, v: &Value) {
    fs::write(path, serde_json::to_vec_pretty(v).unwrap()).unwrap();
}

pub fn edit(path: &Path, f: impl FnOnce(&mut Value)) {
    let mut v = read(path);
    f(&mut v);
    write(path, &v);
}

pub fn remove(v: &mut Value, key: &str) {
    v.as_object_mut().unwrap().remove(key);
}

pub fn copy_dir(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for entry in fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), target).unwrap();
        }
    }
}
