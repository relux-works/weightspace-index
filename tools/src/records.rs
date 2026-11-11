//! Validation of single JSON records against the index schema.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use jsonschema::Validator;
use serde_json::{Map, Value};

/// One compiled validator per definition in `records.schema.json`.
pub struct Records {
    validators: BTreeMap<String, Validator>,
}

impl Records {
    /// Loads the schema and compiles a validator for each `$defs` entry.
    pub fn load(path: &Path) -> Result<Self, String> {
        let root = read_json(path)?;
        let names: Vec<String> = root
            .get("$defs")
            .and_then(Value::as_object)
            .ok_or_else(|| format!("{}: no $defs", path.display()))?
            .keys()
            .cloned()
            .collect();
        let mut validators = BTreeMap::new();
        for name in names {
            let mut schema = root.clone();
            schema
                .as_object_mut()
                .expect("schema root is an object")
                .insert("$ref".into(), Value::String(format!("#/$defs/{name}")));
            let validator = jsonschema::draft202012::new(&schema)
                .map_err(|e| format!("{}: {name}: {e}", path.display()))?;
            validators.insert(name, validator);
        }
        Ok(Self { validators })
    }

    /// Validates `value` against the definition `def`.
    pub fn validate(&self, def: &str, value: &Value) -> Result<(), String> {
        let validator = self
            .validators
            .get(def)
            .ok_or_else(|| format!("no schema definition {def:?}"))?;
        let errors: Vec<String> = validator
            .iter_errors(value)
            .map(|e| format!("{}: {e}", e.instance_path()))
            .collect();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("; "))
        }
    }
}

/// Reads a file holding one JSON object.
pub fn read_json(path: &Path) -> Result<Value, String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let value: Map<String, Value> =
        serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(Value::Object(value))
}
