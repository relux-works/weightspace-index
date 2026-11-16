use safetensors::Dtype;
use serde_json::json;

use crate::error::ToolError;

/// Full or sparse storage for one changed tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Representation {
    Full,
    Sparse,
}

/// One changed tensor considered by [`choose_representations`].
pub struct ChangedTensor<'a> {
    pub name: &'a str,
    pub dtype: Dtype,
    pub shape: &'a [u64],
    pub numel: u64,
    pub changed: u64,
}

/// Chooses a representation for a replacement file that would contain
/// only this one changed tensor (plus the standard
/// `__metadata__ {"format":"pt"}`), by the exact serialized sizes of
/// the two candidate files. Real files hold several changed tensors,
/// whose layout interacts, so `ops` uses [`choose_representations`]
/// for them; this single-tensor form stays for callers deciding one
/// tensor in isolation.
pub fn choose_representation(
    name: &str,
    dtype: Dtype,
    shape: &[u64],
    numel: u64,
    changed: u64,
) -> Result<Representation, ToolError> {
    let choices = choose_representations(&[ChangedTensor {
        name,
        dtype,
        shape,
        numel,
        changed,
    }])?;
    Ok(choices[0])
}

/// Chooses full or sparse storage for every changed tensor in one
/// replacement file, comparing actual serializer-bound file sizes.
///
/// The size of a given assignment is computed exactly as
/// `safetensors::serialize` (0.8.0 `prepare`) would lay the file out:
/// entries sorted by descending `Dtype` alignment order then name,
/// `data_offsets` cumulative from 0 in that order, header JSON with
/// those real start/end offsets (hence their real decimal digit
/// lengths) and the `__metadata__ {"format":"pt"}` entry this tool
/// always writes, the header padded with spaces up to a multiple of
/// 8, then `8 + padded header + tensor data`. The unit tests pin
/// this computation against `safetensors::serialize` output itself.
///
/// Decisions interact (one tensor's choice shifts later offsets,
/// digit lengths and the padding), so the choice is iterated from
/// all-full to a fixed point: each tensor in turn is set to sparse
/// only when the whole file is strictly smaller with it sparse,
/// holding the others at their current choices; ties store full.
/// The fixed point is guaranteed — flipping any single tensor from
/// the result would not make the serialized file smaller — but no
/// exhaustive search over all combinations is claimed.
pub fn choose_representations(
    tensors: &[ChangedTensor<'_>],
) -> Result<Vec<Representation>, ToolError> {
    if tensors.is_empty() {
        return Ok(Vec::new());
    }
    let templates = entry_templates(tensors)?;
    let mut choices = vec![Representation::Full; tensors.len()];
    let mut current = serialized_len(&templates, &choices)?;
    loop {
        let mut flipped = false;
        for i in 0..tensors.len() {
            let original = choices[i];
            choices[i] = match original {
                Representation::Full => Representation::Sparse,
                Representation::Sparse => Representation::Full,
            };
            let alternative = serialized_len(&templates, &choices)?;
            let (full_len, sparse_len) = match original {
                Representation::Full => (current, alternative),
                Representation::Sparse => (alternative, current),
            };
            let desired = if sparse_len < full_len {
                Representation::Sparse
            } else {
                Representation::Full
            };
            choices[i] = desired;
            if desired != original {
                current = if desired == Representation::Sparse {
                    sparse_len
                } else {
                    full_len
                };
                flipped = true;
            }
        }
        if !flipped {
            return Ok(choices);
        }
    }
}

/// One possible header/data entry of the replacement file: the full
/// entry of a tensor, or one leg of its sparse trio.
struct EntryTemplate {
    owner: usize,
    representation: Representation,
    dtype: Dtype,
    name: String,
    byte_len: u64,
    /// Rendered header entry length with `data_offsets` digits
    /// removed; the real length adds the decimal digits of the
    /// entry's actual start and end offsets.
    header_fixed_len: u64,
}

fn entry_templates(tensors: &[ChangedTensor<'_>]) -> Result<Vec<EntryTemplate>, ToolError> {
    let overflow = || ToolError::internal("stored byte comparison overflows");
    let mul = |a: u64, b: u64| a.checked_mul(b).ok_or_else(overflow);
    let mut templates = Vec::with_capacity(tensors.len() * 4);
    for (owner, tensor) in tensors.iter().enumerate() {
        if tensor.dtype.bitsize() % 8 != 0 {
            return Err(ToolError::internal(format!(
                "{}: dtype {:?} elements are not whole bytes",
                tensor.name, tensor.dtype
            )));
        }
        if tensor.changed == 0 || tensor.changed > tensor.numel {
            return Err(ToolError::internal(format!(
                "{}: {} changed elements out of {}",
                tensor.name, tensor.changed, tensor.numel
            )));
        }
        let elem = (tensor.dtype.bitsize() / 8) as u64;
        let value_bytes = mul(tensor.changed, elem)?;
        let sparse_shape = [tensor.changed];
        let specs = [
            (
                Representation::Full,
                tensor.name.to_owned(),
                tensor.dtype,
                tensor.shape,
                mul(tensor.numel, elem)?,
            ),
            (
                Representation::Sparse,
                format!("{}#idx", tensor.name),
                Dtype::I64,
                &sparse_shape[..],
                mul(tensor.changed, 8)?,
            ),
            (
                Representation::Sparse,
                format!("{}#new", tensor.name),
                tensor.dtype,
                &sparse_shape[..],
                value_bytes,
            ),
            (
                Representation::Sparse,
                format!("{}#old", tensor.name),
                tensor.dtype,
                &sparse_shape[..],
                value_bytes,
            ),
        ];
        for (representation, name, dtype, shape, byte_len) in specs {
            templates.push(EntryTemplate {
                header_fixed_len: header_entry_fixed_len(&name, dtype, shape),
                owner,
                representation,
                dtype,
                name,
                byte_len,
            });
        }
    }
    // The serializer sorts by descending dtype alignment, then name;
    // each entry's key is fixed, so sorting every possible entry once
    // gives each assignment's order as a subsequence.
    templates.sort_by(|a, b| b.dtype.cmp(&a.dtype).then_with(|| a.name.cmp(&b.name)));
    Ok(templates)
}

/// Exact serialized byte length of the replacement file for one
/// assignment, laid out as `safetensors::serialize` lays it out
/// (see [`choose_representations`]).
fn serialized_len(
    templates: &[EntryTemplate],
    choices: &[Representation],
) -> Result<u64, ToolError> {
    let overflow = || ToolError::internal("stored byte comparison overflows");
    let add = |a: u64, b: u64| a.checked_add(b).ok_or_else(overflow);
    // The "__metadata__":{"format":"pt"} entry always written first.
    let mut header_len = metadata_entry_len();
    let mut entries = 1u64;
    let mut offset = 0u64;
    for template in templates {
        if choices[template.owner] != template.representation {
            continue;
        }
        let end = add(offset, template.byte_len)?;
        let entry_len = add(
            template.header_fixed_len,
            add(decimal_digits(offset), decimal_digits(end))?,
        )?;
        header_len = add(header_len, entry_len)?;
        entries += 1;
        offset = end;
    }
    // Object braces plus one comma between adjacent entries.
    header_len = add(header_len, add(2, entries - 1)?)?;
    let padded = add(header_len, 7)? / 8 * 8;
    add(add(8, padded)?, offset)
}

fn decimal_digits(value: u64) -> u64 {
    let mut digits = 1;
    let mut rest = value;
    while rest >= 10 {
        rest /= 10;
        digits += 1;
    }
    digits
}

/// Header bytes of one `"name":{…}` entry as the serializer renders
/// it, minus the two `data_offsets` numbers (rendered here as the
/// one-digit placeholders `0,0`).
fn header_entry_fixed_len(name: &str, dtype: Dtype, shape: &[u64]) -> u64 {
    let rendered = json!({ name: {
        "dtype": dtype.to_string(),
        "shape": shape,
        "data_offsets": [0, 0],
    }});
    let text = serde_json::to_string(&rendered).expect("JSON values serialize");
    // Strip the outer braces of the one-entry object and the two
    // placeholder digits.
    (text.len() - 2 - 2) as u64
}

/// Header bytes of the `"__metadata__":{"format":"pt"}` entry.
fn metadata_entry_len() -> u64 {
    let rendered = json!({ "__metadata__": { "format": "pt" } });
    let text = serde_json::to_string(&rendered).expect("JSON values serialize");
    (text.len() - 2) as u64
}

/// Serialized length for a complete candidate list and assignment;
/// the spelling the oracle unit tests use.
#[cfg(test)]
fn serialized_file_len(
    tensors: &[ChangedTensor<'_>],
    choices: &[Representation],
) -> Result<u64, ToolError> {
    serialized_len(&entry_templates(tensors)?, choices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choose_thresholds() {
        // BF16, 16 elements: full data is 32 B; one sparse change is
        // 12 B of data but needs three header entries instead of one,
        // so the exact stored size still favors the full tensor.
        let shape = [16u64];
        assert_eq!(
            choose_representation("w", Dtype::BF16, &shape, 16, 1).unwrap(),
            Representation::Full
        );
        assert_eq!(
            choose_representation("w", Dtype::BF16, &shape, 16, 16).unwrap(),
            Representation::Full
        );
        // At 512 elements the data saving (1024 B vs 12 B) outweighs the
        // two extra header entries.
        assert_eq!(
            choose_representation("w", Dtype::BF16, &[512], 512, 1).unwrap(),
            Representation::Sparse
        );
        // A scalar can never win sparse: one element costs 12 B sparse
        // against 2 B full, before headers.
        assert_eq!(
            choose_representation("s", Dtype::BF16, &[], 1, 1).unwrap(),
            Representation::Full
        );
        // Nearby control just past the break-even flips to full.
        let big = [1000u64];
        let mut last_sparse = 0;
        for changed in 1..=1000 {
            if choose_representation("w", Dtype::F32, &big, 1000, changed).unwrap()
                == Representation::Sparse
            {
                last_sparse = changed;
            }
        }
        assert!((1..1000).contains(&last_sparse));
        assert_eq!(
            choose_representation("w", Dtype::F32, &big, 1000, last_sparse + 1).unwrap(),
            Representation::Full
        );
    }

    /// Serialized length of the replacement file for one assignment,
    /// measured on actual `safetensors::serialize` output (zero-filled
    /// data of the right lengths; only sizes matter here).
    fn oracle_len(tensors: &[ChangedTensor<'_>], choices: &[Representation]) -> usize {
        use safetensors::tensor::TensorView;
        use std::collections::HashMap;
        let mut specs: Vec<(String, Dtype, Vec<usize>, usize)> = Vec::new();
        for (tensor, choice) in tensors.iter().zip(choices) {
            let elem = tensor.dtype.bitsize() / 8;
            let usize_of = |v: u64| usize::try_from(v).unwrap();
            match choice {
                Representation::Full => specs.push((
                    tensor.name.to_owned(),
                    tensor.dtype,
                    tensor.shape.iter().map(|&d| usize_of(d)).collect(),
                    usize_of(tensor.numel) * elem,
                )),
                Representation::Sparse => {
                    let n = usize_of(tensor.changed);
                    specs.push((format!("{}#idx", tensor.name), Dtype::I64, vec![n], n * 8));
                    specs.push((
                        format!("{}#new", tensor.name),
                        tensor.dtype,
                        vec![n],
                        n * elem,
                    ));
                    specs.push((
                        format!("{}#old", tensor.name),
                        tensor.dtype,
                        vec![n],
                        n * elem,
                    ));
                }
            }
        }
        let datas: Vec<Vec<u8>> = specs.iter().map(|(_, _, _, len)| vec![0u8; *len]).collect();
        let views: Vec<(String, TensorView<'_>)> = specs
            .iter()
            .zip(&datas)
            .map(|((name, dtype, shape, _), data)| {
                (
                    name.clone(),
                    TensorView::new(*dtype, shape.clone(), data).unwrap(),
                )
            })
            .collect();
        let mut metadata: HashMap<String, String> = HashMap::new();
        metadata.insert("format".into(), "pt".into());
        safetensors::serialize(
            views.iter().map(|(name, view)| (name.as_str(), view)),
            Some(metadata),
        )
        .unwrap()
        .len()
    }

    fn oracle_choice(tensors: &[ChangedTensor<'_>], choices: &[Representation]) -> u64 {
        let ours = serialized_file_len(tensors, choices).unwrap();
        assert_eq!(
            ours as usize,
            oracle_len(tensors, choices),
            "serializer-bound size disagrees with safetensors::serialize for {choices:?}"
        );
        ours
    }

    #[test]
    fn reviewer_flip_bf16_w200_23_changes() {
        // The independent reviewer's verified flip: official
        // serialize gives full 504 bytes vs sparse 508 for BF16
        // w[200] with 23 changes; sparse must not be chosen.
        let tensors = [ChangedTensor {
            name: "w",
            dtype: Dtype::BF16,
            shape: &[200],
            numel: 200,
            changed: 23,
        }];
        assert_eq!(oracle_choice(&tensors, &[Representation::Full]), 504);
        assert_eq!(oracle_choice(&tensors, &[Representation::Sparse]), 508);
        assert_eq!(
            choose_representation("w", Dtype::BF16, &[200], 200, 23).unwrap(),
            Representation::Full
        );
        assert_eq!(
            choose_representations(&tensors).unwrap(),
            [Representation::Full]
        );
    }

    #[test]
    fn serialized_len_matches_official_serializer_sweep() {
        // Single-tensor boundary sweep: for every changed count the
        // computed size must equal the official serializer's output
        // size for both forms, and the choice must follow the oracle
        // (sparse only when strictly smaller).
        for (dtype, numel) in [
            (Dtype::BF16, 200u64),
            (Dtype::F32, 100),
            (Dtype::U8, 257),
            (Dtype::I64, 64),
        ] {
            let shape = [numel];
            for changed in 1..=numel {
                let tensors = [ChangedTensor {
                    name: "w",
                    dtype,
                    shape: &shape,
                    numel,
                    changed,
                }];
                let full = oracle_choice(&tensors, &[Representation::Full]);
                let sparse = oracle_choice(&tensors, &[Representation::Sparse]);
                let expected = if sparse < full {
                    Representation::Sparse
                } else {
                    Representation::Full
                };
                assert_eq!(
                    choose_representations(&tensors).unwrap(),
                    [expected],
                    "{dtype:?} numel={numel} changed={changed}: full={full} sparse={sparse}"
                );
            }
        }
    }

    #[test]
    fn multi_tensor_choice_matches_brute_force_oracle() {
        // Multi-tensor cases: every assignment's computed size must
        // equal the official serializer's, and the chosen assignment
        // must achieve the brute-force minimum size (ties toward
        // full are checked separately at the fixed point).
        let cases: Vec<Vec<ChangedTensor<'_>>> = vec![
            vec![
                ChangedTensor {
                    name: "a.weight",
                    dtype: Dtype::F64,
                    shape: &[37],
                    numel: 37,
                    changed: 5,
                },
                ChangedTensor {
                    name: "w",
                    dtype: Dtype::BF16,
                    shape: &[200],
                    numel: 200,
                    changed: 23,
                },
                ChangedTensor {
                    name: "z.bias",
                    dtype: Dtype::U8,
                    shape: &[1000],
                    numel: 1000,
                    changed: 42,
                },
            ],
            vec![
                ChangedTensor {
                    name: "m.mid",
                    dtype: Dtype::F32,
                    shape: &[8, 8],
                    numel: 64,
                    changed: 9,
                },
                ChangedTensor {
                    name: "a",
                    dtype: Dtype::I32,
                    shape: &[128],
                    numel: 128,
                    changed: 30,
                },
                ChangedTensor {
                    name: "zz",
                    dtype: Dtype::F16,
                    shape: &[333],
                    numel: 333,
                    changed: 17,
                },
                ChangedTensor {
                    name: "b",
                    dtype: Dtype::I64,
                    shape: &[50],
                    numel: 50,
                    changed: 4,
                },
            ],
        ];
        for tensors in &cases {
            let chosen = choose_representations(tensors).unwrap();
            let chosen_len = oracle_choice(tensors, &chosen);
            let mut best = u64::MAX;
            for mask in 0..(1u32 << tensors.len()) {
                let assignment: Vec<Representation> = (0..tensors.len())
                    .map(|i| {
                        if mask & (1 << i) == 0 {
                            Representation::Full
                        } else {
                            Representation::Sparse
                        }
                    })
                    .collect();
                best = best.min(oracle_choice(tensors, &assignment));
            }
            assert_eq!(chosen_len, best, "chosen {chosen:?} is not minimal");
            // Fixed point with ties toward full: no single flip is
            // strictly smaller, and no sparse entry is a tie.
            for i in 0..tensors.len() {
                let mut flipped = chosen.clone();
                flipped[i] = match flipped[i] {
                    Representation::Full => Representation::Sparse,
                    Representation::Sparse => Representation::Full,
                };
                let flipped_len = oracle_choice(tensors, &flipped);
                match chosen[i] {
                    Representation::Full => assert!(flipped_len >= chosen_len),
                    Representation::Sparse => assert!(flipped_len > chosen_len),
                }
            }
        }
    }
}
