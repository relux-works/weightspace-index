//! Snapshot-backed official safetensors reader.
//!
//! [`Checkpoint::open`] first takes a private snapshot of the
//! resolved input file (see `snapshot`): bounded streaming copy
//! into a private file, memory-mapped read-only, with the SHA-256
//! computed over the mapped bytes. The
//! official [`SafeTensors::read_metadata`] parses that mapping and
//! its [`Metadata`]/`TensorInfo` descriptors and [`Dtype`] values
//! are the source of truth; no header byte is interpreted by hand.
//! Tensor bytes are then read, compared and hashed in bounded
//! chunks directly from the mapping, so no checkpoint is loaded
//! wholly into a heap buffer and later mutation of the caller's
//! original file cannot affect an opened checkpoint.
//!
//! Acceptance follows the official reader exactly, including where
//! it is stricter than a lenient reader would be: tensor ranges
//! must cover the data region contiguously (no gaps, no trailing
//! bytes), duplicate tensor-entry fields are refused by the
//! crate's own deserialization, and there is no legacy shape
//! alias. On top of that this tool still refuses dtypes whose
//! elements are not whole stored bytes, because its sparse
//! addressing is element-wise.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use safetensors::Dtype;
use safetensors::SafeTensors;
use safetensors::tensor::Metadata;
use sha2::{Digest, Sha256};

use crate::error::ToolError;
use crate::files::{CHUNK_BYTES, Sha256Hex, check_cancelled};
use crate::snapshot::Snapshot;

const LEN_PREFIX: u64 = 8;

/// One tensor's descriptor, copied from the official
/// `safetensors::tensor::TensorInfo`: dtype, shape, and byte
/// offsets relative to the start of the data region.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorMeta {
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub data_offsets: (u64, u64),
}

impl TensorMeta {
    /// Element count; a scalar (`shape == []`) has one element and a
    /// zero dimension makes the tensor empty. Checked arithmetic only.
    pub fn numel(&self, path: &Path, name: &str) -> Result<u64, ToolError> {
        self.shape.iter().try_fold(1u64, |acc, &d| {
            acc.checked_mul(d).ok_or_else(|| {
                ToolError::malformed(path, format!("{name}: shape element count overflows"))
            })
        })
    }

    /// Stored bytes per element. Callers have already refused dtypes
    /// whose elements are not whole bytes.
    pub fn elem_bytes(&self) -> u64 {
        (self.dtype.bitsize() / 8) as u64
    }

    /// Stored byte length (`end - start`). Descriptors copied from
    /// the official reader always have `start <= end`; a foreign
    /// caller-constructed descriptor with inverted offsets yields
    /// 0 here instead of panicking (its reads and diffs are then
    /// refused with typed errors at their entry points).
    pub fn byte_len(&self) -> u64 {
        self.data_offsets.1.saturating_sub(self.data_offsets.0)
    }

    /// Whether the byte offsets are inverted (`start > end`),
    /// which no descriptor from the official reader can be.
    pub(crate) fn offsets_inverted(&self) -> bool {
        self.data_offsets.0 > self.data_offsets.1
    }
}

/// One safetensors file parsed by the official reader over its own
/// private snapshot. Owns the snapshot mapping and the official
/// [`Metadata`]; borrowed `SafeTensors` views are never stored, so
/// there is no self-referential state.
#[derive(Debug)]
pub struct Checkpoint {
    /// The path as supplied by the caller.
    pub path: PathBuf,
    /// The canonicalized regular file the snapshot was copied from.
    resolved: PathBuf,
    snapshot: Snapshot,
    official: Metadata,
    header_len: u64,
    pub tensors: BTreeMap<String, TensorMeta>,
    pub metadata: BTreeMap<String, String>,
}

impl Checkpoint {
    /// Snapshots and opens a safetensors file (checkpoint or patch
    /// file; both take this same official reader path), refusing
    /// everything the official reader refuses plus sub-byte dtypes.
    /// The private snapshot copy is made in the system temporary
    /// directory; see [`Self::open_in`] to choose it.
    pub fn open(path: &Path) -> Result<Self, ToolError> {
        let cancelled = AtomicBool::new(false);
        Self::open_in(path, &std::env::temp_dir(), &cancelled)
    }

    /// Like [`Self::open`], but the private snapshot copy is made
    /// in `temp_dir` (which needs room for a copy of the input) and
    /// the copy and hash stop early when `cancelled` is set.
    pub fn open_in(
        path: &Path,
        temp_dir: &Path,
        cancelled: &AtomicBool,
    ) -> Result<Self, ToolError> {
        let (resolved, snapshot) = Snapshot::create(path, temp_dir, cancelled)?;
        // The official reader requires the whole file as one
        // contiguous slice — exactly what the snapshot mapping is.
        // An empty snapshot yields an empty slice here and the
        // official `HeaderTooSmall` refusal.
        let (header_len, official) = SafeTensors::read_metadata(snapshot.bytes())
            .map_err(|e| ToolError::malformed(path, e.to_string()))?;
        let header_len = u64::try_from(header_len)
            .map_err(|_| ToolError::malformed(path, "header too large"))?;

        let mut tensors = BTreeMap::new();
        for (name, info) in official.tensors() {
            let shape = info
                .shape
                .iter()
                .map(|&d| {
                    u64::try_from(d)
                        .map_err(|_| ToolError::malformed(path, format!("{name}: shape too large")))
                })
                .collect::<Result<Vec<u64>, _>>()?;
            let offsets = (
                u64::try_from(info.data_offsets.0).map_err(|_| {
                    ToolError::malformed(path, format!("{name}: data offset too large"))
                })?,
                u64::try_from(info.data_offsets.1).map_err(|_| {
                    ToolError::malformed(path, format!("{name}: data offset too large"))
                })?,
            );
            tensors.insert(
                name,
                TensorMeta {
                    dtype: info.dtype,
                    shape,
                    data_offsets: offsets,
                },
            );
        }
        // Whole-byte gate, in name order so the refusal is
        // deterministic: element-wise sparse addressing cannot name
        // an element that is not a whole number of stored bytes.
        for (name, meta) in &tensors {
            if meta.dtype.bitsize() % 8 != 0 {
                return Err(ToolError::UnsupportedDtype {
                    path: path.to_path_buf(),
                    tensor: name.clone(),
                    dtype: meta.dtype,
                });
            }
        }
        let metadata = official
            .metadata()
            .as_ref()
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();
        Ok(Self {
            path: path.to_path_buf(),
            resolved,
            snapshot,
            official,
            header_len,
            tensors,
            metadata,
        })
    }

    /// Absolute file offset where tensor data begins. The snapshot
    /// is byte-identical to the input at open time, so this is also
    /// the offset in the original file (and in a verified copy).
    pub fn data_start(&self) -> u64 {
        LEN_PREFIX + self.header_len
    }

    /// The canonicalized regular file the snapshot was copied from.
    pub fn resolved_path(&self) -> &Path {
        &self.resolved
    }

    /// SHA-256 of the snapshot bytes — the mapped bytes parsed
    /// and read through this checkpoint (see `snapshot` for their
    /// custody and trust boundary).
    pub fn snapshot_sha256(&self) -> &Sha256Hex {
        self.snapshot.sha256()
    }

    /// The official header descriptors this checkpoint was built
    /// from.
    pub fn official_metadata(&self) -> &Metadata {
        &self.official
    }

    pub fn tensor<'a>(&'a self, name: &str) -> Result<&'a TensorMeta, ToolError> {
        self.tensors
            .get(name)
            .ok_or_else(|| ToolError::malformed(&self.path, format!("tensor {name} not in file")))
    }

    /// The tensor's stored bytes as a view into the snapshot
    /// mapping. Callers consume it in bounded chunks; it is never
    /// copied whole by the comparison paths. Fallible rather than
    /// indexing, so even a descriptor that did not come from this
    /// checkpoint is a refusal, never a panic.
    fn tensor_slice(&self, meta: &TensorMeta) -> Result<&[u8], ToolError> {
        let out_of_range = || ToolError::malformed(&self.path, "tensor offset out of range");
        let absolute = |offset: u64| {
            self.data_start()
                .checked_add(offset)
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(out_of_range)
        };
        let start = absolute(meta.data_offsets.0)?;
        let end = absolute(meta.data_offsets.1)?;
        self.snapshot
            .bytes()
            .get(start..end)
            .ok_or_else(|| ToolError::malformed(&self.path, "tensor offset out of range"))
    }

    /// Copies into `buf` at `offset` bytes inside tensor `meta`,
    /// from the snapshot mapping.
    pub fn read_tensor_at(
        &self,
        meta: &TensorMeta,
        offset: u64,
        buf: &mut [u8],
    ) -> Result<(), ToolError> {
        let data = self.tensor_slice(meta)?;
        let start = usize::try_from(offset)
            .ok()
            .filter(|&s| s <= data.len())
            .ok_or_else(|| ToolError::malformed(&self.path, "tensor offset out of range"))?;
        let chunk = data
            .get(start..start + buf.len())
            .ok_or_else(|| ToolError::malformed(&self.path, "tensor offset out of range"))?;
        buf.copy_from_slice(chunk);
        Ok(())
    }

    /// Streams one tensor's raw stored bytes through SHA-256 in
    /// bounded chunks over the snapshot mapping.
    pub fn tensor_hash(
        &self,
        meta: &TensorMeta,
        cancelled: &AtomicBool,
    ) -> Result<Sha256Hex, ToolError> {
        let mut hasher = Sha256::new();
        for chunk in self.tensor_slice(meta)?.chunks(CHUNK_BYTES) {
            check_cancelled(cancelled)?;
            hasher.update(chunk);
        }
        Ok(Sha256Hex::from_digest(&hasher.finalize()))
    }
}
