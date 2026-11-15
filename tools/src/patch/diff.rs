use std::sync::atomic::AtomicBool;

use crate::checkpoint::{Checkpoint, TensorMeta};
use crate::error::ToolError;
use crate::files::{CHUNK_BYTES, check_cancelled};

use super::file::check_meta;

/// Result of scanning one tensor pair.
pub struct TensorDiff {
    pub changed: u64,
    /// Present only when collection was requested and something changed:
    /// `(idx bytes, new bytes, old bytes)`.
    pub sparse: Option<(Vec<u8>, Vec<u8>, Vec<u8>)>,
}

/// Compares one tensor's stored bytes between base and edited in
/// bounded chunks over their snapshot mappings (never a whole
/// tensor at once). With `collect`, also materializes the sparse
/// arrays, which the replacement format itself requires as owned
/// tensors.
pub fn diff_tensor(
    name: &str,
    base: &Checkpoint,
    base_meta: &TensorMeta,
    edited: &Checkpoint,
    edited_meta: &TensorMeta,
    cancelled: &AtomicBool,
    collect: bool,
) -> Result<TensorDiff, ToolError> {
    check_meta(&base.path, name, base_meta)?;
    check_meta(&edited.path, name, edited_meta)?;
    let elem = usize::try_from(base_meta.elem_bytes()).expect("element size fits usize");
    let numel = base_meta.numel(&base.path, name)?;
    let chunk_elems = (CHUNK_BYTES / elem).max(1) as u64;
    let mut changed = 0u64;
    let mut arrays = collect.then(|| (Vec::new(), Vec::new(), Vec::new()));
    let mut base_buf = vec![0u8; CHUNK_BYTES];
    let mut edited_buf = vec![0u8; CHUNK_BYTES];
    let mut at = 0u64;
    while at < numel {
        check_cancelled(cancelled)?;
        let n = chunk_elems.min(numel - at);
        let bytes = usize::try_from(n).expect("chunk fits usize") * elem;
        base.read_tensor_at(
            base_meta,
            at * base_meta.elem_bytes(),
            &mut base_buf[..bytes],
        )?;
        edited.read_tensor_at(
            edited_meta,
            at * edited_meta.elem_bytes(),
            &mut edited_buf[..bytes],
        )?;
        for i in 0..usize::try_from(n).expect("chunk fits usize") {
            let lo = i * elem;
            let hi = lo + elem;
            if base_buf[lo..hi] != edited_buf[lo..hi] {
                let flat = at + i as u64;
                changed += 1;
                if let Some((idx, new, old)) = &mut arrays {
                    idx.extend_from_slice(&(flat as i64).to_le_bytes());
                    new.extend_from_slice(&edited_buf[lo..hi]);
                    old.extend_from_slice(&base_buf[lo..hi]);
                }
            }
        }
        at += n;
    }
    Ok(TensorDiff {
        changed,
        sparse: if changed == 0 { None } else { arrays },
    })
}
