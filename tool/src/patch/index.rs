use std::sync::atomic::AtomicBool;

use crate::checkpoint::{Checkpoint, TensorMeta};
use crate::error::ToolError;
use crate::files::check_cancelled;

use super::file::check_meta;

/// Largest number of sparse indices handled in one streaming block.
pub(crate) const INDEX_BLOCK: u64 = 4096;

/// Streams a `#idx` tensor in bounded blocks, validating that indices
/// are non-negative and strictly increasing across the whole stream.
/// The callback receives each block and the ordinal of its first
/// entry, and reads the matching `#new`/`#old` values through the
/// checkpoint's snapshot mapping.
pub fn for_each_index_block(
    checkpoint: &Checkpoint,
    tensor: &str,
    idx: &TensorMeta,
    cancelled: &AtomicBool,
    mut f: impl FnMut(&[i64], u64) -> Result<(), ToolError>,
) -> Result<(), ToolError> {
    check_meta(&checkpoint.path, tensor, idx)?;
    let count = idx.byte_len() / 8;
    let mut done = 0u64;
    let mut previous = -1i64;
    let mut buf = vec![0u8; usize::try_from(INDEX_BLOCK * 8).expect("block fits usize")];
    let mut block: Vec<i64> =
        Vec::with_capacity(usize::try_from(INDEX_BLOCK).expect("block fits usize"));
    while done < count {
        check_cancelled(cancelled)?;
        let n = INDEX_BLOCK.min(count - done);
        let bytes = usize::try_from(n * 8).expect("block fits usize");
        checkpoint.read_tensor_at(idx, done * 8, &mut buf[..bytes])?;
        block.clear();
        block.extend(
            buf[..bytes]
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().expect("8-byte chunk"))),
        );
        for &index in &block {
            if index <= previous {
                return Err(ToolError::malformed(
                    &checkpoint.path,
                    format!("{tensor}#idx: indices must be sorted, unique and non-negative"),
                ));
            }
            previous = index;
        }
        f(&block, done)?;
        done += n;
    }
    Ok(())
}
