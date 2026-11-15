//! T2 heap bound (consultation section on T2): this binary's own
//! counting `GlobalAlloc` wrapping `System` tracks live heap, and
//! the one test asserts that peak heap growth during
//! `Checkpoint::open`, a sparse `build_replacement`, a sparse
//! `apply_replacement`, and — now that M4 has landed — a
//! full-form `build_replacement` over a 64 MiB synthetic
//! checkpoint each stay under 8 MiB. Fixtures are built outside
//! the measured windows (sparse files plus chunked writes), so
//! fixture construction never allocates a whole tensor either.
//! This test is alone in its binary: the counter is process-wide,
//! so no sibling test may allocate concurrently.

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::fs;
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use common::*;
use weightspace_tools::checkpoint::Checkpoint;
use weightspace_tools::ops::{ApplyReplacement, BuildReplacement};

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static BASELINE: AtomicUsize = AtomicUsize::new(0);
static PEAK_DELTA: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let current = CURRENT.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            let delta = current.saturating_sub(BASELINE.load(Ordering::Relaxed));
            PEAK_DELTA.fetch_max(delta, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) };
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

fn reset_peak() {
    BASELINE.store(CURRENT.load(Ordering::Relaxed), Ordering::Relaxed);
    PEAK_DELTA.store(0, Ordering::Relaxed);
}

fn peak_delta() -> usize {
    PEAK_DELTA.load(Ordering::Relaxed)
}

const LEN: u64 = 64 * 1024 * 1024;
const LIMIT: usize = 8 * 1024 * 1024;

/// Writes a one-tensor `U8[LEN]` checkpoint without allocating its
/// payload: all zero bytes as a sparse file for `fill == None`, or
/// chunk-written `fill` bytes, then applies `(offset, value)`
/// edits inside the tensor data.
fn write_big(path: &Path, fill: Option<u8>, edits: &[(u64, u8)]) -> u64 {
    let header = format!(r#"{{"big":{{"dtype":"U8","shape":[{LEN}],"data_offsets":[0,{LEN}]}}}}"#);
    let mut file = fs::File::create(path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header.as_bytes()).unwrap();
    let data_start = 8 + header.len() as u64;
    match fill {
        None => file.set_len(data_start + LEN).unwrap(),
        Some(byte) => {
            let chunk = vec![byte; 1024 * 1024];
            let mut left = LEN;
            while left > 0 {
                let n = left.min(chunk.len() as u64) as usize;
                file.write_all(&chunk[..n]).unwrap();
                left -= n as u64;
            }
        }
    }
    for (offset, value) in edits {
        file.seek(SeekFrom::Start(data_start + offset)).unwrap();
        file.write_all(&[*value]).unwrap();
    }
    data_start
}

fn make_model_from(weights: &Path, dir: &Path) -> PathBuf {
    let model = dir.join("model");
    fs::create_dir_all(&model).unwrap();
    fs::copy(weights, model.join("model.safetensors")).unwrap();
    model
}

#[test]
fn heap_peak_bounded() {
    let tmp = tempfile::tempdir().unwrap();
    let base = tmp.path().join("base.safetensors");
    let edited_sparse = tmp.path().join("edited-sparse.safetensors");
    let edited_full = tmp.path().join("edited-full.safetensors");
    write_big(&base, None, &[]);
    write_big(&edited_sparse, None, &[(17, 1), (LEN - 3, 2)]);
    write_big(&edited_full, Some(0xab), &[]);

    reset_peak();
    let checkpoint = Checkpoint::open(&base).unwrap();
    assert_eq!(checkpoint.tensor("big").unwrap().byte_len(), LEN);
    drop(checkpoint);
    assert!(
        peak_delta() < LIMIT,
        "Checkpoint::open peak heap growth {} >= {LIMIT}",
        peak_delta()
    );

    let sparse_patch = tmp.path().join("sparse.safetensors");
    reset_peak();
    build(&BuildReplacement {
        base: base.clone(),
        edited: edited_sparse,
        manifest: None,
        out: sparse_patch.clone(),
    })
    .unwrap();
    assert!(
        peak_delta() < LIMIT,
        "sparse build_replacement peak heap growth {} >= {LIMIT}",
        peak_delta()
    );

    let model = make_model_from(&base, tmp.path());
    reset_peak();
    apply(&ApplyReplacement {
        patch: sparse_patch,
        model,
        out: tmp.path().join("patched"),
        expect_tensor_sha256: BTreeMap::new(),
    })
    .unwrap();
    assert!(
        peak_delta() < LIMIT,
        "sparse apply_replacement peak heap growth {} >= {LIMIT}",
        peak_delta()
    );

    // The full-form path M4 zero-copies: views over the mapping,
    // streamed into the temp handle, gated streaming. Before M4
    // this leg held the tensor about three times in heap memory.
    reset_peak();
    build(&BuildReplacement {
        base,
        edited: edited_full,
        manifest: None,
        out: tmp.path().join("full.safetensors"),
    })
    .unwrap();
    assert!(
        peak_delta() < LIMIT,
        "full-form build_replacement peak heap growth {} >= {LIMIT}",
        peak_delta()
    );
}
