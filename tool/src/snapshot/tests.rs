use super::*;

use std::fs;
use std::time::{Duration, Instant};

fn never_cancelled() -> AtomicBool {
    AtomicBool::new(false)
}

fn dir_entries(dir: &Path) -> Vec<PathBuf> {
    let mut entries: Vec<PathBuf> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    entries.sort();
    entries
}

/// A scratch tree with separate source and dedicated snapshot dirs.
fn scratch() -> (tempfile::TempDir, PathBuf, PathBuf) {
    let tmp = tempfile::tempdir().unwrap();
    let work = tmp.path().join("work");
    let snap = tmp.path().join("snap");
    fs::create_dir(&work).unwrap();
    fs::create_dir(&snap).unwrap();
    (tmp, work, snap)
}

fn expected_sha(bytes: &[u8]) -> Sha256Hex {
    Sha256Hex::from_digest(&Sha256::digest(bytes))
}

// S1: snapshot bytes and digest equal an independent SHA-256 of the source at copy time; later source overwrites are invisible.
#[test]
fn snapshot_is_private_copy_of_source() {
    let (_tmp, work, snap) = scratch();
    let source = work.join("source.bin");
    let bytes: Vec<u8> = (0..8192u32).map(|i| (i % 251) as u8).collect();
    fs::write(&source, &bytes).unwrap();

    let cancelled = never_cancelled();
    let (resolved, snapshot) = Snapshot::create(&source, &snap, &cancelled).unwrap();
    assert_eq!(resolved, fs::canonicalize(&source).unwrap());
    assert_eq!(snapshot.bytes(), bytes.as_slice());
    assert_eq!(snapshot.sha256(), &expected_sha(&bytes));

    // Overwriting the source afterwards changes nothing observed
    // through the snapshot; on Windows this also witnesses
    // WriteFile-to-mapping coherence of the private copy.
    fs::write(&source, vec![0xEE; bytes.len()]).unwrap();
    assert_eq!(snapshot.bytes(), bytes.as_slice());
    assert_eq!(snapshot.sha256(), &expected_sha(&bytes));
}

// S2: the dedicated snapshot dir is empty while the snapshot lives and after drop (Unix unlink-before-write form).
#[cfg(unix)]
#[test]
fn snapshot_has_no_name_while_alive() {
    let (_tmp, work, snap) = scratch();
    let source = work.join("source.bin");
    fs::write(&source, vec![b'A'; 64]).unwrap();

    let cancelled = never_cancelled();
    let (_resolved, snapshot) = Snapshot::create(&source, &snap, &cancelled).unwrap();
    assert_eq!(
        dir_entries(&snap),
        Vec::<PathBuf>::new(),
        "snapshot dir must be empty while the snapshot is alive"
    );
    assert_eq!(snapshot.bytes(), vec![b'A'; 64].as_slice());
    drop(snapshot);
    assert_eq!(
        dir_entries(&snap),
        Vec::<PathBuf>::new(),
        "snapshot dir must be empty after drop"
    );
}

// S3: the named snapshot cannot be opened or deleted by name while alive, and is gone after drop (Windows share-0 form).
#[cfg(windows)]
#[test]
fn snapshot_name_unopenable_while_alive() {
    // Not compiled or verified on the Linux lane that seals X3;
    // the future windows CI lane compiles and runs this.
    let (_tmp, work, snap) = scratch();
    let source = work.join("source.bin");
    fs::write(&source, vec![b'A'; 64]).unwrap();

    let cancelled = never_cancelled();
    let (_resolved, snapshot) = Snapshot::create(&source, &snap, &cancelled).unwrap();
    let entries = dir_entries(&snap);
    assert_eq!(entries.len(), 1, "exactly one snapshot entry: {entries:?}");
    let named = &entries[0];
    let name = named.file_name().unwrap().to_string_lossy();
    assert!(name.starts_with(".weightspace-snapshot-"), "{name}");

    // Positive control: attribute-only inspection still works.
    fs::symlink_metadata(named).expect("metadata open must succeed");
    // Every data or delete access fails with a sharing violation
    // (ERROR_SHARING_VIOLATION = 32; delete may report access
    // denied = 5 instead).
    let read_err = File::open(named).expect_err("read open must fail");
    assert_eq!(read_err.raw_os_error(), Some(32), "{read_err}");
    let write_err = fs::OpenOptions::new()
        .write(true)
        .open(named)
        .expect_err("write open must fail");
    assert_eq!(write_err.raw_os_error(), Some(32), "{write_err}");
    let remove_err = fs::remove_file(named).expect_err("delete must fail");
    assert!(
        matches!(remove_err.raw_os_error(), Some(32) | Some(5)),
        "{remove_err}"
    );
    assert_eq!(snapshot.bytes(), vec![b'A'; 64].as_slice());

    drop(snapshot);
    assert_eq!(
        dir_entries(&snap),
        Vec::<PathBuf>::new(),
        "snapshot dir must be empty after drop"
    );
}

// S4: deleting or renaming the source after create does not affect the snapshot.
#[test]
fn snapshot_survives_source_delete_and_rename() {
    let (_tmp, work, snap) = scratch();
    let cancelled = never_cancelled();

    let source = work.join("source.bin");
    fs::write(&source, b"keep-me").unwrap();
    let (_resolved, snapshot) = Snapshot::create(&source, &snap, &cancelled).unwrap();
    fs::remove_file(&source).unwrap();
    assert_eq!(snapshot.bytes(), b"keep-me");
    assert_eq!(snapshot.sha256(), &expected_sha(b"keep-me"));
    drop(snapshot);

    let source2 = work.join("source2.bin");
    let renamed = work.join("renamed.bin");
    fs::write(&source2, b"and-me").unwrap();
    let (_resolved, snapshot) = Snapshot::create(&source2, &snap, &cancelled).unwrap();
    fs::rename(&source2, &renamed).unwrap();
    assert_eq!(snapshot.bytes(), b"and-me");
    assert_eq!(fs::read(&renamed).unwrap(), b"and-me");
}

// S5: a length mismatch during the copy is refused as InputChanged, the snapshot dir is empty, and the source is unchanged.
#[test]
fn size_change_during_copy_refused() {
    let (_tmp, work, snap) = scratch();
    let source = work.join("source.bin");
    fs::write(&source, b"ours").unwrap();
    let before = fs::read(&source).unwrap();

    // Pass a recorded length the copy cannot match, directly
    // (the copy-and-compare seam), instead of racing a writer.
    let (_resolved, file) = open_input(&source).unwrap();
    let recorded = len_of(&file, &source).unwrap();
    let cancelled = never_cancelled();
    let err = Snapshot::from_source(&source, file, recorded + 5, &snap, &cancelled)
        .expect_err("size-changed source must be refused");
    assert!(matches!(err, ToolError::InputChanged { .. }), "{err}");
    assert_eq!(
        dir_entries(&snap),
        Vec::<PathBuf>::new(),
        "refused snapshot must leave nothing"
    );
    assert_eq!(fs::read(&source).unwrap(), before, "source mutated");
}

// S6: a cancelled snapshot returns Cancelled and leaves nothing in the snapshot dir.
#[test]
fn cancelled_snapshot_leaves_nothing() {
    let (_tmp, work, snap) = scratch();
    let source = work.join("source.bin");
    fs::write(&source, vec![7u8; 4096]).unwrap();

    let cancelled = AtomicBool::new(true);
    let err = Snapshot::create(&source, &snap, &cancelled)
        .expect_err("cancelled snapshot must be refused");
    assert!(matches!(err, ToolError::Cancelled), "{err}");
    assert_eq!(
        dir_entries(&snap),
        Vec::<PathBuf>::new(),
        "cancelled snapshot must leave nothing"
    );
    assert_eq!(fs::read(&source).unwrap(), vec![7u8; 4096]);
}

// S7: a non-regular input is refused as NotRegularFile and leaves nothing (Unix also covers a FIFO, opened non-blocking).
#[test]
fn non_regular_input_refused() {
    let (_tmp, work, snap) = scratch();
    let cancelled = never_cancelled();

    let err =
        Snapshot::create(&work, &snap, &cancelled).expect_err("directory input must be refused");
    if cfg!(windows) {
        // Opening a directory itself fails on Windows (std sets no
        // FILE_FLAG_BACKUP_SEMANTICS), so the refusal surfaces as
        // Io there; the windows lane pins which.
        assert!(
            matches!(err, ToolError::NotRegularFile { .. } | ToolError::Io { .. }),
            "{err}"
        );
    } else {
        assert!(matches!(err, ToolError::NotRegularFile { .. }), "{err}");
    }
    assert_eq!(dir_entries(&snap), Vec::<PathBuf>::new());

    #[cfg(unix)]
    {
        let fifo = work.join("input.fifo");
        let status = std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .expect("mkfifo must be available for the FIFO case");
        assert!(status.success(), "mkfifo failed: {status}");
        // Must return (the open is non-blocking), never hang.
        let err =
            Snapshot::create(&fifo, &snap, &cancelled).expect_err("FIFO input must be refused");
        assert!(matches!(err, ToolError::NotRegularFile { .. }), "{err}");
        assert_eq!(dir_entries(&snap), Vec::<PathBuf>::new());
    }
}

// S8: a process killed while holding a snapshot leaves no snapshot file behind.
#[test]
fn killed_process_leaves_no_snapshot() {
    // Child form: this same test binary re-run with the env vars
    // below creates a snapshot, signals readiness, and waits to
    // be killed. It must never fall through to the parent form.
    if std::env::var_os("WEIGHTSPACE_S8_CHILD").is_some() {
        let source = PathBuf::from(std::env::var_os("WEIGHTSPACE_S8_SOURCE").unwrap());
        let snap = PathBuf::from(std::env::var_os("WEIGHTSPACE_S8_SNAPDIR").unwrap());
        let ready = PathBuf::from(std::env::var_os("WEIGHTSPACE_S8_READY").unwrap());
        let cancelled = never_cancelled();
        let (_resolved, snapshot) = Snapshot::create(&source, &snap, &cancelled).unwrap();
        assert_eq!(snapshot.bytes(), b"child-held");
        fs::write(&ready, b"ready").unwrap();
        loop {
            std::thread::sleep(Duration::from_secs(60));
        }
    }

    let (_tmp, work, snap) = scratch();
    let source = work.join("source.bin");
    fs::write(&source, b"child-held").unwrap();
    let ready = work.join("ready");
    let exe = std::env::current_exe().unwrap();
    let mut child = std::process::Command::new(exe)
        .args([
            "snapshot::tests::killed_process_leaves_no_snapshot",
            "--exact",
            "--test-threads=1",
        ])
        .env("WEIGHTSPACE_S8_CHILD", "1")
        .env("WEIGHTSPACE_S8_SOURCE", &source)
        .env("WEIGHTSPACE_S8_SNAPDIR", &snap)
        .env("WEIGHTSPACE_S8_READY", &ready)
        .spawn()
        .expect("spawn S8 child");

    let deadline = Instant::now() + Duration::from_secs(30);
    while !ready.exists() {
        if let Some(status) = child.try_wait().unwrap() {
            panic!("S8 child exited before signalling readiness: {status}");
        }
        assert!(Instant::now() < deadline, "S8 child never became ready");
        std::thread::sleep(Duration::from_millis(25));
    }
    #[cfg(unix)]
    assert_eq!(
        dir_entries(&snap),
        Vec::<PathBuf>::new(),
        "Unix snapshot is unnamed even while the child holds it"
    );

    child.kill().expect("kill S8 child");
    child.wait().expect("wait for S8 child");
    // Poll: on Windows the delete-on-close cleanup lands when the
    // kernel closes the child's handles at termination.
    let deadline = Instant::now() + Duration::from_secs(5);
    while !dir_entries(&snap).is_empty() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(25));
    }
    assert_eq!(
        dir_entries(&snap),
        Vec::<PathBuf>::new(),
        "killed child must leave no snapshot"
    );
    assert_eq!(fs::read(&source).unwrap(), b"child-held");
}

// A foreign file in the snapshot dir is neither read as the snapshot nor deleted, alive or on drop.
#[test]
fn foreign_file_in_snapshot_dir_is_ignored_and_kept() {
    let (_tmp, work, snap) = scratch();
    let source = work.join("source.bin");
    let foreign = snap.join(".weightspace-snapshot-foreign.tmp");
    fs::write(&source, vec![b'A'; 1024]).unwrap();
    fs::write(&foreign, vec![b'B'; 1024]).unwrap();

    let cancelled = never_cancelled();
    let (_resolved, snapshot) = Snapshot::create(&source, &snap, &cancelled).unwrap();
    assert_eq!(snapshot.bytes(), vec![b'A'; 1024].as_slice());
    assert_eq!(snapshot.sha256(), &expected_sha(&vec![b'A'; 1024]));
    assert_eq!(fs::read(&foreign).unwrap(), vec![b'B'; 1024]);
    drop(snapshot);
    // Only the foreign file remains: drop deletes nothing by name
    // on Unix, and on Windows only the snapshot's own
    // delete-on-close name goes.
    assert_eq!(dir_entries(&snap), vec![foreign.clone()]);
    assert_eq!(fs::read(&foreign).unwrap(), vec![b'B'; 1024]);
    assert_eq!(fs::read(&source).unwrap(), vec![b'A'; 1024]);
}

// Gap B: the snapshot hash reports per-chunk byte progress through
// the same sink as the copy: a copy ramp to the total, then a hash
// ramp restarting at 0 with intermediates, ending at the total.
#[test]
fn snapshot_hash_reports_chunk_progress() {
    let (_tmp, work, snap) = scratch();
    let source = work.join("source.bin");
    let bytes: Vec<u8> = (0..(1024 * 1024 + 12345u32))
        .map(|i| (i % 251) as u8)
        .collect();
    fs::write(&source, &bytes).unwrap();
    let total = bytes.len() as u64;

    let cancelled = never_cancelled();
    let mut records: Vec<(u64, u64)> = Vec::new();
    let (_resolved, snapshot) =
        Snapshot::create_with_progress(&source, &snap, &cancelled, &mut |done, tot| {
            records.push((done, tot));
        })
        .unwrap();
    assert_eq!(snapshot.sha256(), &expected_sha(&bytes));
    assert!(
        records.iter().all(|&(_, tot)| tot == total),
        "every report must carry the file total: {records:?}"
    );
    let copy_end = records
        .iter()
        .position(|&(done, _)| done == total)
        .expect("copy ramp must reach the total");
    assert!(
        records[..copy_end]
            .iter()
            .all(|&(done, _)| done > 0 && done < total),
        "copy reports before completion are intermediates: {records:?}"
    );
    let hash = &records[copy_end + 1..];
    assert_eq!(
        hash.first(),
        Some(&(0, total)),
        "hash ramp must restart at 0: {records:?}"
    );
    assert_eq!(
        hash.len() - 1,
        (total as usize).div_ceil(CHUNK_BYTES),
        "one hash report per chunk: {records:?}"
    );
    assert!(
        hash.iter().any(|&(done, _)| done > 0 && done < total),
        "hash ramp must report intermediates: {records:?}"
    );
    assert!(
        hash.windows(2).all(|w| w[0].0 < w[1].0),
        "hash ramp must be strictly increasing: {records:?}"
    );
    assert_eq!(hash.last(), Some(&(total, total)));
}

// Gap B: cancellation arriving at the start of the snapshot-hash
// ramp aborts the hash, returns no snapshot, and leaves nothing.
#[test]
fn snapshot_hash_cancellation_leaves_nothing() {
    let (_tmp, work, snap) = scratch();
    let source = work.join("source.bin");
    let bytes = vec![7u8; 512 * 1024];
    fs::write(&source, &bytes).unwrap();
    let before = fs::read(&source).unwrap();

    let cancelled = never_cancelled();
    let copy_complete = std::cell::Cell::new(false);
    let flag = &cancelled;
    let seen = &copy_complete;
    let err = Snapshot::create_with_progress(&source, &snap, &cancelled, &mut |done, tot| {
        if done == tot {
            seen.set(true);
        } else if done == 0 && seen.get() {
            // The hash ramp's opening (0, total): cancel here so
            // the failure must come from the hash loop itself.
            flag.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    })
    .expect_err("cancelled snapshot hash must be refused");
    assert!(matches!(err, ToolError::Cancelled), "{err}");
    assert_eq!(
        dir_entries(&snap),
        Vec::<PathBuf>::new(),
        "cancelled snapshot hash must leave nothing"
    );
    assert_eq!(fs::read(&source).unwrap(), before, "source mutated");
}

// An empty input maps to an empty slice (memmap2 length-0), so no empty special case is needed; the official reader still refuses it.
#[test]
fn empty_input_maps_to_empty_slice() {
    let (_tmp, work, snap) = scratch();
    let source = work.join("empty.bin");
    fs::write(&source, []).unwrap();

    let cancelled = never_cancelled();
    let (_resolved, snapshot) = Snapshot::create(&source, &snap, &cancelled).unwrap();
    assert_eq!(snapshot.bytes(), &[] as &[u8]);
    assert_eq!(snapshot.sha256(), &expected_sha(&[]));
}
