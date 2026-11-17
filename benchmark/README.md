# Durable local paired runner

`bench` executes a frozen set of local adapter requests, retaining each attempt
and accepted outcome in a synced journal. Resume skips accepted requests. Final
export is refused until every exact item/draw/B/P identity is satisfied under the
frozen failure policy. Counts alone do not establish completeness.

This first slice supports native Linux and trusted local files/executables. It
makes no model-quality claim. Real model adapter qualification, automatic Git
shard claims/assembly, the complete MATH-500 specification and machine admission
remain unfinished. It does not change the Rust `weightspace` tool.

## Build and run

Go 1.26.8 or newer; standard library only. From the repository root:

```sh
mkdir -p .temp/benchmark
cd benchmark
go build -o ../.temp/benchmark/bench ./cmd/bench
```

With a frozen plan and a compatible local adapter, from the repository root:

```sh
.temp/benchmark/bench validate --plan /absolute/run/plan.json
.temp/benchmark/bench run --plan /absolute/run/plan.json --journal /absolute/run/requests.jsonl --adapter /absolute/adapter -- arg1 arg2
.temp/benchmark/bench status --plan /absolute/run/plan.json --journal /absolute/run/requests.jsonl
.temp/benchmark/bench final --plan /absolute/run/plan.json --journal /absolute/run/requests.jsonl --out /absolute/run/final.json
```

The adapter arguments are literal argv entries, never shell code from the plan.
The plan pins their compact JSON array SHA-256 as well as the executable bytes.
An empty argument list hashes `[]`. Repeating the same `run` command resumes;
changing budgets, identities, executable bytes or arguments refuses. The
executable must remain immutable throughout a run. Executables above 64 MiB are
outside this slice. No model files or datasets are downloaded.

See [PROTOCOL.md](PROTOCOL.md) for the exact v1 JSON fields, hashing rules,
adapter stdin/stdout, failure policies, concurrency and crash contract. The
export contains outcomes, completion byte hashes, journal hash and an export
seal. Quality is always `UNKNOWN`; there are no placeholder accuracy numbers.
`record_all` retains timeout, error, truncation and resource-stop as terminal
outcomes in the request denominator. `require_stop` retains failed attempts,
then stops incomplete; a later run can retry. Neither policy grades an answer.

`status` and `run` print only `INCOMPLETE` progress (required, accepted, attempts,
uncertain attempts, failed attempts), even when all receipts are available. Only
`final` may print `COMPLETE`, after verifying the full journal and writing a
new sealed export. Existing exports are never overwritten. Keep the journal:
it preserves retry and uncertainty evidence beyond the accepted outcomes.

## Files and ownership

Create a private run directory first. Journal paths and their parent directories
must remain under exclusive operator control. Keep the persistent sibling
`requests.jsonl.lock`; never delete, rename or replace it while the journal is
in use. All operations take a nonblocking local Linux file lock. This is not a
multi-machine lock or a network-filesystem protocol.

Records are append-only, chained and individually hashed, with file sync before
acknowledgement. A crash can leave an uncertain start that is retried with a new
attempt ID; adapter side effects may run twice. There is no globally exactly-once
execution guarantee. A previously accepted outcome is never accepted twice.
Empty journals, truncated tails, bad hashes and conflicts cause refusal with
original bytes preserved. There is no automatic repair command. Preserve the
original and obtain an independently audited recovery rather than deleting a
line. Final publication uses a synced temporary file and a no-replace hard link;
a crash may leave a complete `.bench-final-*` temporary file. An unsuccessful
directory sync may leave a complete final file with uncertain durability.

Adapters must not spawn descendants. The runner kills and waits for only its
direct child PID on timeout/cancellation, limits both output streams, and does
not inherit the caller's environment. These are I/O/process limits, not an OS
sandbox. The operator is responsible for selecting a trusted local executable.

Use nonsensitive synthetic or otherwise authorized data. Recognized token
patterns and nonempty secret-named JSON fields are refused, including in decoded
input/completion bytes. This is a bounded filter, not a general secret detector.
No raw stderr or rejected adapter output is exported.

## Contributor checks and tools

From `benchmark/`, run every selected check even when another fails:

```sh
go test -count=1 -v ./...
go test -race -count=1 ./...
go vet ./...
test -z "$(gofmt -l .)"
go test -count=1 -v -run 'TestCLI(CompleteAndResume|FinalGateControl|ConcurrencyAndSignal)$' .
```

`go build` produces the requested binary under ignored `.temp/benchmark/`.
`go test` builds the real CLI and a deterministic Go test adapter into disposable
test directories, launches them as separate processes, and removes those
directories when finished. Tests record child PIDs. The last command is the
focused real CLI smoke/control lane; it includes an intentionally incomplete
candidate that must refuse final output before the corresponding valid run
succeeds. Synthetic adapters are orchestration fixtures, not model validation.

Tests also cover exact paired grids, receipt identity/hash conflicts, retry,
resource-stop/error/timeout accounting, corrupt/truncated journals, concurrent
writers and signal cancellation. Test-only callbacks terminate fresh CLI
fixtures at write/sync/ack boundaries. Production exposes no fault flags or
environment variables. These checks exercise process crashes; storage hardware
power-loss behavior has not been qualified. CI runs Linux test/race/vet/format/
smoke as independent lanes with fail-fast disabled. Apple and Windows runtime
execution are not claimed.
