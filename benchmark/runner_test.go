package benchmark

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

var cliBinary, adapterBinary string

func TestMain(m *testing.M) {
	if os.Getenv("BENCH_TEST_CLI") == "1" {
		os.Exit(m.Run())
	}
	if raceEnabled {
		os.Setenv("GORACE", "atexit_sleep_ms=0")
	}
	dir, err := os.MkdirTemp("", "bench-process-tests-")
	if err != nil {
		fmt.Fprintln(os.Stderr, "test directory failed")
		os.Exit(1)
	}
	cliBinary = filepath.Join(dir, "bench")
	adapterBinary = filepath.Join(dir, "adapter")
	for _, pair := range [][2]string{{cliBinary, "./cmd/bench"}, {adapterBinary, "./testdata/adapter"}} {
		buildArgs := []string{"build"}
		if raceEnabled && pair[1] == "./cmd/bench" {
			buildArgs = append(buildArgs, "-race")
			fmt.Println("building race-instrumented CLI")
		}
		buildArgs = append(buildArgs, "-o", pair[0], pair[1])
		cmd := exec.Command("go", buildArgs...)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err = cmd.Start(); err != nil {
			os.RemoveAll(dir)
			os.Exit(1)
		}
		fmt.Printf("build child PID %d\n", cmd.Process.Pid)
		if err = cmd.Wait(); err != nil {
			os.RemoveAll(dir)
			os.Exit(1)
		}
	}
	code := m.Run()
	os.RemoveAll(dir)
	os.Exit(code)
}

// This test-only entry point drives the same CLI with deterministic crash
// hooks in a fresh process. The production binary has no such controls.
func TestCLIProcess(t *testing.T) {
	if os.Getenv("BENCH_TEST_CLI") != "1" {
		return
	}
	args := os.Args
	for len(args) > 0 && args[0] != "--" {
		args = args[1:]
	}
	if len(args) == 0 {
		os.Exit(98)
	}
	hook := func(point string) {
		if point == os.Getenv("BENCH_TEST_FAULT") {
			fmt.Fprintln(os.Stderr, "test_crash_"+point)
			os.Exit(73)
		}
	}
	os.Exit(cli(context.Background(), args[1:], os.Stdout, os.Stderr, hook))
}

type fixture struct {
	dir, plan, journal, counter string
	p                           Plan
	args                        []string
}

func newFixture(t *testing.T, mode, policy string) *fixture {
	t.Helper()
	dir := t.TempDir()
	counter := filepath.Join(dir, "calls")
	args := []string{"--mode", mode, "--count", counter}
	adapter, err := os.ReadFile(adapterBinary)
	if err != nil {
		t.Fatal(err)
	}
	input := []byte("neutral synthetic input")
	p := Plan{Version: 1, Profile: Profile{Protocol, Hash([]byte("profile"))}, Models: Models{Hash([]byte("base")), Hash([]byte("patched"))}, AdapterSHA256: Hash(adapter), AdapterArgsSHA256: ArgsDigest(args), Items: []Item{{"item1", base64.StdEncoding.EncodeToString(input), Hash(input)}}, Draws: []Draw{{"draw1", 7}}, FailurePolicy: policy, TimeoutMS: 2000, MaxOutputBytes: 4096, MaxTokens: 8}
	for _, arm := range []string{"B", "P"} {
		model := p.Models.B
		if arm == "P" {
			model = p.Models.P
		}
		p.Requests = append(p.Requests, Request{"item1", arm, "draw1", 7, model, Hash(input), p.Profile.SHA256})
	}
	f := &fixture{dir, filepath.Join(dir, "plan.json"), filepath.Join(dir, "journal.jsonl"), counter, p, args}
	f.save(t)
	return f
}
func (f *fixture) save(t *testing.T) { t.Helper(); mustWrite(t, f.plan, compact(f.p)) }
func mustWrite(t *testing.T, path string, b []byte) {
	t.Helper()
	if err := os.WriteFile(path, b, 0600); err != nil {
		t.Fatal(err)
	}
}
func read(t *testing.T, path string) []byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return b
}
func (f *fixture) runArgs() []string {
	return append([]string{"run", "--plan", f.plan, "--journal", f.journal, "--adapter", adapterBinary, "--"}, f.args...)
}
func (f *fixture) op(name string) []string {
	return []string{name, "--plan", f.plan, "--journal", f.journal}
}
func (f *fixture) finalArgs(path string) []string { return append(f.op("final"), "--out", path) }
func execute(t *testing.T, cmd *exec.Cmd) (string, string, int) {
	t.Helper()
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Logf("CLI child PID %d", cmd.Process.Pid)
	err := cmd.Wait()
	code := 0
	if err != nil {
		if e, ok := err.(*exec.ExitError); ok {
			code = e.ExitCode()
		} else {
			t.Fatal(err)
		}
	}
	return stdout.String(), stderr.String(), code
}
func call(t *testing.T, args ...string) (string, string, int) {
	t.Helper()
	return execute(t, exec.Command(cliBinary, args...))
}
func good(t *testing.T, args ...string) string {
	t.Helper()
	out, err, code := call(t, args...)
	if code != 0 {
		t.Fatalf("exit %d: %s", code, err)
	}
	return out
}
func bad(t *testing.T, want string, args ...string) {
	t.Helper()
	out, err, code := call(t, args...)
	if code == 0 || !strings.Contains(err, want) || out != "" {
		t.Fatalf("want %q and no stdout; exit=%d stdout=%q stderr=%q", want, code, out, err)
	}
}
func noFile(t *testing.T, path string) {
	t.Helper()
	if _, err := os.Lstat(path); !os.IsNotExist(err) {
		t.Fatal("forbidden final file exists or stat failed")
	}
}
func counts(t *testing.T, f *fixture) int {
	t.Helper()
	b, err := os.ReadFile(f.counter)
	if os.IsNotExist(err) {
		return 0
	}
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(b)), "\n")
	for _, s := range lines {
		t.Log("adapter request/attempt/PID " + s)
	}
	return len(lines)
}
func crash(t *testing.T, f *fixture, point string) {
	t.Helper()
	args := append([]string{"-test.run=^TestCLIProcess$", "--"}, f.runArgs()...)
	cmd := exec.Command(os.Args[0], args...)
	cmd.Env = append(os.Environ(), "BENCH_TEST_CLI=1", "BENCH_TEST_FAULT="+point)
	out, err, code := execute(t, cmd)
	if code != 73 || out != "" || !strings.Contains(err, "test_crash_"+point) {
		t.Fatalf("fault not applied: %d %q %q", code, out, err)
	}
}
func getProgress(t *testing.T, f *fixture) progress {
	t.Helper()
	var p progress
	if err := json.Unmarshal([]byte(good(t, f.op("status")...)), &p); err != nil {
		t.Fatal(err)
	}
	if p.State != "INCOMPLETE" {
		t.Fatal("status must never emit final state")
	}
	return p
}

// A valid paired plan traverses the real production CLI and a separate Go
// executable, exports verified outcomes, and reuses accepted receipts on resume.
// Synthetic completions prove orchestration only, not model behavior.
func TestCLICompleteAndResume(t *testing.T) {
	f := newFixture(t, "stop", "record_all")
	good(t, "validate", "--plan", f.plan)
	if out := good(t, f.runArgs()...); strings.Contains(out, "COMPLETE\"") && !strings.Contains(out, "INCOMPLETE") {
		t.Fatal(out)
	}
	p := getProgress(t, f)
	if p.Accepted != 2 || p.Required != 2 || p.Attempts != 2 || p.UncertainAttempts != 0 {
		t.Fatalf("bad progress %+v", p)
	}
	before := read(t, f.journal)
	good(t, f.runArgs()...)
	if !bytes.Equal(before, read(t, f.journal)) || counts(t, f) != 2 {
		t.Fatal("resume repeated accepted work")
	}
	dest := filepath.Join(f.dir, "final.json")
	good(t, f.finalArgs(dest)...)
	var s sealed
	if err := strict(read(t, dest), &s); err != nil {
		t.Fatal(err)
	}
	body := sealBody{s.Version, s.State, s.PlanSHA256, s.JournalSHA256, s.Quality, s.Outcomes}
	if s.SHA256 != Hash(compact(body)) || s.JournalSHA256 != Hash(before) || s.PlanSHA256 != f.p.Digest() || s.Quality != "UNKNOWN" || s.State != "COMPLETE" || len(s.Outcomes) != 2 {
		t.Fatal("invalid sealed result")
	}
	for i, r := range s.Outcomes {
		if r.Request != f.p.Requests[i] || r.Finish != "stop" || r.RequestID != requestID(f.p.Digest(), r.Request) {
			t.Fatal("outcome binding")
		}
	}
	original := read(t, dest)
	bad(t, "export_exists_or_link_failed", f.finalArgs(dest)...)
	if !bytes.Equal(original, read(t, dest)) {
		t.Fatal("overwrote export")
	}
	counts(t, f)
}

// The final gate's known-violation control runs against an intentionally
// incomplete durable candidate, rejects for completeness (not setup), and
// creates neither final stdout nor a final file; the same candidate completes.
func TestCLIFinalGateControl(t *testing.T) {
	f := newFixture(t, "stop", "record_all")
	crash(t, f, "receipt_ack")
	p := getProgress(t, f)
	if p.Accepted != 1 || p.Required != 2 {
		t.Fatal(p)
	}
	before := read(t, f.journal)
	dest := filepath.Join(f.dir, "forbidden.json")
	bad(t, "incomplete_target", f.finalArgs(dest)...)
	noFile(t, dest)
	if !bytes.Equal(before, read(t, f.journal)) {
		t.Fatal("gate mutated journal")
	}
	good(t, f.runArgs()...)
	good(t, f.finalArgs(dest)...)
	if counts(t, f) != 2 {
		t.Fatal("unexpected retries")
	}
}

// Exact cross-product validation rejects equal-count identity substitutions,
// missing pairs, duplicate sets, changed seeds/digests and incompatible profiles;
// each isolated rejection is paired with a valid CLI validation control.
func TestCLIPlanRefusals(t *testing.T) {
	cases := []struct {
		name, want string
		edit       func(*Plan)
	}{
		{"missing", "incomplete_target", func(p *Plan) { p.Requests = p.Requests[:1] }},
		{"duplicate", "duplicate_request", func(p *Plan) { p.Requests[1] = p.Requests[0] }},
		{"foreign", "request_identity_mismatch", func(p *Plan) { p.Requests[1].Item = "foreign" }},
		{"arm", "invalid_arm", func(p *Plan) { p.Requests[1].Arm = "Q" }},
		{"seed", "request_identity_mismatch", func(p *Plan) { p.Requests[1].Seed++ }},
		{"model", "request_identity_mismatch", func(p *Plan) { p.Requests[1].ModelSHA256 = Hash([]byte("foreign")) }},
		{"profile", "request_identity_mismatch", func(p *Plan) { p.Requests[1].ProfileSHA256 = Hash([]byte("foreign")) }},
		{"input", "request_identity_mismatch", func(p *Plan) { p.Requests[1].InputSHA256 = Hash([]byte("foreign")) }},
		{"protocol", "incompatible_profile", func(p *Plan) { p.Profile.Protocol = "other" }},
		{"item_duplicate", "duplicate_item", func(p *Plan) { p.Items = append(p.Items, p.Items[0]) }},
		{"draw_duplicate", "duplicate_draw", func(p *Plan) { p.Draws = append(p.Draws, p.Draws[0]) }},
		{"budget", "invalid_budget", func(p *Plan) { p.TimeoutMS = 0 }},
		{"policy", "invalid_failure_policy", func(p *Plan) { p.FailurePolicy = "ignore" }},
		{"input_hash", "input_hash_mismatch", func(p *Plan) { p.Items[0].InputSHA256 = Hash(nil) }},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			f := newFixture(t, "stop", "record_all")
			good(t, "validate", "--plan", f.plan)
			c.edit(&f.p)
			f.save(t)
			bad(t, c.want, f.runArgs()...)
			noFile(t, f.journal)
		})
	}
}

// Required-field, duplicate-key, null, unknown-field and sensitive-value
// refusals occur at the CLI boundary without executing an adapter. Neutral
// content remains usable; synthetic secret markers are never real credentials.
func TestCLIJSONAndCustody(t *testing.T) {
	cases := []struct {
		name, want string
		edit       func([]byte) []byte
	}{
		{"duplicate", "duplicate_key", func(b []byte) []byte {
			return bytes.Replace(b, []byte(`"version":1`), []byte(`"version":1,"version":1`), 1)
		}},
		{"missing", "missing_field", func(b []byte) []byte { return bytes.Replace(b, []byte(`"version":1,`), nil, 1) }},
		{"unknown", "unknown_field", func(b []byte) []byte { return append([]byte(`{"extra":1,`), b[1:]...) }},
		{"null", "null_field", func(b []byte) []byte {
			return bytes.Replace(b, []byte(`"draws":[{"id":"draw1","seed":7}]`), []byte(`"draws":null`), 1)
		}},
		{"trailing", "trailing_json", func(b []byte) []byte { return append(b, []byte(` {}`)...) }},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			f := newFixture(t, "stop", "record_all")
			good(t, "validate", "--plan", f.plan)
			mustWrite(t, f.plan, c.edit(read(t, f.plan)))
			bad(t, c.want, f.runArgs()...)
			noFile(t, f.journal)
		})
	}
	for _, content := range []string{`{"api_key":"synthetic"}`, `{"password":"synthetic"}`, "sk-" + strings.Repeat("z", 24), "Bear" + "er " + "synthetic-marker"} {
		f := newFixture(t, "stop", "record_all")
		good(t, "validate", "--plan", f.plan)
		f.p.Items[0].InputBase64 = base64.StdEncoding.EncodeToString([]byte(content))
		f.p.Items[0].InputSHA256 = Hash([]byte(content))
		for i := range f.p.Requests {
			f.p.Requests[i].InputSHA256 = f.p.Items[0].InputSHA256
		}
		f.save(t)
		bad(t, "sensitive_", f.runArgs()...)
		noFile(t, f.journal)
	}
	f := newFixture(t, "stop", "record_all")
	f.p.Items[0].InputBase64 = base64.StdEncoding.EncodeToString([]byte(`{"password":""}`))
	f.p.Items[0].InputSHA256 = Hash([]byte(`{"password":""}`))
	for i := range f.p.Requests {
		f.p.Requests[i].InputSHA256 = f.p.Items[0].InputSHA256
	}
	f.save(t)
	good(t, f.runArgs()...)
}

// Corrupt or foreign adapter receipts cannot acquire acceptance. The durable
// start is retained as uncertain, and a matching positive adapter succeeds.
func TestCLIAdapterRefusals(t *testing.T) {
	for _, c := range []struct{ mode, want string }{{"wrong_digest", "completion_hash_mismatch"}, {"foreign", "receipt_identity_mismatch"}, {"wrong_model", "receipt_identity_mismatch"}, {"wrong_profile", "receipt_identity_mismatch"}, {"wrong_input", "receipt_identity_mismatch"}, {"wrong_plan", "receipt_identity_mismatch"}, {"bad_tokens", "invalid_tokens"}, {"runner_finish", "invalid_runner_failure"}, {"secret", "sensitive_field"}, {"malformed", "adapter_receipt_refused"}} {
		t.Run(c.mode, func(t *testing.T) {
			control := newFixture(t, "stop", "record_all")
			good(t, control.runArgs()...)
			f := newFixture(t, c.mode, "record_all")
			bad(t, c.want, f.runArgs()...)
			p := getProgress(t, f)
			if p.Accepted != 0 || p.UncertainAttempts != 1 {
				t.Fatal(p)
			}
			dest := filepath.Join(f.dir, "final")
			bad(t, "incomplete_target", f.finalArgs(dest)...)
			noFile(t, dest)
			counts(t, f)
		})
	}
	f := newFixture(t, "stop", "record_all")
	good(t, f.runArgs()...)
	original := read(t, f.journal)
	f.p.AdapterSHA256 = Hash([]byte("wrong"))
	f.save(t)
	bad(t, "adapter_hash_mismatch", f.runArgs()...)
	if !bytes.Equal(original, read(t, f.journal)) {
		t.Fatal("adapter mismatch mutated evidence")
	}
	f = newFixture(t, "stop", "record_all")
	f.args = append(f.args, string([]byte{0xff}))
	f.p.AdapterArgsSHA256 = ArgsDigest(f.args)
	f.save(t)
	bad(t, "invalid_argument_utf8", f.runArgs()...)
	noFile(t, f.journal)
	f = newFixture(t, "stop", "record_all")
	f.p.AdapterArgsSHA256 = Hash(nil)
	f.save(t)
	bad(t, "adapter_arguments_mismatch", f.runArgs()...)
	noFile(t, f.journal)
}

// Real timeout, process exit and output overflow as well as declared adapter
// failures stay in the record_all denominator and export with UNKNOWN quality.
// require_stop retains failures but cannot pass the final gate until a retry.
func TestCLIFailurePolicy(t *testing.T) {
	for _, mode := range []string{"stop", "truncation", "error", "resource_stop", "timeout", "exit", "overflow"} {
		t.Run(mode, func(t *testing.T) {
			f := newFixture(t, mode, "record_all")
			if mode == "timeout" {
				f.p.TimeoutMS = 150
				f.save(t)
			}
			good(t, f.runArgs()...)
			p := getProgress(t, f)
			if p.Accepted != 2 || p.UncertainAttempts != 0 {
				t.Fatal(p)
			}
			if mode != "stop" && p.FailedAttempts != 2 {
				t.Fatal(p)
			}
			dest := filepath.Join(f.dir, "final")
			good(t, f.finalArgs(dest)...)
			var s sealed
			if err := strict(read(t, dest), &s); err != nil {
				t.Fatal(err)
			}
			want := mode
			if mode == "exit" || mode == "overflow" {
				want = "adapter_error"
			}
			for _, r := range s.Outcomes {
				if r.Finish != want || s.Quality != "UNKNOWN" {
					t.Fatal("failure presented as success")
				}
			}
			counts(t, f)
		})
	}
	f := newFixture(t, "retry_once", "require_stop")
	bad(t, "failure_policy_incomplete", f.runArgs()...)
	p := getProgress(t, f)
	if p.Accepted != 0 || p.FailedAttempts != 1 || p.UncertainAttempts != 0 {
		t.Fatal(p)
	}
	dest := filepath.Join(f.dir, "final")
	bad(t, "incomplete_target", f.finalArgs(dest)...)
	noFile(t, dest)
	good(t, f.runArgs()...)
	p = getProgress(t, f)
	if p.Accepted != 2 || p.Attempts != 3 || p.FailedAttempts != 1 {
		t.Fatal(p)
	}
	good(t, f.finalArgs(dest)...)
	if counts(t, f) != 3 {
		t.Fatal("lost failure attempt")
	}
}

// A fresh CLI process dies at each isolated write/sync/ack boundary. A
// surviving complete line is replayed, accepted outcomes are never repeated,
// and unmatched starts stay visible when a new attempt completes the request.
// This is process-crash coverage, not physical power-loss qualification.
func TestCLICrashBoundaries(t *testing.T) {
	for _, point := range []string{"start_written", "start_synced", "before_receipt", "receipt_written", "receipt_synced", "receipt_ack"} {
		t.Run(point, func(t *testing.T) {
			f := newFixture(t, "stop", "record_all")
			crash(t, f, point)
			before := getProgress(t, f)
			afterReceipt := strings.HasPrefix(point, "receipt_")
			if afterReceipt {
				if before.Accepted != 1 {
					t.Fatal(before)
				}
			} else {
				if before.Accepted != 0 || before.UncertainAttempts != 1 {
					t.Fatal(before)
				}
			}
			good(t, f.runArgs()...)
			after := getProgress(t, f)
			if after.Accepted != 2 {
				t.Fatal(after)
			}
			expectedCalls := 2
			if point == "before_receipt" {
				expectedCalls = 3
			}
			if counts(t, f) != expectedCalls {
				t.Fatal("wrong attempt reuse")
			}
			if afterReceipt && after.Attempts != 2 {
				t.Fatal(after)
			}
			if !afterReceipt && (after.Attempts != 3 || after.UncertainAttempts != 1) {
				t.Fatal(after)
			}
			good(t, f.finalArgs(filepath.Join(f.dir, "final"))...)
		})
	}
}

func appendEvents(t *testing.T, path string, events ...event) {
	t.Helper()
	b := read(t, path)
	lines := bytes.Split(bytes.TrimSuffix(b, []byte{'\n'}), []byte{'\n'})
	var last record
	if err := strict(lines[len(lines)-1], &last); err != nil {
		t.Fatal(err)
	}
	for _, e := range events {
		body := recordBody{last.Seq + 1, last.SHA256, e}
		last = record{body.Seq, body.PrevSHA256, e, Hash(compact(body))}
		b = append(b, append(compact(last), '\n')...)
	}
	mustWrite(t, path, b)
}
func sampleReceipt(e Envelope, text string) Receipt {
	return Receipt{1, e.PlanSHA256, e.RequestID, e.AttemptID, e.Request, base64.StdEncoding.EncodeToString([]byte(text)), Hash([]byte(text)), 1, "stop", ""}
}

// Even syntactically valid and rehashed journal records must obey the receipt
// contract. Corruption, duplicates, conflicts, missing outcomes and changed
// plans refuse final output and preserve every original byte. Valid controls
// run beside each independently planted violation.
func TestCLIJournalRefusals(t *testing.T) {
	for _, c := range []struct {
		name, want string
		edit       func(*testing.T, *fixture)
	}{
		{"truncated", "truncated_journal_tail", func(t *testing.T, f *fixture) { b := read(t, f.journal); mustWrite(t, f.journal, b[:len(b)-1]) }},
		{"corrupt", "corrupt_journal", func(t *testing.T, f *fixture) {
			b := read(t, f.journal)
			mustWrite(t, f.journal, append(b, []byte("broken\n")...))
		}},
		{"chain", "journal_chain_mismatch", func(t *testing.T, f *fixture) {
			b := read(t, f.journal)
			b = bytes.Replace(b, []byte(`"seq":1`), []byte(`"seq":9`), 1)
			mustWrite(t, f.journal, b)
		}},
		{"missing", "incomplete_target", func(t *testing.T, f *fixture) {
			b := read(t, f.journal)
			lines := bytes.Split(bytes.TrimSuffix(b, []byte{'\n'}), []byte{'\n'})
			mustWrite(t, f.journal, append(bytes.Join(lines[:len(lines)-1], []byte{'\n'}), '\n'))
		}},
		{"hash", "completion_hash_mismatch", func(t *testing.T, f *fixture) {
			b := read(t, f.journal)
			lines := bytes.Split(bytes.TrimSuffix(b, []byte{'\n'}), []byte{'\n'})
			var r record
			json.Unmarshal(lines[len(lines)-1], &r)
			r.Event.Receipt.CompletionSHA256 = Hash([]byte("wrong"))
			r.SHA256 = Hash(compact(recordBody{r.Seq, r.PrevSHA256, r.Event}))
			lines[len(lines)-1] = compact(r)
			mustWrite(t, f.journal, append(bytes.Join(lines, []byte{'\n'}), '\n'))
		}},
		{"duplicate", "duplicate_receipt", func(t *testing.T, f *fixture) {
			b := read(t, f.journal)
			lines := bytes.Split(bytes.TrimSuffix(b, []byte{'\n'}), []byte{'\n'})
			var r record
			json.Unmarshal(lines[len(lines)-1], &r)
			appendEvents(t, f.journal, r.Event)
		}},
		{"foreign", "receipt_identity_mismatch", func(t *testing.T, f *fixture) {
			b := read(t, f.journal)
			lines := bytes.Split(bytes.TrimSuffix(b, []byte{'\n'}), []byte{'\n'})
			var r record
			json.Unmarshal(lines[len(lines)-1], &r)
			r.Event.Receipt.Request.InputSHA256 = Hash([]byte("foreign"))
			r.SHA256 = Hash(compact(recordBody{r.Seq, r.PrevSHA256, r.Event}))
			lines[len(lines)-1] = compact(r)
			mustWrite(t, f.journal, append(bytes.Join(lines, []byte{'\n'}), '\n'))
		}},
		{"sensitive_completion", "sensitive_field", func(t *testing.T, f *fixture) {
			b := read(t, f.journal)
			lines := bytes.Split(bytes.TrimSuffix(b, []byte{'\n'}), []byte{'\n'})
			var r record
			json.Unmarshal(lines[len(lines)-1], &r)
			text := []byte(`{"password":"synthetic-marker"}`)
			r.Event.Receipt.CompletionBase64 = base64.StdEncoding.EncodeToString(text)
			r.Event.Receipt.CompletionSHA256 = Hash(text)
			r.SHA256 = Hash(compact(recordBody{r.Seq, r.PrevSHA256, r.Event}))
			lines[len(lines)-1] = compact(r)
			mustWrite(t, f.journal, append(bytes.Join(lines, []byte{'\n'}), '\n'))
		}},
		{"plan", "journal_plan_mismatch", func(t *testing.T, f *fixture) { f.p.MaxTokens++; f.save(t) }},
		{"empty", "empty_journal", func(t *testing.T, f *fixture) { mustWrite(t, f.journal, nil) }},
	} {
		t.Run(c.name, func(t *testing.T) {
			f := newFixture(t, "stop", "record_all")
			good(t, f.runArgs()...)
			good(t, f.finalArgs(filepath.Join(f.dir, "control"))...)
			c.edit(t, f)
			before := read(t, f.journal)
			dest := filepath.Join(f.dir, "forbidden")
			bad(t, c.want, f.finalArgs(dest)...)
			noFile(t, dest)
			if !bytes.Equal(before, read(t, f.journal)) {
				t.Fatal("refusal changed journal")
			}
			counts(t, f)
		})
	}
	f := newFixture(t, "stop", "record_all")
	crash(t, f, "start_synced")
	b := read(t, f.journal)
	lines := bytes.Split(bytes.TrimSuffix(b, []byte{'\n'}), []byte{'\n'})
	var r record
	if err := strict(lines[1], &r); err != nil {
		t.Fatal(err)
	}
	first := *r.Event.Start
	second := first
	second.AttemptID = strings.Repeat("a", 32)
	one := sampleReceipt(first, "first")
	two := sampleReceipt(second, "conflicting")
	appendEvents(t, f.journal, event{Kind: "start", PlanSHA256: f.p.Digest(), Start: &second}, event{Kind: "receipt", PlanSHA256: f.p.Digest(), Receipt: &one})
	// Positive control: two uncertain attempts and one accepted receipt replay.
	if p := getProgress(t, f); p.Accepted != 1 || p.UncertainAttempts != 1 {
		t.Fatal(p)
	}
	appendEvents(t, f.journal, event{Kind: "receipt", PlanSHA256: f.p.Digest(), Receipt: &two})
	before := read(t, f.journal)
	dest := filepath.Join(f.dir, "forbidden")
	bad(t, "conflicting_receipt", f.finalArgs(dest)...)
	noFile(t, dest)
	bad(t, "conflicting_receipt", f.runArgs()...)
	if !bytes.Equal(before, read(t, f.journal)) {
		t.Fatal("conflict changed prior evidence")
	}
}

// Two independent CLI writers contend on one persistent lock. The loser
// refuses without writes, the winner's direct child is cancelled by SIGTERM,
// and the unmatched start remains incomplete. File-link aliases are refused.
func TestCLIConcurrencyAndSignal(t *testing.T) {
	f := newFixture(t, "timeout_on_p", "record_all")
	f.p.TimeoutMS = 2500
	f.save(t)
	crash(t, f, "receipt_ack")
	cmd := exec.Command(cliBinary, f.runArgs()...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Logf("concurrent CLI child PID %d", cmd.Process.Pid)
	done := false
	defer func() {
		if !done {
			cmd.Process.Kill()
			cmd.Wait()
		}
	}()
	deadline := time.Now().Add(2 * time.Second)
	for {
		if b, err := os.ReadFile(f.counter); err == nil && len(strings.Split(strings.TrimSpace(string(b)), "\n")) == 2 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("adapter did not start")
		}
		time.Sleep(5 * time.Millisecond)
	}
	before := read(t, f.journal)
	bad(t, "journal_busy", f.runArgs()...)
	bad(t, "journal_busy", f.op("status")...)
	if !bytes.Equal(before, read(t, f.journal)) {
		t.Fatal("second writer modified active journal")
	}
	if err := cmd.Process.Signal(syscall.SIGTERM); err != nil {
		t.Fatal(err)
	}
	err := cmd.Wait()
	done = true
	if err == nil || stdout.Len() != 0 || !strings.Contains(stderr.String(), "interrupted_incomplete") {
		t.Fatalf("signal: %v %q %q", err, stdout.String(), stderr.String())
	}
	p := getProgress(t, f)
	if p.Accepted != 1 || p.UncertainAttempts != 1 {
		t.Fatal(p)
	}
	if counts(t, f) != 2 {
		t.Fatal("unexpected execution")
	}
	// The direct PID was recorded by the adapter; do not kill any other process.
	lines := strings.Split(strings.TrimSpace(string(read(t, f.counter))), "\n")
	fields := strings.Fields(lines[len(lines)-1])
	pid, _ := strconv.Atoi(fields[2])
	if err := syscall.Kill(pid, 0); err != syscall.ESRCH {
		t.Fatalf("direct adapter PID still exists: %v", err)
	}
	dest := filepath.Join(f.dir, "final")
	bad(t, "incomplete_target", f.finalArgs(dest)...)
	noFile(t, dest)
	good(t, f.runArgs()...)
	good(t, f.finalArgs(dest)...)
	p = getProgress(t, f)
	if p.Accepted != 2 || p.Attempts != 3 || p.UncertainAttempts != 1 || counts(t, f) != 3 {
		t.Fatal("lost accepted state or retry evidence", p)
	}
}

// Symlink and hardlink aliases cannot bypass local ownership or cause an
// existing journal to be replaced; a normal single-link journal is usable.
func TestCLIFileAliases(t *testing.T) {
	for _, kind := range []string{"symlink", "hardlink"} {
		t.Run(kind, func(t *testing.T) {
			f := newFixture(t, "stop", "record_all")
			good(t, f.runArgs()...)
			before := read(t, f.journal)
			target := f.journal
			f.journal = filepath.Join(f.dir, "alias")
			var err error
			if kind == "symlink" {
				err = os.Symlink(target, f.journal)
			} else {
				err = os.Link(target, f.journal)
			}
			if err != nil {
				t.Fatal(err)
			}
			out, _, code := call(t, f.runArgs()...)
			if code == 0 || out != "" {
				t.Fatal("alias accepted")
			}
			if !bytes.Equal(before, read(t, target)) {
				t.Fatal("alias refusal changed original")
			}
			counts(t, f)
		})
	}
}

// A grid larger than one pair verifies every frozen item/draw/arm identity;
// substituting another valid grid identity at the same total count refuses.
func TestCLIPairedGrid(t *testing.T) {
	f := newFixture(t, "stop", "record_all")
	other := f.p.Items[0]
	other.ID = "item2"
	f.p.Items = append(f.p.Items, other)
	f.p.Draws = append(f.p.Draws, Draw{"draw2", 9})
	f.p.Requests = nil
	for _, i := range f.p.Items {
		for _, d := range f.p.Draws {
			for _, arm := range []string{"B", "P"} {
				model := f.p.Models.B
				if arm == "P" {
					model = f.p.Models.P
				}
				f.p.Requests = append(f.p.Requests, Request{i.ID, arm, d.ID, d.Seed, model, i.InputSHA256, f.p.Profile.SHA256})
			}
		}
	}
	f.save(t)
	good(t, f.runArgs()...)
	p := getProgress(t, f)
	if p.Required != 8 || p.Accepted != 8 {
		t.Fatal(p)
	}
	good(t, f.finalArgs(filepath.Join(f.dir, "control"))...)
	if counts(t, f) != 8 {
		t.Fatal("wrong grid")
	}
	f.p.Requests[7] = f.p.Requests[0]
	f.save(t)
	bad(t, "duplicate_request", "validate", "--plan", f.plan)
}
