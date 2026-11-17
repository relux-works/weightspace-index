package benchmark

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"io"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
	"unicode/utf8"
)

type boundedBuffer struct {
	b        bytes.Buffer
	limit    int
	overflow bool
}

func (w *boundedBuffer) Write(p []byte) (int, error) {
	remaining := w.limit - w.b.Len()
	if len(p) > remaining {
		if remaining > 0 {
			w.b.Write(p[:remaining])
		}
		w.overflow = true
		return len(p), nil
	}
	return w.b.Write(p)
}
func runnerFailure(e Envelope, finish, label string) Receipt {
	return Receipt{1, e.PlanSHA256, e.RequestID, e.AttemptID, e.Request, "", Hash(nil), 0, finish, label}
}
func invoke(ctx context.Context, p Plan, e Envelope, path string, args []string) (Receipt, error) {
	child, cancel := context.WithTimeout(ctx, time.Duration(p.TimeoutMS)*time.Millisecond)
	defer cancel()
	cmd := exec.CommandContext(child, path, args...)
	cmd.Env = []string{"PATH=/usr/bin:/bin", "LANG=C", "TZ=UTC"}
	cmd.Stdin = bytes.NewReader(append(compact(e), '\n'))
	stdout := &boundedBuffer{limit: p.MaxOutputBytes}
	stderr := &boundedBuffer{limit: p.MaxOutputBytes}
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	cmd.WaitDelay = 100 * time.Millisecond
	// CommandContext kills only cmd.Process, the direct child PID it started.
	err := cmd.Run()
	if ctx.Err() != nil {
		return Receipt{}, errors.New("interrupted_incomplete")
	}
	if child.Err() != nil {
		return runnerFailure(e, "timeout", "deadline"), nil
	}
	if err != nil {
		return runnerFailure(e, "adapter_error", "process_failed"), nil
	}
	if stdout.overflow || stderr.overflow {
		return runnerFailure(e, "adapter_error", "output_limit"), nil
	}
	var r Receipt
	if err = strict(stdout.b.Bytes(), &r); err != nil {
		return Receipt{}, fail("adapter_receipt_refused", err)
	}
	if err = validateReceipt(p, e, r, true); err != nil {
		return Receipt{}, err
	}
	return r, nil
}
func adapterCheck(p Plan, path string, args []string) error {
	if !filepath.IsAbs(path) {
		return errors.New("absolute_adapter_required")
	}
	for _, arg := range args {
		if !utf8.ValidString(arg) {
			return errors.New("invalid_argument_utf8")
		}
		if err := safeText([]byte(arg)); err != nil {
			return err
		}
		parts := strings.SplitN(strings.TrimLeft(arg, "-"), "=", 2)
		if len(parts) == 2 && secretKey.MatchString(parts[0]) && parts[1] != "" {
			return errors.New("sensitive_argument")
		}
	}
	if ArgsDigest(args) != p.AdapterArgsSHA256 {
		return errors.New("adapter_arguments_mismatch")
	}
	b, err := boundedRead(path, 64<<20)
	if err != nil {
		return err
	}
	if Hash(b) != p.AdapterSHA256 {
		return errors.New("adapter_hash_mismatch")
	}
	return nil
}

type progress struct {
	State             string `json:"state"`
	Required          int    `json:"required"`
	Accepted          int    `json:"accepted"`
	Attempts          int    `json:"attempts"`
	UncertainAttempts int    `json:"uncertain_attempts"`
	FailedAttempts    int    `json:"failed_attempts"`
}

func (j *journal) progress() progress {
	failures := 0
	for _, r := range j.receipts {
		if r.Finish != "stop" {
			failures++
		}
	}
	return progress{"INCOMPLETE", len(j.plan.Requests), len(j.outcomes), len(j.attempts), len(j.attempts) - len(j.receipts), failures}
}
func run(ctx context.Context, j *journal, path string, args []string) error {
	for _, req := range j.plan.Requests {
		id := requestID(j.planHash, req)
		if _, ok := j.outcomes[id]; ok {
			continue
		}
		if ctx.Err() != nil {
			return errors.New("interrupted_incomplete")
		}
		b := make([]byte, 16)
		if _, err := rand.Read(b); err != nil {
			return errors.New("attempt_random_failed")
		}
		e := j.envelope(req, hex.EncodeToString(b))
		if err := j.append(event{Kind: "start", PlanSHA256: e.PlanSHA256, Start: &e}); err != nil {
			return err
		}
		r, err := invoke(ctx, j.plan, e, path, args)
		if err != nil {
			return err
		}
		j.point("before_receipt")
		if ctx.Err() != nil {
			return errors.New("interrupted_incomplete")
		}
		if err = j.append(event{Kind: "receipt", PlanSHA256: e.PlanSHA256, Receipt: &r}); err != nil {
			return err
		}
		j.point("receipt_ack")
		if !accepted(j.plan, r) {
			return errors.New("failure_policy_incomplete")
		}
	}
	return nil
}

type sealBody struct {
	Version       int       `json:"version"`
	State         string    `json:"state"`
	PlanSHA256    string    `json:"plan_sha256"`
	JournalSHA256 string    `json:"journal_sha256"`
	Quality       string    `json:"quality"`
	Outcomes      []Receipt `json:"outcomes"`
}
type sealed struct {
	Version       int       `json:"version"`
	State         string    `json:"state"`
	PlanSHA256    string    `json:"plan_sha256"`
	JournalSHA256 string    `json:"journal_sha256"`
	Quality       string    `json:"quality"`
	Outcomes      []Receipt `json:"outcomes"`
	SHA256        string    `json:"sha256"`
}

func final(j *journal, path string) error {
	if len(j.outcomes) != len(j.plan.Requests) {
		return errors.New("incomplete_target")
	}
	outcomes := make([]Receipt, 0, len(j.plan.Requests))
	for _, req := range j.plan.Requests {
		r, ok := j.outcomes[requestID(j.planHash, req)]
		if !ok || r.Request != req || !accepted(j.plan, r) {
			return errors.New("incomplete_pairing")
		}
		outcomes = append(outcomes, r)
	}
	hash, err := j.digest()
	if err != nil {
		return err
	}
	body := sealBody{1, "COMPLETE", j.planHash, hash, "UNKNOWN", outcomes}
	result := sealed{body.Version, body.State, body.PlanSHA256, body.JournalSHA256, body.Quality, body.Outcomes, Hash(compact(body))}
	b := compact(result)
	var checked sealed
	if err = strict(b, &checked); err != nil {
		return err
	}
	return publish(path, append(b, '\n'))
}

// Main is shared by the production entry point and fresh-process integration
// tests. Only tests pass a hook; production has no fault flag or environment API.
func Main(ctx context.Context, args []string, out, errOut io.Writer) int {
	return cli(ctx, args, out, errOut, nil)
}
func cli(ctx context.Context, args []string, out, errOut io.Writer, hook func(string)) int {
	err := operate(ctx, args, out, hook)
	if err != nil {
		fmt.Fprintln(errOut, err)
		return 1
	}
	return 0
}
func operate(ctx context.Context, args []string, out io.Writer, hook func(string)) error {
	if len(args) == 0 {
		return errors.New("command_required")
	}
	op := args[0]
	if op != "validate" && op != "run" && op != "status" && op != "final" {
		return errors.New("unknown_command")
	}
	f := flag.NewFlagSet(op, flag.ContinueOnError)
	f.SetOutput(io.Discard)
	planPath := f.String("plan", "", "plan JSON file")
	var journalPath, adapterPath, outputPath *string
	if op != "validate" {
		journalPath = f.String("journal", "", "journal file")
	}
	if op == "run" {
		adapterPath = f.String("adapter", "", "absolute local executable")
	}
	if op == "final" {
		outputPath = f.String("out", "", "new final export")
	}
	if err := f.Parse(args[1:]); err != nil {
		return errors.New("invalid_flags")
	}
	if *planPath == "" || (journalPath != nil && *journalPath == "") || (adapterPath != nil && *adapterPath == "") || (outputPath != nil && *outputPath == "") {
		return errors.New("missing_flags")
	}
	if op != "run" && len(f.Args()) > 0 {
		return errors.New("unexpected_arguments")
	}
	p, err := loadPlan(*planPath)
	if err != nil {
		return err
	}
	if op == "validate" {
		_, err = fmt.Fprintf(out, "{\"state\":\"VALID\",\"plan_sha256\":\"%s\",\"required\":%d}\n", p.Digest(), len(p.Requests))
		return err
	}
	if op == "run" {
		if err = adapterCheck(p, *adapterPath, f.Args()); err != nil {
			return err
		}
	}
	j, err := openJournal(*journalPath, p, op == "run", hook)
	if err != nil {
		return err
	}
	defer j.close()
	if op == "final" {
		if err = final(j, *outputPath); err != nil {
			return err
		}
		_, err = io.WriteString(out, "{\"state\":\"COMPLETE\",\"quality\":\"UNKNOWN\"}\n")
		return err
	}
	if op == "run" {
		if err = run(ctx, j, *adapterPath, f.Args()); err != nil {
			return err
		}
	}
	_, err = out.Write(append(compact(j.progress()), '\n'))
	return err
}
