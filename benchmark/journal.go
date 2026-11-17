package benchmark

import (
	"bufio"
	"bytes"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
)

type event struct {
	Kind       string    `json:"kind"`
	PlanSHA256 string    `json:"plan_sha256"`
	Start      *Envelope `json:"start,omitempty"`
	Receipt    *Receipt  `json:"receipt,omitempty"`
}
type recordBody struct {
	Seq        int    `json:"seq"`
	PrevSHA256 string `json:"prev_sha256"`
	Event      event  `json:"event"`
}
type record struct {
	Seq        int    `json:"seq"`
	PrevSHA256 string `json:"prev_sha256"`
	Event      event  `json:"event"`
	SHA256     string `json:"sha256"`
}
type journal struct {
	file     *os.File
	lock     *os.File
	plan     Plan
	planHash string
	requests map[string]Request
	inputs   map[string]string
	seq      int
	prev     string
	attempts map[string]Envelope
	receipts map[string]Receipt
	outcomes map[string]Receipt
	hook     func(string)
}

func syncDir(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return errors.New("directory_open_failed")
	}
	defer f.Close()
	if err = f.Sync(); err != nil {
		return errors.New("directory_sync_failed")
	}
	return nil
}
func boundedRead(path string, limit int64) ([]byte, error) {
	f, err := openRegular(path, os.O_RDONLY)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	b, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil {
		return nil, errors.New("file_read_failed")
	}
	if int64(len(b)) > limit {
		return nil, errors.New("file_size_limit")
	}
	return b, nil
}
func loadPlan(path string) (Plan, error) {
	var p Plan
	b, err := boundedRead(path, maxPlan)
	if err != nil {
		return p, err
	}
	if err = strict(b, &p); err != nil {
		return p, err
	}
	return p, p.Validate()
}
func openJournal(path string, p Plan, create bool, hook func(string)) (*journal, error) {
	lock, err := openRegular(path+".lock", os.O_CREATE|os.O_RDWR)
	if err != nil {
		return nil, err
	}
	if err = lockFile(lock); err != nil {
		lock.Close()
		return nil, err
	}
	j := &journal{lock: lock, plan: p, planHash: p.Digest(), requests: map[string]Request{}, inputs: map[string]string{}, prev: strings.Repeat("0", 64), attempts: map[string]Envelope{}, receipts: map[string]Receipt{}, outcomes: map[string]Receipt{}, hook: hook}
	for _, r := range p.Requests {
		j.requests[requestID(j.planHash, r)] = r
	}
	for _, i := range p.Items {
		j.inputs[i.ID] = i.InputBase64
	}
	good := false
	defer func() {
		if !good {
			j.close()
		}
	}()
	if err = syncDir(filepath.Dir(path)); err != nil {
		return nil, err
	}
	flags := os.O_RDWR | os.O_APPEND
	j.file, err = openRegular(path, flags)
	if err != nil && create {
		// O_EXCL is essential: even an uncooperative concurrent creator is never overwritten.
		j.file, err = openRegular(path, flags|os.O_CREATE|os.O_EXCL)
		if err == nil {
			if err = j.file.Sync(); err != nil {
				return nil, errors.New("journal_sync_failed")
			}
			if err = syncDir(filepath.Dir(path)); err != nil {
				return nil, err
			}
			if err = j.append(event{Kind: "header", PlanSHA256: j.planHash}); err != nil {
				return nil, err
			}
			good = true
			return j, nil
		}
	}
	if err != nil {
		return nil, err
	}
	st, err := j.file.Stat()
	if err != nil {
		return nil, errors.New("journal_stat_failed")
	}
	if st.Size() > maxJournal {
		return nil, errors.New("journal_size_limit")
	}
	reader := bufio.NewReaderSize(j.file, 64<<10)
	for {
		line, err := readLine(reader, 3<<20)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		var r record
		if err = strict(line, &r); err != nil {
			return nil, fail("corrupt_journal", err)
		}
		if err = j.apply(r); err != nil {
			return nil, err
		}
	}
	if j.seq == 0 {
		return nil, errors.New("empty_journal")
	}
	// A complete pre-crash write can survive without its prior acknowledgement.
	// Sync replayed bytes before relying on them for new work or final output.
	if err = j.file.Sync(); err != nil {
		return nil, errors.New("journal_sync_failed")
	}
	good = true
	return j, nil
}
func readLine(r *bufio.Reader, limit int) ([]byte, error) {
	var b []byte
	for {
		part, err := r.ReadSlice('\n')
		if len(b)+len(part) > limit {
			return nil, errors.New("journal_line_limit")
		}
		b = append(b, part...)
		if err == nil {
			if len(b) == 1 {
				return nil, errors.New("empty_journal_record")
			}
			return b[:len(b)-1], nil
		}
		if err == bufio.ErrBufferFull {
			continue
		}
		if err == io.EOF && len(b) > 0 {
			return nil, errors.New("truncated_journal_tail")
		}
		return nil, err
	}
}
func (j *journal) close() {
	if j.file != nil {
		j.file.Close()
	}
	if j.lock != nil {
		j.lock.Close()
	}
}
func (j *journal) apply(r record) error {
	if r.Seq != j.seq+1 || r.PrevSHA256 != j.prev || r.SHA256 != Hash(compact(recordBody{r.Seq, r.PrevSHA256, r.Event})) {
		return errors.New("journal_chain_mismatch")
	}
	e := r.Event
	if e.PlanSHA256 != j.planHash {
		return errors.New("journal_plan_mismatch")
	}
	switch e.Kind {
	case "header":
		if j.seq != 0 || e.Start != nil || e.Receipt != nil {
			return errors.New("invalid_header")
		}
	case "start":
		if j.seq == 0 || e.Start == nil || e.Receipt != nil {
			return errors.New("invalid_start")
		}
		s := *e.Start
		if !attemptPattern.MatchString(s.AttemptID) {
			return errors.New("invalid_attempt")
		}
		known, found := j.requests[s.RequestID]
		if !found || known != s.Request || s != j.envelope(s.Request, s.AttemptID) {
			return errors.New("start_identity_mismatch")
		}
		if _, ok := j.attempts[s.AttemptID]; ok {
			return errors.New("duplicate_attempt")
		}
		if _, ok := j.outcomes[s.RequestID]; ok {
			return errors.New("already_accepted_request")
		}
		j.attempts[s.AttemptID] = s
	case "receipt":
		if j.seq == 0 || e.Start != nil || e.Receipt == nil {
			return errors.New("invalid_receipt_event")
		}
		receipt := *e.Receipt
		s, ok := j.attempts[receipt.AttemptID]
		if !ok {
			return errors.New("unknown_attempt")
		}
		if _, ok = j.receipts[receipt.AttemptID]; ok {
			return errors.New("duplicate_receipt")
		}
		if err := validateReceipt(j.plan, s, receipt, false); err != nil {
			return err
		}
		if _, ok = j.outcomes[receipt.RequestID]; ok {
			return errors.New("conflicting_receipt")
		}
		j.receipts[receipt.AttemptID] = receipt
		if accepted(j.plan, receipt) {
			j.outcomes[receipt.RequestID] = receipt
		}
	default:
		return errors.New("unknown_event")
	}
	j.seq = r.Seq
	j.prev = r.SHA256
	return nil
}
func (j *journal) point(name string) {
	if j.hook != nil {
		j.hook(name)
	}
}
func (j *journal) append(e event) error {
	body := recordBody{j.seq + 1, j.prev, e}
	r := record{body.Seq, body.PrevSHA256, e, Hash(compact(body))}
	// Validate the transition before writing; apply only mutates in-memory state
	// after every rejecting check. Any subsequent I/O failure ends this run.
	if err := j.apply(r); err != nil {
		return err
	}
	b := append(compact(r), '\n')
	st, err := j.file.Stat()
	if err != nil {
		return errors.New("journal_stat_failed")
	}
	if st.Size()+int64(len(b)) > maxJournal {
		return errors.New("journal_size_limit")
	}
	n, err := j.file.Write(b)
	if err != nil || n != len(b) {
		return errors.New("journal_write_failed")
	}
	j.point(e.Kind + "_written")
	if err = j.file.Sync(); err != nil {
		return errors.New("journal_sync_failed")
	}
	j.point(e.Kind + "_synced")
	return nil
}
func (j *journal) digest() (string, error) {
	if _, err := j.file.Seek(0, io.SeekStart); err != nil {
		return "", errors.New("journal_seek_failed")
	}
	b, err := io.ReadAll(io.LimitReader(j.file, maxJournal+1))
	if err != nil || len(b) > maxJournal {
		return "", errors.New("journal_read_failed")
	}
	return Hash(b), nil
}
func publish(path string, b []byte) error {
	// Every caller validates before entering here. Link is a no-replace publish.
	parent := filepath.Dir(path)
	tmp, err := os.CreateTemp(parent, ".bench-final-*")
	if err != nil {
		return errors.New("export_temp_failed")
	}
	name := tmp.Name()
	defer os.Remove(name)
	defer tmp.Close()
	n, err := io.Copy(tmp, bytes.NewReader(b))
	if err != nil || n != int64(len(b)) {
		return errors.New("export_write_failed")
	}
	if err = tmp.Sync(); err != nil {
		return errors.New("export_sync_failed")
	}
	if err = os.Link(name, path); err != nil {
		return errors.New("export_exists_or_link_failed")
	}
	if err = syncDir(parent); err != nil {
		return err
	}
	return nil
}

func (j *journal) envelope(r Request, attempt string) Envelope {
	return Envelope{1, j.planHash, requestID(j.planHash, r), attempt, r, j.inputs[r.Item], j.plan.MaxTokens}
}
