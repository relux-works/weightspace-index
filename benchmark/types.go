// Package benchmark implements the frozen local-request-v1 runner protocol.
package benchmark

import (
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"regexp"
	"strings"
	"unicode/utf8"
)

const Protocol = "local-request-v1"
const maxPlan = 8 << 20
const maxJournal = 256 << 20

var hexDigest = regexp.MustCompile(`^[0-9a-f]{64}$`)
var attemptPattern = regexp.MustCompile(`^[0-9a-f]{32}$`)
var identifier = regexp.MustCompile(`^[A-Za-z0-9_-]{1,64}$`)
var failurePattern = regexp.MustCompile(`^[a-z][a-z0-9_]{0,63}$`)
var secretKey = regexp.MustCompile(`(?i)(^|[_-])(token|secret|password|credential|authorization|cookie|session|api[_-]?key|private[_-]?key)([_-]|$)`)
var tokenPattern = regexp.MustCompile(`(?i)(bearer\s+\S+|sk-[a-z0-9_-]{16,}|gh[pousr]_[a-z0-9]{20,}|github_pat_[a-z0-9_]{20,}|eyJ[a-z0-9_-]{8,}\.[a-z0-9_-]{8,}\.[a-z0-9_-]{8,})`)

type Profile struct {
	Protocol string `json:"protocol"`
	SHA256   string `json:"sha256"`
}
type Models struct {
	B string `json:"B"`
	P string `json:"P"`
}
type Item struct {
	ID          string `json:"id"`
	InputBase64 string `json:"input_base64"`
	InputSHA256 string `json:"input_sha256"`
}
type Draw struct {
	ID   string `json:"id"`
	Seed int64  `json:"seed"`
}
type Request struct {
	Item          string `json:"item"`
	Arm           string `json:"arm"`
	Draw          string `json:"draw"`
	Seed          int64  `json:"seed"`
	ModelSHA256   string `json:"model_sha256"`
	InputSHA256   string `json:"input_sha256"`
	ProfileSHA256 string `json:"profile_sha256"`
}
type Plan struct {
	Version           int       `json:"version"`
	Profile           Profile   `json:"profile"`
	Models            Models    `json:"models"`
	AdapterSHA256     string    `json:"adapter_sha256"`
	AdapterArgsSHA256 string    `json:"adapter_args_sha256"`
	Items             []Item    `json:"items"`
	Draws             []Draw    `json:"draws"`
	Requests          []Request `json:"requests"`
	FailurePolicy     string    `json:"failure_policy"`
	TimeoutMS         int       `json:"timeout_ms"`
	MaxOutputBytes    int       `json:"max_output_bytes"`
	MaxTokens         int       `json:"max_tokens"`
}
type Envelope struct {
	Version     int     `json:"version"`
	PlanSHA256  string  `json:"plan_sha256"`
	RequestID   string  `json:"request_id"`
	AttemptID   string  `json:"attempt_id"`
	Request     Request `json:"request"`
	InputBase64 string  `json:"input_base64"`
	MaxTokens   int     `json:"max_tokens"`
}
type Receipt struct {
	Version          int     `json:"version"`
	PlanSHA256       string  `json:"plan_sha256"`
	RequestID        string  `json:"request_id"`
	AttemptID        string  `json:"attempt_id"`
	Request          Request `json:"request"`
	CompletionBase64 string  `json:"completion_base64"`
	CompletionSHA256 string  `json:"completion_sha256"`
	Tokens           int     `json:"tokens"`
	Finish           string  `json:"finish"`
	Failure          string  `json:"failure"`
}

func Hash(b []byte) string { s := sha256.Sum256(b); return hex.EncodeToString(s[:]) }
func compact(v any) []byte {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return b
}
func (p Plan) Digest() string { return Hash(compact(p)) }
func requestID(plan string, r Request) string {
	return Hash(compact(struct {
		Plan    string  `json:"plan_sha256"`
		Request Request `json:"request"`
	}{plan, r}))
}
func ArgsDigest(args []string) string {
	if args == nil {
		args = []string{}
	}
	return Hash(compact(args))
}

// strict decodes the documented shape: required fields, no duplicates/nulls,
// no unknown fields and exactly one value, including inside nested objects.
func strict(data []byte, dst any) error {
	if !utf8.Valid(data) {
		return errors.New("invalid_utf8")
	}
	d := json.NewDecoder(bytes.NewReader(data))
	d.UseNumber()
	v, err := readValue(d, 0)
	if err != nil {
		return err
	}
	if _, err = d.Token(); err != io.EOF {
		return errors.New("trailing_json")
	}
	if err = shape(v, reflect.TypeOf(dst).Elem()); err != nil {
		return err
	}
	d = json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	if err = d.Decode(dst); err != nil {
		return errors.New("invalid_json_shape")
	}
	return safeValue(v)
}
func readValue(d *json.Decoder, depth int) (any, error) {
	if depth > 32 {
		return nil, errors.New("json_depth")
	}
	t, err := d.Token()
	if err != nil {
		return nil, errors.New("invalid_json")
	}
	if delim, ok := t.(json.Delim); ok {
		switch delim {
		case '{':
			m := map[string]any{}
			for d.More() {
				key, err := d.Token()
				if err != nil {
					return nil, errors.New("invalid_json")
				}
				k, ok := key.(string)
				if !ok {
					return nil, errors.New("invalid_json")
				}
				if _, ok := m[k]; ok {
					return nil, errors.New("duplicate_key")
				}
				v, err := readValue(d, depth+1)
				if err != nil {
					return nil, err
				}
				m[k] = v
			}
			if t, err = d.Token(); err != nil || t != json.Delim('}') {
				return nil, errors.New("invalid_json")
			}
			return m, nil
		case '[':
			a := []any{}
			for d.More() {
				v, err := readValue(d, depth+1)
				if err != nil {
					return nil, err
				}
				a = append(a, v)
			}
			if t, err = d.Token(); err != nil || t != json.Delim(']') {
				return nil, errors.New("invalid_json")
			}
			return a, nil
		}
		return nil, errors.New("invalid_json")
	}
	return t, nil
}
func shape(v any, t reflect.Type) error {
	if t.Kind() == reflect.Pointer {
		if v == nil {
			return errors.New("null_field")
		}
		return shape(v, t.Elem())
	}
	if v == nil {
		return errors.New("null_field")
	}
	switch t.Kind() {
	case reflect.Struct:
		m, ok := v.(map[string]any)
		if !ok {
			return errors.New("object_required")
		}
		known := map[string]bool{}
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			tag := strings.Split(f.Tag.Get("json"), ",")
			k := tag[0]
			known[k] = true
			value, exists := m[k]
			optional := len(tag) > 1 && tag[1] == "omitempty"
			if !exists {
				if optional {
					continue
				}
				return errors.New("missing_field")
			}
			if err := shape(value, f.Type); err != nil {
				return err
			}
		}
		for k := range m {
			if !known[k] {
				return errors.New("unknown_field")
			}
		}
	case reflect.Slice:
		a, ok := v.([]any)
		if !ok {
			return errors.New("array_required")
		}
		for _, e := range a {
			if err := shape(e, t.Elem()); err != nil {
				return err
			}
		}
	}
	return nil
}
func safeValue(v any) error {
	switch x := v.(type) {
	case map[string]any:
		for k, v := range x {
			if secretKey.MatchString(k) && v != nil && v != "" {
				return errors.New("sensitive_field")
			}
			if err := safeValue(v); err != nil {
				return err
			}
		}
	case []any:
		for _, v := range x {
			if err := safeValue(v); err != nil {
				return err
			}
		}
	case string:
		if tokenPattern.MatchString(x) {
			return errors.New("sensitive_value")
		}
	}
	return nil
}
func safeText(b []byte) error {
	if tokenPattern.Match(b) {
		return errors.New("sensitive_value")
	}
	// Structured content is inspected without echoing any rejected value.
	if utf8.Valid(b) && json.Valid(b) {
		d := json.NewDecoder(bytes.NewReader(b))
		d.UseNumber()
		v, err := readValue(d, 0)
		if err != nil {
			return err
		}
		return safeValue(v)
	}
	return nil
}
func decodeBytes(s string) ([]byte, error) {
	b, err := base64.StdEncoding.Strict().DecodeString(s)
	if err != nil || base64.StdEncoding.EncodeToString(b) != s {
		return nil, errors.New("invalid_base64")
	}
	return b, nil
}
func (p Plan) Validate() error {
	if p.Version != 1 || p.Profile.Protocol != Protocol || !hexDigest.MatchString(p.Profile.SHA256) {
		return errors.New("incompatible_profile")
	}
	for _, s := range []string{p.Models.B, p.Models.P, p.AdapterSHA256, p.AdapterArgsSHA256} {
		if !hexDigest.MatchString(s) {
			return errors.New("invalid_digest")
		}
	}
	if p.FailurePolicy != "record_all" && p.FailurePolicy != "require_stop" {
		return errors.New("invalid_failure_policy")
	}
	if p.TimeoutMS < 1 || p.TimeoutMS > 3600000 || p.MaxOutputBytes < 256 || p.MaxOutputBytes > 1<<20 || p.MaxTokens < 1 || p.MaxTokens > 1000000 {
		return errors.New("invalid_budget")
	}
	if len(p.Items) == 0 || len(p.Draws) == 0 || len(p.Items) > 5000 || len(p.Draws) > 5000 || len(p.Items)*len(p.Draws)*2 > 10000 {
		return errors.New("invalid_target")
	}
	items := map[string]Item{}
	draws := map[string]Draw{}
	for _, i := range p.Items {
		if !identifier.MatchString(i.ID) {
			return errors.New("invalid_item")
		}
		if _, ok := items[i.ID]; ok {
			return errors.New("duplicate_item")
		}
		b, err := decodeBytes(i.InputBase64)
		if err != nil {
			return err
		}
		if len(b) > p.MaxOutputBytes || Hash(b) != i.InputSHA256 {
			return errors.New("input_hash_mismatch")
		}
		if err = safeText(b); err != nil {
			return err
		}
		items[i.ID] = i
	}
	for _, d := range p.Draws {
		if !identifier.MatchString(d.ID) {
			return errors.New("invalid_draw")
		}
		if _, ok := draws[d.ID]; ok {
			return errors.New("duplicate_draw")
		}
		draws[d.ID] = d
	}
	if len(p.Requests) != len(p.Items)*len(p.Draws)*2 {
		return errors.New("incomplete_target")
	}
	seen := map[string]bool{}
	for _, r := range p.Requests {
		i, ok := items[r.Item]
		d, okd := draws[r.Draw]
		m := p.Models.B
		if r.Arm == "P" {
			m = p.Models.P
		} else if r.Arm != "B" {
			return errors.New("invalid_arm")
		}
		if !ok || !okd || r.Seed != d.Seed || r.ModelSHA256 != m || r.InputSHA256 != i.InputSHA256 || r.ProfileSHA256 != p.Profile.SHA256 {
			return errors.New("request_identity_mismatch")
		}
		key := r.Item + "/" + r.Arm + "/" + r.Draw
		if seen[key] {
			return errors.New("duplicate_request")
		}
		seen[key] = true
	}
	return nil
}
func validateReceipt(p Plan, e Envelope, r Receipt, adapter bool) error {
	if r.Version != 1 || r.PlanSHA256 != e.PlanSHA256 || r.RequestID != e.RequestID || r.AttemptID != e.AttemptID || r.Request != e.Request {
		return errors.New("receipt_identity_mismatch")
	}
	b, err := decodeBytes(r.CompletionBase64)
	if err != nil {
		return err
	}
	if len(b) > p.MaxOutputBytes || Hash(b) != r.CompletionSHA256 {
		return errors.New("completion_hash_mismatch")
	}
	if err = safeText(b); err != nil {
		return err
	}
	if r.Tokens < 0 || r.Tokens > p.MaxTokens {
		return errors.New("invalid_tokens")
	}
	switch r.Finish {
	case "stop":
		if r.Failure != "" {
			return errors.New("invalid_failure")
		}
	case "truncation", "error", "resource_stop":
		if !failurePattern.MatchString(r.Failure) {
			return errors.New("invalid_failure")
		}
	case "timeout", "adapter_error":
		if adapter || len(b) != 0 || r.Tokens != 0 || !failurePattern.MatchString(r.Failure) {
			return errors.New("invalid_runner_failure")
		}
	default:
		return errors.New("invalid_finish")
	}
	return nil
}
func accepted(p Plan, r Receipt) bool   { return p.FailurePolicy == "record_all" || r.Finish == "stop" }
func fail(code string, err error) error { return fmt.Errorf("%s: %w", code, err) }
