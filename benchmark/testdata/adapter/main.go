// Deterministic synthetic adapter for process-level contract tests. This is
// not a model adapter or model validation.
package main

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"
)

type request struct {
	Item          string `json:"item"`
	Arm           string `json:"arm"`
	Draw          string `json:"draw"`
	Seed          int64  `json:"seed"`
	ModelSHA256   string `json:"model_sha256"`
	InputSHA256   string `json:"input_sha256"`
	ProfileSHA256 string `json:"profile_sha256"`
}
type envelope struct {
	Version     int     `json:"version"`
	PlanSHA256  string  `json:"plan_sha256"`
	RequestID   string  `json:"request_id"`
	AttemptID   string  `json:"attempt_id"`
	Request     request `json:"request"`
	InputBase64 string  `json:"input_base64"`
	MaxTokens   int     `json:"max_tokens"`
}
type receipt struct {
	Version          int     `json:"version"`
	PlanSHA256       string  `json:"plan_sha256"`
	RequestID        string  `json:"request_id"`
	AttemptID        string  `json:"attempt_id"`
	Request          request `json:"request"`
	CompletionBase64 string  `json:"completion_base64"`
	CompletionSHA256 string  `json:"completion_sha256"`
	Tokens           int     `json:"tokens"`
	Finish           string  `json:"finish"`
	Failure          string  `json:"failure"`
}

func main() {
	mode := flag.String("mode", "stop", "")
	counter := flag.String("count", "", "")
	flag.Parse()
	var e envelope
	if err := json.NewDecoder(io.LimitReader(os.Stdin, 2<<20)).Decode(&e); err != nil {
		os.Exit(2)
	}
	if *counter != "" {
		f, err := os.OpenFile(*counter, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
		if err != nil {
			os.Exit(3)
		}
		fmt.Fprintf(f, "%s %s %d\n", e.RequestID, e.AttemptID, os.Getpid())
		f.Sync()
		f.Close()
	}
	if *mode == "timeout" || (*mode == "timeout_on_p" && e.Request.Arm == "P") {
		time.Sleep(10 * time.Second)
	}
	if *mode == "exit" {
		os.Exit(9)
	}
	if *mode == "overflow" {
		fmt.Print(strings.Repeat("x", 2<<20))
		return
	}
	if *mode == "malformed" {
		fmt.Print("{\"broken\":")
		return
	}
	b := []byte("synthetic completion")
	if *mode == "secret" {
		b = []byte(`{"password":"synthetic-value"}`)
	}
	sum := sha256.Sum256(b)
	r := receipt{1, e.PlanSHA256, e.RequestID, e.AttemptID, e.Request, base64.StdEncoding.EncodeToString(b), hex.EncodeToString(sum[:]), 1, "stop", ""}
	switch *mode {
	case "truncation", "error", "resource_stop":
		r.Finish = *mode
		r.Failure = "synthetic_failure"
	case "wrong_digest":
		r.CompletionSHA256 = strings.Repeat("0", 64)
	case "foreign":
		r.RequestID = strings.Repeat("1", 64)
	case "wrong_model":
		r.Request.ModelSHA256 = strings.Repeat("2", 64)
	case "wrong_profile":
		r.Request.ProfileSHA256 = strings.Repeat("2", 64)
	case "wrong_input":
		r.Request.InputSHA256 = strings.Repeat("2", 64)
	case "wrong_plan":
		r.PlanSHA256 = strings.Repeat("2", 64)
	case "bad_tokens":
		r.Tokens = e.MaxTokens + 1
	case "runner_finish":
		r.Finish = "timeout"
		r.Failure = "deadline"
	case "retry_once":
		if e.Request.Arm == "B" {
			data, _ := os.ReadFile(*counter)
			if len(strings.Split(strings.TrimSpace(string(data)), "\n")) == 1 {
				r.Finish = "resource_stop"
				r.Failure = "synthetic_failure"
			}
		}
	}
	json.NewEncoder(os.Stdout).Encode(r)
}
