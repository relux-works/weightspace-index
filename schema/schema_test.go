package schema_test

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"reflect"
	"regexp"
	"strings"
	"testing"
)

// This bounded test helper interprets only the keywords present in this schema.
// It performs no network, file resolution, payload parsing or application.
type schemaCheck struct{ root map[string]any }

// JSON Schema uses ECMA-style regexes, where $ may precede a final newline.
// The schema appends a strict-end assertion. Translate only that exact suffix
// to Go's absolute end anchor; all other syntax is the shared RE2 subset.
func compilePattern(pattern string) (*regexp.Regexp, error) {
	const strictEnd = `$(?![\s\S])`
	if strings.HasSuffix(pattern, strictEnd) {
		pattern = strings.TrimSuffix(pattern, strictEnd) + `\z`
	}
	return regexp.Compile(pattern)
}

func readJSONFile(path string) (map[string]any, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var v map[string]any
	if err := json.Unmarshal(b, &v); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return v, nil
}

func readJSON(t *testing.T, path string) map[string]any {
	t.Helper()
	v, err := readJSONFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func clone(v any) any {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	var out any
	if err := json.Unmarshal(b, &out); err != nil {
		panic(err)
	}
	return out
}

func loadSchema() (schemaCheck, error) {
	root, err := readJSONFile("records.schema.json")
	if err != nil {
		return schemaCheck{}, err
	}
	if err := checkKeywords(root); err != nil {
		return schemaCheck{}, err
	}
	return schemaCheck{root}, nil
}

func newCheck(t *testing.T) schemaCheck {
	t.Helper()
	c, err := loadSchema()
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func checkKeywords(s map[string]any) error {
	allowed := strings.Fields("$schema $defs $ref title description type const enum oneOf properties required additionalProperties items minItems maxItems minimum pattern minLength")
	for k, v := range s {
		known := false
		for _, a := range allowed {
			known = known || k == a
		}
		if !known {
			return fmt.Errorf("unsupported schema keyword %q", k)
		}
		switch k {
		case "$defs", "properties":
			for _, child := range v.(map[string]any) {
				if err := checkKeywords(child.(map[string]any)); err != nil {
					return err
				}
			}
		case "items":
			if err := checkKeywords(v.(map[string]any)); err != nil {
				return err
			}
		case "oneOf":
			for _, child := range v.([]any) {
				if err := checkKeywords(child.(map[string]any)); err != nil {
					return err
				}
			}
		case "pattern":
			if _, err := compilePattern(v.(string)); err != nil {
				return err
			}
		}
	}
	return nil
}

func (c schemaCheck) validate(s map[string]any, v any) error {
	if r, ok := s["$ref"].(string); ok {
		if !strings.HasPrefix(r, "#/$defs/") {
			return fmt.Errorf("unsupported schema reference %s", r)
		}
		name := strings.TrimPrefix(r, "#/$defs/")
		def, ok := c.root["$defs"].(map[string]any)[name].(map[string]any)
		if !ok {
			return fmt.Errorf("unresolved schema reference %s", r)
		}
		if err := c.validate(def, v); err != nil {
			return err
		}
	}
	if want, ok := s["const"]; ok && !reflect.DeepEqual(want, v) {
		return fmt.Errorf("const: got %v, want %v", v, want)
	}
	if choices, ok := s["enum"].([]any); ok {
		found := false
		for _, choice := range choices {
			found = found || reflect.DeepEqual(choice, v)
		}
		if !found {
			return fmt.Errorf("value outside enum: %v", v)
		}
	}
	if choices, ok := s["oneOf"].([]any); ok {
		count := 0
		var failures []string
		for _, choice := range choices {
			if err := c.validate(choice.(map[string]any), v); err == nil {
				count++
			} else {
				failures = append(failures, err.Error())
			}
		}
		if count != 1 {
			return fmt.Errorf("oneOf: matched %d; %s", count, strings.Join(failures, "; "))
		}
	}
	if typ, ok := s["type"].(string); ok {
		valid := false
		switch typ {
		case "object":
			_, valid = v.(map[string]any)
		case "array":
			_, valid = v.([]any)
		case "string":
			_, valid = v.(string)
		case "number", "integer":
			n, ok := v.(float64)
			valid = ok && (typ == "number" || math.Trunc(n) == n)
		default:
			return fmt.Errorf("unsupported type %s", typ)
		}
		if !valid {
			return fmt.Errorf("expected %s, got %T", typ, v)
		}
	}
	if m, ok := v.(map[string]any); ok {
		props, _ := s["properties"].(map[string]any)
		if req, ok := s["required"].([]any); ok {
			for _, field := range req {
				if _, exists := m[field.(string)]; !exists {
					return fmt.Errorf("required field missing: %s", field)
				}
			}
		}
		for k, val := range m {
			if child, exists := props[k]; exists {
				if err := c.validate(child.(map[string]any), val); err != nil {
					return fmt.Errorf("%s: %w", k, err)
				}
			} else if extra, ok := s["additionalProperties"].(bool); ok && !extra {
				return fmt.Errorf("unexpected field: %s", k)
			}
		}
	}
	if a, ok := v.([]any); ok {
		if n, ok := s["minItems"].(float64); ok && float64(len(a)) < n {
			return fmt.Errorf("too few items")
		}
		if n, ok := s["maxItems"].(float64); ok && float64(len(a)) > n {
			return fmt.Errorf("too many items")
		}
		if item, ok := s["items"].(map[string]any); ok {
			for i, val := range a {
				if err := c.validate(item, val); err != nil {
					return fmt.Errorf("[%d]: %w", i, err)
				}
			}
		}
	}
	if str, ok := v.(string); ok {
		if n, ok := s["minLength"].(float64); ok && float64(len([]rune(str))) < n {
			return fmt.Errorf("string too short")
		}
		if p, ok := s["pattern"].(string); ok {
			re, err := compilePattern(p)
			if err != nil || !re.MatchString(str) {
				return fmt.Errorf("pattern mismatch: %q", str)
			}
		}
	}
	if n, ok := v.(float64); ok {
		if min, ok := s["minimum"].(float64); ok && n < min {
			return fmt.Errorf("number below minimum")
		}
	}
	return nil
}

// as validates v against one named definition.
func (c schemaCheck) as(name string, v any) error {
	if _, ok := c.root["$defs"].(map[string]any)[name]; !ok {
		return fmt.Errorf("no definition %q", name)
	}
	return c.validate(map[string]any{"$ref": "#/$defs/" + name}, v)
}

func (c schemaCheck) expect(t *testing.T, name string, v any, want bool) {
	t.Helper()
	err := c.as(name, v)
	if (err == nil) != want {
		t.Fatalf("%s: accepted=%v, want=%v: %v", name, err == nil, want, err)
	}
}

const example = "../examples/models/example-org/Example-Model-1B/"

func TestShippedGeneratedFiles(t *testing.T) {
	c := newCheck(t)
	c.expect(t, "bases", readJSON(t, "../bases.json"), true)
}

func TestBadRecordControls(t *testing.T) {
	c := newCheck(t)
	file := func(r map[string]any, i int) map[string]any { return r["files"].([]any)[i].(map[string]any) }
	profile := func(r map[string]any) map[string]any { return r["profiles"].([]any)[0].(map[string]any) }
	cases := []struct {
		name, def, path string
		edit            func(map[string]any)
	}{
		{"missing-digest", "version", "better-math/v1/version.json", func(r map[string]any) { delete(file(r, 0), "sha256") }},
		{"word-for-digest", "version", "better-math/v1/version.json", func(r map[string]any) { file(r, 0)["sha256"] = "advertised" }},
		{"uppercase-digest", "version", "better-math/v1/version.json", func(r map[string]any) { file(r, 0)["sha256"] = strings.Repeat("A", 64) }},
		{"missing-size", "version", "better-math/v1/version.json", func(r map[string]any) { delete(file(r, 0), "size_bytes") }},
		{"zero-size", "version", "better-math/v1/version.json", func(r map[string]any) { file(r, 0)["size_bytes"] = float64(0) }},
		{"fractional-size", "version", "better-math/v1/version.json", func(r map[string]any) { file(r, 0)["size_bytes"] = 0.5 }},
		{"no-locator", "version", "better-math/v1/version.json", func(r map[string]any) { file(r, 0)["locators"] = []any{} }},
		{"no-files", "version", "better-math/v1/version.json", func(r map[string]any) { r["files"] = []any{} }},
		{"unknown-role", "version", "better-math/v1/version.json", func(r map[string]any) { file(r, 0)["role"] = "submitted" }},
		{"derived-without-inputs", "version", "better-math/v1/version.json", func(r map[string]any) { delete(file(r, 1), "derived_from") }},
		{"derived-without-tool", "version", "better-math/v1/version.json", func(r map[string]any) { delete(file(r, 1), "tool") }},
		{"source-with-inputs", "version", "better-math/v1/version.json", func(r map[string]any) { file(r, 0)["derived_from"] = file(r, 1)["derived_from"] }},
		{"file-name-with-path", "version", "better-math/v1/version.json", func(r map[string]any) { file(r, 0)["name"] = "files/replacements.safetensors" }},
		{"hidden-operational-field", "version", "better-math/v1/version.json", func(r map[string]any) { r["ignore_mismatch"] = true }},
		{"unsupported-contract", "version", "better-math/v1/version.json", func(r map[string]any) { r["contract_version"] = float64(2) }},
		{"authored-targets", "version", "better-math/v1/version.json", func(r map[string]any) { r["targets"] = []any{"x"} }},
		{"old-kind", "patch", "better-math/patch.json", func(r map[string]any) { r["kind"] = "sparse-exact" }},
		{"whole-model-kind", "patch", "better-math/patch.json", func(r map[string]any) { r["kind"] = "whole" }},
		{"no-authors", "patch", "better-math/patch.json", func(r map[string]any) { r["authors"] = []any{} }},
		{"bad-submitter", "patch", "better-math/patch.json", func(r map[string]any) { r["submitter"] = "not a handle" }},
		{"uppercase-tag", "patch", "better-math/patch.json", func(r map[string]any) { r["tags"] = []any{"Math"} }},
		{"no-versions", "patch", "better-math/patch.json", func(r map[string]any) { r["versions"] = []any{} }},
		{"latest-as-status", "patch", "better-math/patch.json", func(r map[string]any) {
			r["versions"].([]any)[0].(map[string]any)["status"] = "latest"
		}},
		{"maintainers-field", "patch", "better-math/patch.json", func(r map[string]any) { r["maintainers"] = []any{"someone"} }},
		{"no-profiles", "base", "base.json", func(r map[string]any) { r["profiles"] = []any{} }},
		{"branch-revision", "base", "base.json", func(r map[string]any) {
			profile(r)["source"].(map[string]any)["revision"] = "main"
		}},
		{"uppercase-profile-id", "base", "base.json", func(r map[string]any) { profile(r)["id"] = "BF16" }},
		{"profile-without-files", "base", "base.json", func(r map[string]any) { profile(r)["files"] = []any{} }},
		{"run-without-outcomes", "run", "runs/baseline-bf16/run.json", func(r map[string]any) { delete(r, "outcomes") }},
		{"run-zero-denominator", "run", "runs/baseline-bf16/run.json", func(r map[string]any) {
			r["score"].(map[string]any)["denominator"] = float64(0)
		}},
		{"run-unknown-performer", "run", "runs/baseline-bf16/run.json", func(r map[string]any) { r["performed_by"] = "someone" }},
		{"run-unknown-score-without-owner", "run", "runs/baseline-bf16/run.json", func(r map[string]any) {
			r["score"] = map[string]any{"state": "UNKNOWN", "reason": "not scored"}
		}},
		{"matched-without-denominator", "comparison", "better-math/v1/comparisons/bf16-accuracy.json", func(r map[string]any) { delete(r, "denominator") }},
		{"matched-without-outcomes", "comparison", "better-math/v1/comparisons/bf16-accuracy.json", func(r map[string]any) { delete(r, "matched_outcomes") }},
		{"score-in-comparison", "comparison", "better-math/v1/comparisons/bf16-accuracy.json", func(r map[string]any) { r["score"] = float64(1) }},
		{"refused-without-reason", "comparison", "better-math/v1/comparisons/bf16-accuracy.json", func(r map[string]any) {
			r["state"] = "REFUSED"
			delete(r, "denominator")
			delete(r, "matched_outcomes")
			delete(r, "uncertainty")
		}},
		{"unknown-without-owner", "assessment", "better-math/v1/assessments/arithmetic-help.json", func(r map[string]any) { delete(r, "owner") }},
		{"unjustified-support", "assessment", "better-math/v1/assessments/arithmetic-help.json", func(r map[string]any) { r["decision"] = "SUPPORTED" }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := readJSON(t, example+tc.path)
			c.expect(t, tc.def, r, true)
			tc.edit(r)
			c.expect(t, tc.def, r, false)
		})
	}
}

func TestDecidedAssessment(t *testing.T) {
	c := newCheck(t)
	policy := readJSON(t, example+"better-math/v1/comparisons/bf16-accuracy.json")["matched_outcomes"]
	a := map[string]any{
		"contract_version": float64(1), "intended_use": "SYNTHETIC", "decision": "INCONCLUSIVE",
		"policy": policy, "comparisons": []any{"bf16-accuracy"}, "reason": "SYNTHETIC control",
	}
	for _, d := range []string{"SUPPORTED", "REJECT", "INCONCLUSIVE"} {
		a["decision"] = d
		c.expect(t, "assessment", a, true)
	}
	a["comparisons"] = []any{}
	c.expect(t, "assessment", a, false)
	a["comparisons"] = []any{"bf16-accuracy"}
	delete(a, "policy")
	c.expect(t, "assessment", a, false)
}

func TestGeneratedShapes(t *testing.T) {
	c := newCheck(t)
	base := readJSON(t, example+"base.json")
	version := readJSON(t, example+"better-math/v1/version.json")
	f := clone(version["files"].([]any)[0]).(map[string]any)
	f["format"] = "safetensors"
	run := map[string]any{"id": "baseline-bf16", "profile": "bf16", "performed_by": "index",
		"protocol": map[string]any{"name": "example-protocol", "sha256": strings.Repeat("d", 64)}, "state": "COMPLETE",
		"score": map[string]any{"metric": "accuracy", "value": 0.5, "unit": "fraction", "denominator": float64(200)}}
	index := map[string]any{
		"contract_version": float64(1), "base": "example-org/Example-Model-1B", "profiles": base["profiles"],
		"baseline_runs": []any{run},
		"patches": []any{map[string]any{
			"slug": "better-math", "title": "Better arithmetic", "summary": "SYNTHETIC", "kind": "replacement",
			"authors": []any{map[string]any{"name": "Example Author"}}, "submitter": "example-user", "license": "Apache-2.0",
			"tags": []any{}, "latest": float64(1),
			"versions": []any{map[string]any{
				"version": float64(1), "status": "active", "changelog": "First release.", "target_profile": "bf16",
				"targets": []any{"model.layers.0.mlp.down_proj.weight"}, "files": []any{f}, "runs": []any{}, "assessments": []any{},
			}},
		}},
	}
	c.expect(t, "index", index, true)
	bad := clone(index).(map[string]any)
	delete(bad["patches"].([]any)[0].(map[string]any)["versions"].([]any)[0].(map[string]any), "targets")
	c.expect(t, "index", bad, false)
	bad = clone(index).(map[string]any)
	bad["patches"].([]any)[0].(map[string]any)["versions"].([]any)[0].(map[string]any)["files"].([]any)[0].(map[string]any)["format"] = "pickle"
	c.expect(t, "index", bad, false)

	c.expect(t, "digestLookup", map[string]any{"contract_version": float64(1), "matches": []any{map[string]any{"base": "a/b", "profile": "bf16"}}}, true)
	c.expect(t, "digestLookup", map[string]any{"contract_version": float64(1), "matches": []any{}}, false)
	c.expect(t, "repoLookup", map[string]any{"contract_version": float64(1), "matches": []any{map[string]any{"base": "a/b", "profiles": []any{"q4"}}}}, true)
	c.expect(t, "repoLookup", map[string]any{"contract_version": float64(1), "matches": []any{map[string]any{"base": "a/b", "profiles": []any{}}}}, false)
	c.expect(t, "bases", map[string]any{"contract_version": float64(1), "bases": []any{"a/b/c"}}, false)
}

func TestLocatorControls(t *testing.T) {
	c := newCheck(t)
	local := map[string]any{"kind": "index-path", "path": "files/nested/adapter_model.safetensors"}
	hf := map[string]any{"kind": "hf", "repo_type": "model", "repo_id": "Example-Publisher/Example.Model_1", "revision": strings.Repeat("a", 40), "path": "nested/model.part-00001.safetensors"}
	c.expect(t, "locator", local, true)
	for _, typ := range []string{"model", "dataset"} {
		v := clone(hf).(map[string]any)
		v["repo_type"] = typ
		c.expect(t, "locator", v, true)
	}
	v := clone(hf).(map[string]any)
	v["repo_type"] = "space"
	c.expect(t, "locator", v, false)
	for _, path := range []string{"../escape", "/absolute", "a/../escape", "a/./b", "a//b", "a\\b", "a/%2e%2e/b", "a/b?x", "a/b#x", "a/\x00b", ".hidden/file", "a/b/", "a/b\n", "a/b\r\n"} {
		t.Run("path-"+path, func(t *testing.T) {
			for _, source := range []map[string]any{local, hf} {
				v := clone(source).(map[string]any)
				v["path"] = path
				c.expect(t, "locator", v, false)
			}
		})
	}
	for _, revision := range []string{"main", "v1", "abc1234", strings.Repeat("a", 39), strings.Repeat("g", 40), strings.Repeat("a", 40) + "\n"} {
		v := clone(hf).(map[string]any)
		v["revision"] = revision
		c.expect(t, "locator", v, false)
	}
	for _, repo := range []string{"name-only", "namespace/name/extra", "../name", "https://example.invalid/model", "name/other?x"} {
		v := clone(hf).(map[string]any)
		v["repo_id"] = repo
		c.expect(t, "locator", v, false)
	}
	v = clone(local).(map[string]any)
	v["storage"] = "git-lfs"
	c.expect(t, "locator", v, false)
	v = clone(hf).(map[string]any)
	v["url"] = "https://example.invalid/payload"
	c.expect(t, "locator", v, false)
}

func TestHelperControls(t *testing.T) {
	c := newCheck(t)
	for _, tc := range []struct {
		s  map[string]any
		v  any
		ok bool
	}{
		{map[string]any{"oneOf": []any{map[string]any{"type": "string"}, map[string]any{"minLength": float64(1)}}}, "x", false},
		{map[string]any{"type": "string", "minLength": float64(2)}, "é", false},
		{map[string]any{"type": "string", "minLength": float64(1)}, "é", true},
		{map[string]any{"type": "integer"}, 1.5, false},
		{map[string]any{"type": "integer"}, float64(1), true},
		{map[string]any{"$ref": "#/$defs/missing"}, "x", false},
		{map[string]any{"$ref": "https://example.invalid/schema"}, "x", false},
	} {
		if err := c.validate(tc.s, tc.v); (err == nil) != tc.ok {
			t.Fatalf("helper control: %v, want accepted=%v", err, tc.ok)
		}
	}
	if err := checkKeywords(map[string]any{"format": "unimplemented"}); err == nil {
		t.Fatal("unimplemented keyword silently ignored")
	}
}
