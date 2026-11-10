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

func readJSON(t *testing.T, path string) map[string]any {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var v map[string]any
	if err := json.Unmarshal(b, &v); err != nil {
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

func newCheck(t *testing.T) schemaCheck {
	t.Helper()
	c := schemaCheck{readJSON(t, "catalog.schema.json")}
	if err := checkKeywords(c.root); err != nil {
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

func (c schemaCheck) definition(t *testing.T, name string, v any, want bool) {
	t.Helper()
	err := c.validate(map[string]any{"$ref": "#/$defs/" + name}, v)
	if (err == nil) != want {
		t.Fatalf("%s: accepted=%v, want=%v: %v", name, err == nil, want, err)
	}
}

func treatment(record map[string]any) map[string]any {
	return record["treatments"].([]any)[0].(map[string]any)
}

func payload(record map[string]any) map[string]any {
	return treatment(record)["payload"].(map[string]any)
}

func TestShippedDocuments(t *testing.T) {
	c := newCheck(t)
	for _, file := range []string{"../catalog.json", "../examples/TEMPLATE.json"} {
		t.Run(file, func(t *testing.T) {
			if err := c.validate(c.root, readJSON(t, file)); err != nil {
				t.Fatal(err)
			}
		})
	}
	if got := len(readJSON(t, "../catalog.json")["entries"].([]any)); got != 0 {
		t.Fatalf("scaffold contains %d real entries", got)
	}
}

func TestBadRecordControls(t *testing.T) {
	c := newCheck(t)
	cases := []struct {
		name string
		edit func(map[string]any)
	}{
		{"missing-payload-digest", func(r map[string]any) { delete(payload(r), "digest") }},
		{"missing-base-digest", func(r map[string]any) {
			p := r["profiles"].([]any)[0].(map[string]any)
			delete(p["files"].([]any)[0].(map[string]any), "digest")
		}},
		{"unmarked-advertised-hash", func(r map[string]any) { delete(payload(r)["digest"].(map[string]any), "status") }},
		{"bare-hash-string", func(r map[string]any) { payload(r)["digest"] = strings.Repeat("a", 64) }},
		{"verified-without-evidence", func(r map[string]any) { payload(r)["digest"].(map[string]any)["status"] = "verified" }},
		{"bad-digest", func(r map[string]any) { payload(r)["digest"].(map[string]any)["value"] = "advertised" }},
		{"missing-size", func(r map[string]any) { delete(payload(r), "size_bytes") }},
		{"negative-size", func(r map[string]any) { payload(r)["size_bytes"] = float64(-1) }},
		{"fractional-size", func(r map[string]any) { payload(r)["size_bytes"] = 0.5 }},
		{"no-location", func(r map[string]any) { payload(r)["locators"] = []any{} }},
		{"no-profiles", func(r map[string]any) { r["profiles"] = []any{} }},
		{"no-treatments", func(r map[string]any) { r["treatments"] = []any{} }},
		{"no-assessments", func(r map[string]any) { r["assessments"] = []any{} }},
		{"template-promoted", func(r map[string]any) { r["document_kind"] = "catalog" }},
		{"unsupported-version", func(r map[string]any) { r["contract_version"] = float64(2) }},
		{"hidden-operational-field", func(r map[string]any) { treatment(r)["ignore_mismatch"] = true }},
		{"lora-rationale-on-sparse", func(r map[string]any) { treatment(r)["rationale"] = "low-rank-factors" }},
		{"lora-block-on-sparse", func(r map[string]any) { treatment(r)["members"] = []any{} }},
		{"unjustified-support", func(r map[string]any) { r["assessments"].([]any)[0].(map[string]any)["quality"] = "SUPPORTED" }},
		{"score-in-assessment", func(r map[string]any) { r["assessments"].([]any)[0].(map[string]any)["score"] = float64(1) }},
		{"missing-unknown-owner", func(r map[string]any) { delete(treatment(r)["route"].(map[string]any), "owner") }},
		{"undeclared-run", func(r map[string]any) { r["runs"] = []any{map[string]any{"score": float64(1)}} }},
		{"undeclared-comparison", func(r map[string]any) { r["comparisons"] = []any{map[string]any{"gain": float64(1)}} }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := readJSON(t, "../examples/TEMPLATE.json")
			tc.edit(r)
			if err := c.validate(c.root, r); err == nil {
				t.Fatal("deliberately bad record accepted")
			}
		})
	}
}

func TestLocatorCompatibility(t *testing.T) {
	c := newCheck(t)
	local := map[string]any{"kind": "index-path", "storage": "git", "path": "payloads/example-v1/nested/adapter_model.safetensors"}
	hf := map[string]any{"kind": "hf", "repo_type": "model", "repo_id": "Example-Publisher/Example.Model_1", "revision": strings.Repeat("a", 40), "path": "nested/model.part-00001.safetensors"}
	c.definition(t, "locator", local, true)
	for _, typ := range []string{"model", "dataset", "space"} {
		v := clone(hf).(map[string]any)
		v["repo_type"] = typ
		c.definition(t, "locator", v, true)
	}
	for _, path := range []string{"../escape", "/absolute", "a/../escape", "a/./b", "a//b", "a\\b", "a/%2e%2e/b", "a/b?x", "a/b#x", "a/\x00b", ".hidden/file", "a/b/", "a/b\n", "a/b\r\n"} {
		t.Run("path-"+path, func(t *testing.T) {
			for _, source := range []map[string]any{local, hf} {
				v := clone(source).(map[string]any)
				v["path"] = path
				c.definition(t, "locator", v, false)
			}
		})
	}
	for _, revision := range []string{"main", "v1", "abc1234", strings.Repeat("a", 39), strings.Repeat("g", 40), strings.Repeat("a", 40) + "\n"} {
		v := clone(hf).(map[string]any)
		v["revision"] = revision
		c.definition(t, "locator", v, false)
	}
	for _, repo := range []string{"name-only", "namespace/name/extra", "../name", "https://example.invalid/model", "name/other?x"} {
		v := clone(hf).(map[string]any)
		v["repo_id"] = repo
		c.definition(t, "locator", v, false)
	}
	v := clone(local).(map[string]any)
	v["storage"] = "git-lfs"
	c.definition(t, "locator", v, false)
	v = clone(hf).(map[string]any)
	v["url"] = "https://example.invalid/payload"
	c.definition(t, "locator", v, false)
}

func TestKindSeparation(t *testing.T) {
	c := newCheck(t)
	r := readJSON(t, "../examples/TEMPLATE.json")
	sparse := treatment(r)
	a := payload(r)
	base := map[string]any{"id": "template-treatment", "base_profile_ref": "template-native", "route": sparse["route"], "output_identity": sparse["output_identity"]}
	lora := clone(base).(map[string]any)
	lora["kind"], lora["rationale"], lora["mode"], lora["conflict_policy"] = "lora", "low-rank-factors", "runtime-adapter", "TEMPLATE: qualify ordered targets"
	lora["members"] = []any{map[string]any{"payload": a, "configuration": a, "scale": float64(1), "targets": []any{"TEMPLATE.target"}}}
	whole := clone(base).(map[string]any)
	whole["kind"], whole["rationale"], whole["files"], whole["provenance_manifest"], whole["ordered_inputs"] = "whole-artifact", "complete-output", []any{a}, a, []any{"template-native"}
	for _, kind := range []map[string]any{sparse, lora, whole} {
		c.definition(t, "treatment", kind, true)
		for _, other := range []string{"exact-replacements", "low-rank-factors", "complete-output"} {
			if other == kind["rationale"] {
				continue
			}
			bad := clone(kind).(map[string]any)
			bad["rationale"] = other
			c.definition(t, "treatment", bad, false)
		}
	}
	bad := clone(whole).(map[string]any)
	delete(bad, "provenance_manifest")
	c.definition(t, "treatment", bad, false)
	bad = clone(lora).(map[string]any)
	delete(bad["members"].([]any)[0].(map[string]any), "configuration")
	c.definition(t, "treatment", bad, false)
}

func TestDigestEvidenceAndReferences(t *testing.T) {
	c := newCheck(t)
	d := map[string]any{"algorithm": "sha256", "value": strings.Repeat("a", 64), "status": "verified", "source": "SYNTHETIC control", "verification": map[string]any{"method": "whole-file-sha256", "evidence_ref": "template-evidence"}}
	c.definition(t, "digest", d, true)
	d["status"] = "advertised"
	c.definition(t, "digest", d, false)
	delete(d, "verification")
	c.definition(t, "digest", d, true)
	c.definition(t, "reference", map[string]any{"bundle_sha256": strings.Repeat("a", 64), "pointer": "/profiles/0"}, true)
	c.definition(t, "reference", map[string]any{"bundle_sha256": "missing", "pointer": "/profiles/0"}, false)
	c.definition(t, "reference", map[string]any{"bundle_sha256": strings.Repeat("a", 64), "pointer": "/profiles/../0"}, false)
}

func TestArtifactDigestRequiredInBothModes(t *testing.T) {
	c := newCheck(t)
	a := payload(readJSON(t, "../examples/TEMPLATE.json"))
	hf := map[string]any{"kind": "hf", "repo_type": "model", "repo_id": "Example-Publisher/Example.Model_1", "revision": strings.Repeat("a", 40), "path": "nested/payload.npz"}
	for _, locator := range []any{a["locators"].([]any)[0], hf} {
		good := clone(a).(map[string]any)
		good["locators"] = []any{locator}
		c.definition(t, "artifact", good, true)
		noDigest := clone(good).(map[string]any)
		delete(noDigest, "digest")
		c.definition(t, "artifact", noDigest, false)
		unmarked := clone(good).(map[string]any)
		delete(unmarked["digest"].(map[string]any), "status")
		c.definition(t, "artifact", unmarked, false)
	}
}

func TestEvidenceRoleControls(t *testing.T) {
	c := newCheck(t)
	r := readJSON(t, "../examples/TEMPLATE.json")
	a := payload(r) // All identities and numeric values below are SYNTHETIC controls.
	unknown := treatment(r)["route"]
	run := map[string]any{
		"id": "template-run", "profile_ref": "template-native", "treatment_ref": "template-sparse", "state": "INCOMPLETE",
		"protocol": a, "dataset": a, "item_draw_manifest": a, "outcomes": a,
		"coverage": map[string]any{"expected_item_draws": float64(2), "completed_item_draws": float64(1), "failure_policy": "TEMPLATE: retain incomplete work", "errors_and_truncations": "TEMPLATE: outcomes retain every attempt"},
		"score":    unknown, "reason": "TEMPLATE ONLY: no observations exist",
	}
	c.definition(t, "run", run, true)
	badRun := clone(run).(map[string]any)
	delete(badRun, "outcomes")
	c.definition(t, "run", badRun, false)
	run["state"] = "COMPLETE"
	run["coverage"].(map[string]any)["completed_item_draws"] = float64(2)
	run["score"] = map[string]any{"metric": "SYNTHETIC", "value": float64(1), "unit": "SYNTHETIC", "denominator": float64(1)}
	c.definition(t, "run", run, true)
	badRun = clone(run).(map[string]any)
	badRun["score"].(map[string]any)["denominator"] = float64(0)
	c.definition(t, "run", badRun, false)
	comparison := map[string]any{
		"id": "template-comparison", "baseline_run_ref": "template-base-run", "candidate_run_ref": "template-run",
		"estimand": "TEMPLATE: declared within-profile contrast", "selection_design": "TEMPLATE: frozen selection", "weights": unknown,
		"reason": "TEMPLATE ONLY: no comparison exists", "state": "MATCHED", "denominator": float64(1), "matched_outcomes": a, "analysis": a, "uncertainty": unknown,
	}
	c.definition(t, "comparison", comparison, true)
	badComparison := clone(comparison).(map[string]any)
	delete(badComparison, "matched_outcomes")
	c.definition(t, "comparison", badComparison, false)
	badComparison = clone(comparison).(map[string]any)
	badComparison["score"] = float64(1)
	c.definition(t, "comparison", badComparison, false)
	comparison["state"], comparison["denominator"], comparison["uncertainty"] = "REFUSED", unknown, unknown
	c.definition(t, "comparison", comparison, true)
	assessment := clone(r["assessments"].([]any)[0]).(map[string]any)
	assessment["quality"], assessment["policy"], assessment["comparisons"] = "INCONCLUSIVE", a, []any{"template-comparison"}
	c.definition(t, "assessment", assessment, true)
	for _, decision := range []string{"SUPPORTED", "REJECT", "INCONCLUSIVE"} {
		assessment["quality"] = decision
		c.definition(t, "assessment", assessment, true)
	}
	assessment["comparisons"] = []any{}
	c.definition(t, "assessment", assessment, false)
	bundle := map[string]any{"artifact": a, "inventory_pointer": ""}
	c.definition(t, "bundle", bundle, true)
	bundle["inventory_pointer"] = "/positive-subset"
	c.definition(t, "bundle", bundle, false)
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
