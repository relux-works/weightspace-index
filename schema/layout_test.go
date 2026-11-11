package schema_test

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

// The layout checks cover what one JSON file cannot say about itself: where
// files may live, how versions line up with their folders, which records
// refer to which, and whether files hosted here match their recorded digests.

var (
	slugRe    = regexp.MustCompile(`^[a-z0-9]+(?:-[a-z0-9]+)*\z`)
	idRe      = regexp.MustCompile(`^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*\z`)
	versionRe = regexp.MustCompile(`^v[1-9][0-9]*\z`)
	sha256Re  = regexp.MustCompile(`^[a-f0-9]{64}\z`)
)

type layoutCheck struct {
	c    schemaCheck
	errs []error
}

func (l *layoutCheck) fail(format string, args ...any) {
	l.errs = append(l.errs, fmt.Errorf(format, args...))
}

func (l *layoutCheck) readDir(dir string) []fs.DirEntry {
	entries, err := os.ReadDir(dir)
	if err != nil {
		l.fail("%v", err)
	}
	return entries
}

func (l *layoutCheck) record(path, def string) map[string]any {
	v, err := readJSONFile(path)
	if err != nil {
		l.fail("%v", err)
		return nil
	}
	if err := l.c.as(def, v); err != nil {
		l.fail("%s: %v", path, err)
		return nil
	}
	return v
}

// checkModels checks a models/ tree: models/<namespace>/<model>/...
func checkModels(c schemaCheck, root string) []error {
	l := &layoutCheck{c: c}
	for _, ns := range l.readDir(root) {
		nsDir := filepath.Join(root, ns.Name())
		if !ns.IsDir() {
			l.fail("%s: only namespace folders belong here", nsDir)
			continue
		}
		for _, model := range l.readDir(nsDir) {
			if !model.IsDir() {
				l.fail("%s: only model folders belong here", filepath.Join(nsDir, model.Name()))
				continue
			}
			l.base(filepath.Join(nsDir, model.Name()))
		}
	}
	return l.errs
}

func (l *layoutCheck) base(dir string) {
	var base map[string]any
	baseRuns := map[string]map[string]any{}
	var slugs []string
	for _, e := range l.readDir(dir) {
		name, p := e.Name(), filepath.Join(dir, e.Name())
		switch {
		case name == "base.json" && !e.IsDir():
			base = l.record(p, "base")
		case name == "index.json" && !e.IsDir():
			l.record(p, "index")
		case name == "README.md" && !e.IsDir():
		case name == "runs" && e.IsDir():
			baseRuns = l.runs(p)
		case e.IsDir() && slugRe.MatchString(name):
			slugs = append(slugs, name)
		default:
			l.fail("%s: unexpected path", p)
		}
	}
	if base == nil {
		l.fail("%s: missing or invalid base.json", dir)
		return
	}
	profiles := map[string]bool{}
	for _, raw := range base["profiles"].([]any) {
		p := raw.(map[string]any)
		id := p["id"].(string)
		if profiles[id] {
			l.fail("%s: duplicate profile %s", dir, id)
		}
		profiles[id] = true
		paths := map[string]bool{}
		for _, f := range p["files"].([]any) {
			path := f.(map[string]any)["path"].(string)
			if paths[path] {
				l.fail("%s: profile %s lists %s twice", dir, id, path)
			}
			paths[path] = true
		}
	}
	for id, run := range baseRuns {
		if run != nil && !profiles[run["profile"].(string)] {
			l.fail("%s: baseline run %s names unknown profile %s", dir, id, run["profile"])
		}
	}
	for _, slug := range slugs {
		l.patch(filepath.Join(dir, slug), profiles, baseRuns)
	}
}

// runs checks a runs/ folder and returns its runs by ID. Outcome and
// resource files may sit beside run.json; no other JSON may.
func (l *layoutCheck) runs(dir string) map[string]map[string]any {
	out := map[string]map[string]any{}
	for _, e := range l.readDir(dir) {
		p := filepath.Join(dir, e.Name())
		if !e.IsDir() || !idRe.MatchString(e.Name()) {
			l.fail("%s: a run is a folder named by its ID", p)
			continue
		}
		run := l.record(filepath.Join(p, "run.json"), "run")
		out[e.Name()] = run
		for _, f := range l.readDir(p) {
			if f.IsDir() || (f.Name() != "run.json" && strings.HasSuffix(f.Name(), ".json")) {
				l.fail("%s: unexpected path", filepath.Join(p, f.Name()))
			}
		}
		if run != nil {
			l.hosted(p, run, "", false)
		}
	}
	return out
}

func (l *layoutCheck) patch(dir string, profiles map[string]bool, baseRuns map[string]map[string]any) {
	var meta map[string]any
	folders := map[string]bool{}
	for _, e := range l.readDir(dir) {
		name, p := e.Name(), filepath.Join(dir, e.Name())
		switch {
		case name == "patch.json" && !e.IsDir():
			meta = l.record(p, "patch")
		case name == "README.md" && !e.IsDir():
		case name == "notices" && e.IsDir():
		case e.IsDir() && versionRe.MatchString(name):
			folders[name] = true
		default:
			l.fail("%s: unexpected path", p)
		}
	}
	if meta == nil {
		l.fail("%s: missing or invalid patch.json", dir)
		return
	}
	statuses := meta["versions"].([]any)
	for i, raw := range statuses {
		v := raw.(map[string]any)
		folder := fmt.Sprintf("v%d", i+1)
		if int(v["version"].(float64)) != i+1 {
			l.fail("%s: versions must be listed as 1..N without gaps", dir)
			return
		}
		if !folders[folder] {
			l.fail("%s: version %d has no folder", dir, i+1)
			continue
		}
		delete(folders, folder)
		l.version(filepath.Join(dir, folder), meta["kind"].(string), v["status"].(string) == "withdrawn", profiles, baseRuns)
	}
	for folder := range folders {
		l.fail("%s: %s is not listed in patch.json", dir, folder)
	}
}

func (l *layoutCheck) version(dir, kind string, withdrawn bool, profiles map[string]bool, baseRuns map[string]map[string]any) {
	var ver map[string]any
	runs := map[string]map[string]any{}
	records := map[string][]string{}
	for _, e := range l.readDir(dir) {
		name, p := e.Name(), filepath.Join(dir, e.Name())
		switch {
		case name == "version.json" && !e.IsDir():
			ver = l.record(p, "version")
		case name == "files" && e.IsDir():
		case name == "runs" && e.IsDir():
			runs = l.runs(p)
		case (name == "comparisons" || name == "assessments") && e.IsDir():
			for _, f := range l.readDir(p) {
				id, ok := strings.CutSuffix(f.Name(), ".json")
				if f.IsDir() || !ok || !idRe.MatchString(id) {
					l.fail("%s: expected <id>.json", filepath.Join(p, f.Name()))
					continue
				}
				records[name] = append(records[name], id)
			}
		default:
			l.fail("%s: unexpected path", p)
		}
	}
	if ver == nil {
		l.fail("%s: missing or invalid version.json", dir)
		return
	}
	if !profiles[ver["target_profile"].(string)] {
		l.fail("%s: unknown target profile %s", dir, ver["target_profile"])
	}

	names := map[string]bool{}
	sourceDigests := map[string]bool{}
	var sources []string
	files := ver["files"].([]any)
	for _, raw := range files {
		f := raw.(map[string]any)
		name := f["name"].(string)
		if names[name] {
			l.fail("%s: file name %s used twice", dir, name)
		}
		names[name] = true
		if f["role"] == "source" {
			sources = append(sources, name)
			sourceDigests[f["sha256"].(string)] = true
		}
	}
	for _, raw := range files {
		f := raw.(map[string]any)
		if f["role"] != "derived" {
			continue
		}
		for _, d := range f["derived_from"].([]any) {
			if !sourceDigests[d.(string)] {
				l.fail("%s: %s is derived from a file that is not a source of this version", dir, f["name"])
			}
		}
		if !hasSuffix(f["name"].(string), ".safetensors", ".gguf", ".json") {
			l.fail("%s: derived file %s is not safetensors, GGUF or JSON", dir, f["name"])
		}
	}
	if err := checkSources(kind, sources); err != nil {
		l.fail("%s: %v", dir, err)
	}

	referenced := l.hosted(dir, ver, "files/", withdrawn)
	if filesDir := filepath.Join(dir, "files"); !withdrawn {
		filepath.WalkDir(filesDir, func(p string, d fs.DirEntry, err error) error {
			if err != nil || d.IsDir() {
				return nil
			}
			rel, _ := filepath.Rel(dir, p)
			if !referenced[filepath.ToSlash(rel)] {
				l.fail("%s: hosted file is not listed in version.json", p)
			}
			return nil
		})
	}
	if kind == "replacement" && !withdrawn {
		for _, raw := range files {
			f := raw.(map[string]any)
			for _, loc := range f["locators"].([]any) {
				loc := loc.(map[string]any)
				if f["role"] == "source" && loc["kind"] == "index-path" {
					if err := checkReplacementFile(filepath.Join(dir, filepath.FromSlash(loc["path"].(string)))); err != nil {
						l.fail("%s: %s: %v", dir, f["name"], err)
					}
				}
			}
		}
	}

	for id, run := range runs {
		if run != nil && !profiles[run["profile"].(string)] {
			l.fail("%s: run %s names unknown profile %s", dir, id, run["profile"])
		}
	}
	comparisons := map[string]bool{}
	for _, id := range records["comparisons"] {
		comparisons[id] = true
		cmp := l.record(filepath.Join(dir, "comparisons", id+".json"), "comparison")
		if cmp == nil {
			continue
		}
		baseline, patched := baseRuns[cmp["baseline_run"].(string)], runs[cmp["patched_run"].(string)]
		if baseline == nil {
			l.fail("%s: comparison %s names unknown baseline run %s", dir, id, cmp["baseline_run"])
		}
		if patched == nil {
			l.fail("%s: comparison %s names unknown run %s of this version", dir, id, cmp["patched_run"])
		}
		if baseline != nil && patched != nil && baseline["profile"] != patched["profile"] {
			l.fail("%s: comparison %s compares runs on different profiles", dir, id)
		}
		l.hosted(filepath.Join(dir, "comparisons"), cmp, "", false)
	}
	for _, id := range records["assessments"] {
		a := l.record(filepath.Join(dir, "assessments", id+".json"), "assessment")
		if a == nil {
			continue
		}
		cited, _ := a["comparisons"].([]any)
		for _, c := range cited {
			if !comparisons[c.(string)] {
				l.fail("%s: assessment %s cites unknown comparison %s", dir, id, c)
			}
		}
	}
}

// checkSources enforces the two source forms.
func checkSources(kind string, sources []string) error {
	sort.Strings(sources)
	switch kind {
	case "replacement":
		for _, s := range sources {
			if !strings.HasSuffix(s, ".safetensors") {
				return fmt.Errorf("replacement source %s is not .safetensors", s)
			}
		}
		if len(sources) == 0 {
			return errors.New("no source file")
		}
	case "lora":
		peft := len(sources) == 2 && sources[0] == "adapter_config.json" && sources[1] == "adapter_model.safetensors"
		gguf := len(sources) == 1 && strings.HasSuffix(sources[0], ".gguf")
		if !peft && !gguf {
			return fmt.Errorf("LoRA sources must be adapter_model.safetensors with adapter_config.json, or one .gguf; got %v", sources)
		}
	}
	return nil
}

// hosted finds every artifact in rec with an index-path locator, checks the
// path stays under prefix, and unless the version is withdrawn checks the
// file exists with the recorded size and SHA-256. Paths are relative to dir,
// the folder of the declaring file. It returns the referenced paths.
func (l *layoutCheck) hosted(dir string, rec any, prefix string, withdrawn bool) map[string]bool {
	referenced := map[string]bool{}
	var walk func(v any)
	walk = func(v any) {
		switch v := v.(type) {
		case []any:
			for _, x := range v {
				walk(x)
			}
		case map[string]any:
			locs, ok := v["locators"].([]any)
			if !ok {
				for _, x := range v {
					walk(x)
				}
				return
			}
			for _, raw := range locs {
				loc := raw.(map[string]any)
				if loc["kind"] != "index-path" {
					continue
				}
				rel := loc["path"].(string)
				referenced[rel] = true
				if !strings.HasPrefix(rel, prefix) {
					l.fail("%s: hosted path %s must be under %s", dir, rel, prefix)
					continue
				}
				if withdrawn {
					continue
				}
				p := filepath.Join(dir, filepath.FromSlash(rel))
				info, err := os.Lstat(p)
				if err != nil || !info.Mode().IsRegular() {
					l.fail("%s: hosted file missing", p)
					continue
				}
				b, _ := os.ReadFile(p)
				sum := sha256.Sum256(b)
				if hex.EncodeToString(sum[:]) != v["sha256"] || float64(len(b)) != v["size_bytes"] {
					l.fail("%s: size or SHA-256 differs from the record", p)
				}
				if len(b) >= 100_000_000 {
					l.fail("%s: files hosted here must be under 100 MB", p)
				}
			}
		}
	}
	walk(rec)
	return referenced
}

func hasSuffix(s string, suffixes ...string) bool {
	for _, x := range suffixes {
		if strings.HasSuffix(s, x) {
			return true
		}
	}
	return false
}

type stTensor struct {
	Dtype   string   `json:"dtype"`
	Shape   []int64  `json:"shape"`
	Offsets [2]int64 `json:"data_offsets"`
}

// checkReplacementFile checks the replacement convention: every tensor is
// either a full tensor or a complete sparse trio <name>#idx/#new/#old, with
// sorted unique int64 indices and matching value shapes and dtypes.
func checkReplacementFile(path string) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if len(b) < 8 {
		return errors.New("not safetensors")
	}
	n := binary.LittleEndian.Uint64(b[:8])
	if n > uint64(len(b)-8) {
		return errors.New("safetensors header out of range")
	}
	var header map[string]json.RawMessage
	if err := json.Unmarshal(b[8:8+n], &header); err != nil {
		return fmt.Errorf("safetensors header: %w", err)
	}
	data := b[8+n:]
	tensors := map[string]stTensor{}
	sparse := map[string]map[string]stTensor{}
	for name, raw := range header {
		if name == "__metadata__" {
			continue
		}
		var t stTensor
		if err := json.Unmarshal(raw, &t); err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
		if t.Offsets[0] < 0 || t.Offsets[0] > t.Offsets[1] || t.Offsets[1] > int64(len(data)) {
			return fmt.Errorf("%s: data offsets out of range", name)
		}
		base, suffix, isSparse := strings.Cut(name, "#")
		if !isSparse {
			tensors[name] = t
			continue
		}
		if suffix != "idx" && suffix != "new" && suffix != "old" {
			return fmt.Errorf("%s: unknown suffix #%s", name, suffix)
		}
		if sparse[base] == nil {
			sparse[base] = map[string]stTensor{}
		}
		sparse[base][suffix] = t
	}
	if len(tensors)+len(sparse) == 0 {
		return errors.New("no tensors")
	}
	for base, trio := range sparse {
		if _, ok := tensors[base]; ok {
			return fmt.Errorf("%s: both full and sparse", base)
		}
		idx, okIdx := trio["idx"]
		newV, okNew := trio["new"]
		oldV, okOld := trio["old"]
		if !okIdx || !okNew || !okOld {
			return fmt.Errorf("%s: sparse tensors need #idx, #new and #old", base)
		}
		if idx.Dtype != "I64" || len(idx.Shape) != 1 || idx.Shape[0] < 1 || idx.Offsets[1]-idx.Offsets[0] != 8*idx.Shape[0] {
			return fmt.Errorf("%s#idx: must be a non-empty 1-D I64 tensor", base)
		}
		if newV.Dtype != oldV.Dtype || fmt.Sprint(newV.Shape) != fmt.Sprint(idx.Shape) || fmt.Sprint(oldV.Shape) != fmt.Sprint(idx.Shape) {
			return fmt.Errorf("%s: #new and #old must match #idx in shape and each other in dtype", base)
		}
		prev := int64(-1)
		for i := idx.Offsets[0]; i < idx.Offsets[1]; i += 8 {
			v := int64(binary.LittleEndian.Uint64(data[i : i+8]))
			if v <= prev {
				return fmt.Errorf("%s#idx: indices must be sorted, unique and non-negative", base)
			}
			prev = v
		}
	}
	return nil
}

// checkLookups checks generated by-digest/ and by-repo/ trees when present.
func checkLookups(c schemaCheck, root string) []error {
	l := &layoutCheck{c: c}
	digests := filepath.Join(root, "by-digest")
	filepath.WalkDir(digests, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		rel, _ := filepath.Rel(digests, p)
		parts := strings.Split(filepath.ToSlash(rel), "/")
		sum, ok := strings.CutSuffix(parts[len(parts)-1], ".json")
		if len(parts) != 2 || !ok || !sha256Re.MatchString(sum) || parts[0] != sum[:2] {
			l.fail("%s: expected by-digest/<aa>/<sha256>.json", p)
			return nil
		}
		l.record(p, "digestLookup")
		return nil
	})
	repos := filepath.Join(root, "by-repo")
	filepath.WalkDir(repos, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		rel, _ := filepath.Rel(repos, p)
		if len(strings.Split(filepath.ToSlash(rel), "/")) != 2 || !strings.HasSuffix(p, ".json") {
			l.fail("%s: expected by-repo/<namespace>/<repo>.json", p)
			return nil
		}
		l.record(p, "repoLookup")
		return nil
	})
	return l.errs
}

func TestExampleTree(t *testing.T) {
	c := newCheck(t)
	for _, err := range checkModels(c, "../examples/models") {
		t.Error(err)
	}
}

func TestRepositoryTree(t *testing.T) {
	c := newCheck(t)
	if _, err := os.Stat("../models"); err == nil {
		for _, err := range checkModels(c, "../models") {
			t.Error(err)
		}
	}
	for _, err := range checkLookups(c, "..") {
		t.Error(err)
	}
}

// Rule 10 and rule 3: no symlinks, and no two paths differing only in case.
func TestNoSymlinksOrCaseCollisions(t *testing.T) {
	filepath.WalkDir("..", func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			t.Error(err)
			return nil
		}
		if d.Name() == ".git" && d.IsDir() {
			return filepath.SkipDir
		}
		if d.Type()&fs.ModeSymlink != 0 {
			t.Errorf("%s: symlinks are not allowed", p)
		}
		if d.IsDir() {
			entries, _ := os.ReadDir(p)
			seen := map[string]string{}
			for _, e := range entries {
				k := strings.ToLower(e.Name())
				if other, ok := seen[k]; ok {
					t.Errorf("%s: %s and %s differ only in case", p, other, e.Name())
				}
				seen[k] = e.Name()
			}
		}
		return nil
	})
}

// --- controls: each defect in a copy of the example tree must be caught ---

const exampleBase = "example-org/Example-Model-1B/"

func editJSON(t *testing.T, path string, edit func(map[string]any)) {
	t.Helper()
	v := readJSON(t, path)
	edit(v)
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
}

type stEntry struct {
	name, dtype string
	shape       []int64
	data        []byte
}

func buildSafetensors(entries []stEntry) []byte {
	header := map[string]any{}
	var data []byte
	for _, e := range entries {
		header[e.name] = map[string]any{"dtype": e.dtype, "shape": e.shape, "data_offsets": []int{len(data), len(data) + len(e.data)}}
		data = append(data, e.data...)
	}
	h, _ := json.Marshal(header)
	out := binary.LittleEndian.AppendUint64(nil, uint64(len(h)))
	return append(append(out, h...), data...)
}

func i64s(vs ...int64) []byte {
	var b []byte
	for _, v := range vs {
		b = binary.LittleEndian.AppendUint64(b, uint64(v))
	}
	return b
}

// rehost replaces better-math v1's hosted file and updates its record.
func rehost(t *testing.T, root string, b []byte) {
	t.Helper()
	dir := filepath.Join(root, exampleBase, "better-math/v1")
	if err := os.WriteFile(filepath.Join(dir, "files/replacements.safetensors"), b, 0o644); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(b)
	editJSON(t, filepath.Join(dir, "version.json"), func(v map[string]any) {
		files := v["files"].([]any)
		files[0].(map[string]any)["sha256"] = hex.EncodeToString(sum[:])
		files[0].(map[string]any)["size_bytes"] = float64(len(b))
		files[1].(map[string]any)["derived_from"] = []any{hex.EncodeToString(sum[:])}
	})
}

func sparse(idx []int64, withOld bool) []byte {
	const w = "model.layers.0.mlp.down_proj.weight"
	n := int64(len(idx))
	entries := []stEntry{
		{w + "#idx", "I64", []int64{n}, i64s(idx...)},
		{w + "#new", "BF16", []int64{n}, make([]byte, 2*n)},
	}
	if withOld {
		entries = append(entries, stEntry{w + "#old", "BF16", []int64{n}, make([]byte, 2*n)})
	}
	return buildSafetensors(entries)
}

func TestLayoutControls(t *testing.T) {
	c := newCheck(t)
	bm := exampleBase + "better-math/"
	sa := exampleBase + "shorter-answers/"
	cases := []struct {
		name string
		ok   bool
		edit func(t *testing.T, root string)
	}{
		{"unchanged", true, func(*testing.T, string) {}},
		{"rebuilt-valid-sparse", true, func(t *testing.T, root string) { rehost(t, root, sparse([]int64{1, 5, 9}, true)) }},
		{"withdrawn-files-removed", true, func(t *testing.T, root string) {
			editJSON(t, filepath.Join(root, bm, "patch.json"), func(v map[string]any) {
				v["versions"].([]any)[0].(map[string]any)["status"] = "withdrawn"
			})
			os.RemoveAll(filepath.Join(root, bm, "v1/files"))
		}},
		{"version-folder-missing", false, func(t *testing.T, root string) { os.RemoveAll(filepath.Join(root, sa, "v1")) }},
		{"version-folder-unlisted", false, func(t *testing.T, root string) {
			os.CopyFS(filepath.Join(root, sa, "v3"), os.DirFS(filepath.Join(root, sa, "v2")))
		}},
		{"version-gap", false, func(t *testing.T, root string) {
			editJSON(t, filepath.Join(root, sa, "patch.json"), func(v map[string]any) {
				v["versions"].([]any)[1].(map[string]any)["version"] = float64(3)
			})
		}},
		{"bad-slug-folder", false, func(t *testing.T, root string) {
			os.Rename(filepath.Join(root, bm), filepath.Join(root, exampleBase, "Better_Math"))
		}},
		{"unknown-file-in-base", false, func(t *testing.T, root string) {
			os.WriteFile(filepath.Join(root, exampleBase, "notes.txt"), []byte("x"), 0o644)
		}},
		{"unknown-json-in-version", false, func(t *testing.T, root string) {
			os.WriteFile(filepath.Join(root, bm, "v1/extra.json"), []byte("{}"), 0o644)
		}},
		{"missing-base-json", false, func(t *testing.T, root string) { os.Remove(filepath.Join(root, exampleBase, "base.json")) }},
		{"duplicate-profile", false, func(t *testing.T, root string) {
			editJSON(t, filepath.Join(root, exampleBase, "base.json"), func(v map[string]any) {
				ps := v["profiles"].([]any)
				ps[1].(map[string]any)["id"] = "bf16"
			})
		}},
		{"unknown-target-profile", false, func(t *testing.T, root string) {
			editJSON(t, filepath.Join(root, bm, "v1/version.json"), func(v map[string]any) { v["target_profile"] = "fp8" })
		}},
		{"run-unknown-profile", false, func(t *testing.T, root string) {
			editJSON(t, filepath.Join(root, bm, "v1/runs/patched-bf16/run.json"), func(v map[string]any) { v["profile"] = "fp8" })
		}},
		{"hosted-bytes-changed", false, func(t *testing.T, root string) {
			p := filepath.Join(root, bm, "v1/files/replacements.safetensors")
			b, _ := os.ReadFile(p)
			b[len(b)-1] ^= 1
			os.WriteFile(p, b, 0o644)
		}},
		{"hosted-file-missing", false, func(t *testing.T, root string) {
			os.Remove(filepath.Join(root, bm, "v1/files/replacements.safetensors"))
		}},
		{"hosted-file-unlisted", false, func(t *testing.T, root string) {
			os.WriteFile(filepath.Join(root, bm, "v1/files/extra.safetensors"), []byte("x"), 0o644)
		}},
		{"hosted-outside-files", false, func(t *testing.T, root string) {
			editJSON(t, filepath.Join(root, bm, "v1/version.json"), func(v map[string]any) {
				v["files"].([]any)[0].(map[string]any)["locators"].([]any)[0].(map[string]any)["path"] = "replacements.safetensors"
			})
		}},
		{"duplicate-file-name", false, func(t *testing.T, root string) {
			editJSON(t, filepath.Join(root, sa, "v2/version.json"), func(v map[string]any) {
				v["files"].([]any)[1].(map[string]any)["name"] = "adapter_model.safetensors"
			})
		}},
		{"derived-from-foreign-digest", false, func(t *testing.T, root string) {
			editJSON(t, filepath.Join(root, bm, "v1/version.json"), func(v map[string]any) {
				v["files"].([]any)[1].(map[string]any)["derived_from"] = []any{strings.Repeat("9", 64)}
			})
		}},
		{"replacement-gguf-source", false, func(t *testing.T, root string) {
			editJSON(t, filepath.Join(root, bm, "v1/version.json"), func(v map[string]any) {
				v["files"].([]any)[0].(map[string]any)["name"] = "replacements.gguf"
			})
		}},
		{"lora-without-config", false, func(t *testing.T, root string) {
			editJSON(t, filepath.Join(root, sa, "v2/version.json"), func(v map[string]any) {
				v["files"].([]any)[1].(map[string]any)["name"] = "config.json"
			})
		}},
		{"comparison-unknown-run", false, func(t *testing.T, root string) {
			editJSON(t, filepath.Join(root, bm, "v1/comparisons/bf16-accuracy.json"), func(v map[string]any) { v["patched_run"] = "nope" })
		}},
		{"comparison-unknown-baseline", false, func(t *testing.T, root string) {
			editJSON(t, filepath.Join(root, bm, "v1/comparisons/bf16-accuracy.json"), func(v map[string]any) { v["baseline_run"] = "nope" })
		}},
		{"assessment-unknown-comparison", false, func(t *testing.T, root string) {
			cmp := readJSON(t, filepath.Join(root, bm, "v1/comparisons/bf16-accuracy.json"))
			editJSON(t, filepath.Join(root, bm, "v1/assessments/arithmetic-help.json"), func(v map[string]any) {
				v["decision"] = "SUPPORTED"
				delete(v, "owner")
				v["policy"] = cmp["matched_outcomes"]
				v["comparisons"] = []any{"nope"}
			})
		}},
		{"sparse-without-old", false, func(t *testing.T, root string) { rehost(t, root, sparse([]int64{3, 17}, false)) }},
		{"sparse-unsorted", false, func(t *testing.T, root string) { rehost(t, root, sparse([]int64{17, 3}, true)) }},
		{"sparse-duplicate-index", false, func(t *testing.T, root string) { rehost(t, root, sparse([]int64{3, 3}, true)) }},
		{"sparse-unknown-suffix", false, func(t *testing.T, root string) {
			rehost(t, root, buildSafetensors([]stEntry{{"w#delta", "BF16", []int64{1}, make([]byte, 2)}}))
		}},
		{"full-and-sparse", false, func(t *testing.T, root string) {
			const w = "model.layers.0.mlp.down_proj.weight"
			rehost(t, root, buildSafetensors([]stEntry{
				{w, "BF16", []int64{1}, make([]byte, 2)},
				{w + "#idx", "I64", []int64{1}, i64s(0)},
				{w + "#new", "BF16", []int64{1}, make([]byte, 2)},
				{w + "#old", "BF16", []int64{1}, make([]byte, 2)},
			}))
		}},
		{"not-safetensors", false, func(t *testing.T, root string) { rehost(t, root, []byte("not a tensor file")) }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			root := filepath.Join(t.TempDir(), "models")
			if err := os.CopyFS(root, os.DirFS("../examples/models")); err != nil {
				t.Fatal(err)
			}
			tc.edit(t, root)
			errs := checkModels(c, root)
			if (len(errs) == 0) != tc.ok {
				t.Fatalf("accepted=%v, want=%v: %v", len(errs) == 0, tc.ok, errs)
			}
		})
	}
}

func TestLookupControls(t *testing.T) {
	c := newCheck(t)
	sum := strings.Repeat("ab", 32)
	good := `{"contract_version": 1, "matches": [{"base": "a/b", "profile": "bf16"}]}`
	for _, tc := range []struct {
		path, body string
		ok         bool
	}{
		{"by-digest/ab/" + sum + ".json", good, true},
		{"by-digest/cd/" + sum + ".json", good, false},
		{"by-digest/" + sum + ".json", good, false},
		{"by-digest/ab/" + sum + ".json", `{"contract_version": 1, "matches": []}`, false},
		{"by-repo/a/b.json", `{"contract_version": 1, "matches": [{"base": "a/b", "profiles": ["bf16"]}]}`, true},
		{"by-repo/a.json", `{"contract_version": 1, "matches": [{"base": "a/b", "profiles": ["bf16"]}]}`, false},
	} {
		root := t.TempDir()
		p := filepath.Join(root, filepath.FromSlash(tc.path))
		os.MkdirAll(filepath.Dir(p), 0o755)
		os.WriteFile(p, []byte(tc.body), 0o644)
		if errs := checkLookups(c, root); (len(errs) == 0) != tc.ok {
			t.Errorf("%s: accepted=%v, want=%v: %v", tc.path, len(errs) == 0, tc.ok, errs)
		}
	}
}
