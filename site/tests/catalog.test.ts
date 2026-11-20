import assert from "node:assert/strict";
import { test } from "node:test";
import { activeVersion, discoveryText, readCoverage, columnBins } from "../src/lib/catalog.ts";
import { validateStudy } from "../src/lib/combinations.ts";
import type { Patch, Run, Version } from "../src/lib/index.ts";

test("a withdrawn release cannot become a default download", () => {
  const versions = [{ n: 1, status: "active" }, { n: 3, status: "withdrawn" }];
  assert.equal(activeVersion({ latest: 1, versions } as Patch), versions[0]);
  assert.equal(activeVersion({ latest: 3, versions } as Patch), undefined);
  assert.equal(activeVersion({ latest: null, versions } as Patch), undefined);
});

test("catalog search covers the model, authors, goals and available formats", () => {
  const patch = { title: "Quick sums", summary: "Checks arithmetic", authors: [{ name: "Jane" }], tags: ["math"], slug: "sums" };
  const text = discoveryText("Acme/Mini", [patch as Patch], ["LoRA"]);
  for (const word of ["acme/mini", "jane", "arithmetic", "math", "lora"]) assert.ok(text.includes(word));
});

test("coverage accepts a filed item count without treating it as a score", () => {
  assert.deepEqual(readCoverage("2 of 197 item-seed pairs filed"), { filed: 2, total: 197 });
  assert.equal(readCoverage("in progress"), null);
  assert.equal(readCoverage("3 of 2 item-seed pairs filed"), null);
});

test("column density counts all active columns across the full tensor width", () => {
  assert.deepEqual(columnBins([0, 2, 7], 8, 4), [1, 1, 0, 1]);
});

const versions = ["alpha", "beta"].map(slug => ({ n: 1, patch: { slug }, profile: { id: "bf16" }, files: [{role:"source",sha256:"a".repeat(64), name:"adapter_model.safetensors"}] })) as Version[];
const study = {
  contract_version: 1, profile: "bf16", benchmark: "Arithmetic", metric: "Accuracy", unit: "%", direction: "higher", denominator: 100,
  protocol: "Identical 100 prompts, greedy decoding.", baseline: 40, together: 48,
  members: ["alpha", "beta"].map(slug => ({ slug, version: 1, artifact: "adapter_model.safetensors", scale: 1, alone: 45 })),
};
test("a combination needs a joint result for exact versions on one profile, with every solo score or none", () => {
  assert.equal(validateStudy(study, versions).members.length, 2);
  const noSolo = { ...study, members: study.members.map(({ alone: _, ...m }) => m) };
  assert.equal(validateStudy(noSolo, versions).members.length, 2);
  for (const invalid of [
    { ...study, together: null }, { ...study, profile: "q4" }, { ...study, members: [study.members[0], study.members[0]] },
    { ...study, members: [study.members[0], { ...study.members[1], artifact: "unknown" }] },
    { ...study, members: [study.members[0], noSolo.members[1]] }
  ]) assert.throws(() => validateStudy(invalid, versions));
});

test("a large patch list shows a short page and searches the entire catalog", async () => {
  const { filteredPage } = await import("../src/lib/filter.ts");
  const rows = Array.from({ length: 90 }, (_, i) => "patch number " + i);
  assert.equal(filteredPage(rows, "", 8).visible.length, 8);
  assert.equal(filteredPage(rows, "", 8).matched, 90);
  assert.deepEqual(filteredPage(rows, "89 PATCH", 8), { visible: [89], matched: 1 });
  assert.deepEqual(filteredPage(rows, "missing", 8), { visible: [], matched: 0 });
});

test("search terms cover every field a result shows and both formats", async () => {
  const { versionTerms } = await import("../src/lib/search.ts");
  const { matchesAll, queryWords } = await import("../src/lib/filter.ts");
  const v = {
    label: "v1", status: "active", evidence: ["INCOMPLETE run"], runs: [{ protocolLabel: "math500-n197" }],
    profile: { id: "bf16", source: { repo_id: "Acme/Mini", revision: "x" } },
    files: [{ role: "source", name: "replacements.safetensors", sha256: "a" }, { role: "derived", name: "adapter.safetensors", sha256: "b" }],
    patch: { base: { address: "Acme/Mini" }, kind: "replacement", title: "Quick sums", summary: "Checks arithmetic", slug: "sums",
      tags: ["math"], license: "Apache-2.0", authors: [{ name: "Jane" }], submitter: "sam" },
  } as unknown as Version;
  const terms = versionTerms(v);
  for (const q of ["lora", "replacement", "bf16", "math500", "jane quick", "apache-2.0", "acme/mini v1"]) {
    assert.ok(matchesAll(terms, queryWords(q)), q);
  }
  assert.ok(!matchesAll(terms, queryWords("gguf")));
  // Every searched field a result does not show is named, so a hit can say where it matched.
  const { hiddenFields } = await import("../src/lib/search.ts");
  const hidden = hiddenFields(v);
  assert.deepEqual(hidden.map(f => f.label), ["Authors", "Submitted by", "Evidence", "Benchmark protocol", "Patch kind", "Slug"]);
  assert.equal(hidden.find(f => f.label === "Evidence")!.text, "Run incomplete");
});

test("search shows only facets that can narrow the list", async () => {
  const { facetOptions } = await import("../src/lib/search.ts");
  const entry = (base: string, formats: string[]) => ({ base, profile: "bf16", formats, tags: [], license: "MIT", status: "active", benchmarks: [], evidence: [] });
  const one = facetOptions([entry("A/a", ["LoRA", "Replacement"])] as never);
  assert.deepEqual(one, []);
  const two = facetOptions([entry("A/a", ["LoRA", "Replacement"]), entry("B/b", ["LoRA"])] as never);
  assert.deepEqual(two.map(f => f.key), ["base", "format"]);
  assert.deepEqual(two.find(f => f.key === "format")!.options, [{ value: "LoRA", label: "LoRA", count: 2 }, { value: "Replacement", label: "Replacement", count: 1 }]);
});
