import assert from "node:assert/strict";
import { test } from "node:test";
import { CUT, directoryBenchmarkHighlight, FINAL_ANSWERS, groupEvidence, pending, previewResult, resultKey, results, type Result, type Pending } from "../src/lib/insight.ts";
import type { Run, Version } from "../src/lib/types.ts";

test("author claims stay apart from index results, and every filed run is an index run", () => {
  const author = { source: "author" } as Result;
  const index = { source: "index" } as Result;
  const waiting = { source: "author" } as Pending;
  const run = { record: {} } as Run;
  const groups = groupEvidence([author, index], [waiting], [run]);
  assert.deepEqual(groups.map(({source, scores, progress, runs}) => [source, scores, progress, runs]), [
    ["author", [author], [waiting], []], ["index", [index], [], [run]],
  ]);
});

test("the preview favors Measured evidence even if its result is flat or negative", () => {
  const author = { source: "author", base: 2, patched: 9 } as Result;
  for (const patched of [1, 2]) {
    const measured = { source: "index", base: 2, patched } as Result;
    assert.equal(previewResult([author, measured]), measured);
  }
  assert.equal(previewResult([author]), author);
  assert.equal(previewResult([]), undefined);
});

test("a bench run reads as final answers right and its cut share, in percent, against its paired baseline", () => {
  const protocol = { name: "bench/protocols/math500-n197.json", sha256: "a".repeat(64) };
  const score = (value: number, cut: number) => ({ metric: "math_verify", value, value_lm_eval: value + 0.1, cut_at_limit: cut, unit: "mean over items", denominator: 197 });
  const base = { url: "/base/", group: "g", record: { state: "COMPLETE", protocol, score: score(0.2, 0.8) } } as unknown as Run;
  const run = { url: "/run/", group: "g", paired: base, protocolLabel: "math500-n197", record: { state: "COMPLETE", protocol, score: score(0.273, 0.645) } } as unknown as Run;
  const v = { claims: [], comparisons: [], runs: [run] } as unknown as Parameters<typeof results>[0];
  const [head, cut] = results(v);
  assert.deepEqual([head.benchmark, head.metric, head.unit, head.base, head.patched, head.higherIsBetter], ["MATH-500", FINAL_ANSWERS, "%", 20, 27.3, true]);
  assert.deepEqual([cut.name, cut.unit, cut.base, cut.patched, cut.higherIsBetter], [CUT, "%", 80, 64.5, false]);
  assert.deepEqual(results(v, { includeProvisional: true }), [head, cut]);
});

test("an IFEval bench run labels the strict prompt-level metric", () => {
  const protocol = { name: "bench/protocols/ifeval.json", sha256: "a".repeat(64) };
  const record = (value: number) => ({ state: "COMPLETE", protocol, score: {
    metric: "prompt_level_strict_acc", value, unit: "strict prompt-level accuracy", denominator: 541,
  } });
  const base = { url: "/base/", group: "g", record: record(0.2) } as unknown as Run;
  const run = { url: "/run/", group: "g", paired: base, protocolLabel: "ifeval", record: record(0.5) } as unknown as Run;
  const v = { claims: [], comparisons: [], runs: [run] } as unknown as Parameters<typeof results>[0];
  const [score] = results(v);
  assert.deepEqual([score.benchmark, score.metric, score.unit, score.base, score.patched],
    ["IFEval", "strict prompt-level accuracy", "%", 20, 50]);
});

function incompleteBench(
  total: number,
  baseOutcomes: Run["outcomes"],
  patchedOutcomes: Run["outcomes"],
  protocolLabel = "math500-n197",
) {
  const protocol = { name: `bench/protocols/${protocolLabel}.json`, sha256: "a".repeat(64) };
  const record = { state: "INCOMPLETE", protocol, score: {
    state: "UNKNOWN", reason: `3 of ${total} item-seed pairs filed`,
  } };
  const group = {};
  const base = { url: "/base/", version: null, group, record, outcomes: baseOutcomes } as unknown as Run;
  const run = { url: "/run/", group, paired: base, protocolLabel, record, outcomes: patchedOutcomes } as unknown as Run;
  return { claims: [], comparisons: [], runs: [run] } as unknown as Version;
}

const outcome = (id: number, seed: number, correct: boolean, finish_reason = "stop") =>
  ({ id, seed, correct, finish_reason, score: 1 });
const baseOutcomes = [outcome(1, 1, true, "length"), outcome(1, 2, true, "length"), outcome(2, 1, false)];
const patchedOutcomes = [outcome(2, 1, true), outcome(1, 2, true), outcome(1, 1, false, "length")];

test("a quarter of filed pairs unlocks provisional scores, averaging seeds per item", () => {
  const v = incompleteBench(12, baseOutcomes, patchedOutcomes);
  const [head, cut] = results(v, { includeProvisional: true });
  assert.ok(head && cut);
  assert.deepEqual([head.metric, head.base, head.patched, head.n, head.provisional],
    [FINAL_ANSWERS, 50, 75, 2, { filed: 3, total: 12 }]);
  assert.deepEqual([cut.name, cut.base, cut.patched, cut.provisional], [CUT, 66.7, 33.3, head.provisional]);
  assert.deepEqual(results(v), []);
});

test("coverage below a quarter shows progress without a provisional score", () => {
  const v = incompleteBench(13, baseOutcomes, patchedOutcomes);
  assert.deepEqual(results(v, { includeProvisional: true }), []);
  assert.equal(pending(v)[0]?.filed, 3);
});

test("provisional scores require the same unique, graded item–seed pairs on both sides", () => {
  const [first, second, third] = patchedOutcomes;
  for (const outcomes of [null, [first, second], [first, second, second], [first, second, { ...third, seed: 3 }],
      [first, second, { ...third, correct: undefined }]]) {
    const v = incompleteBench(12, baseOutcomes, outcomes);
    assert.deepEqual(results(v, { includeProvisional: true }), []);
    assert.equal(pending(v)[0]?.filed, 3);
  }
});

test("provisional IFEval uses the protocol's strict prompt-level metric", () => {
  const v = incompleteBench(12, baseOutcomes, patchedOutcomes, "ifeval");
  const [head] = results(v, { includeProvisional: true });
  assert.ok(head);
  assert.equal(head.metric, "strict prompt-level accuracy");
});

test("benchmarks with the same measure name have distinct chart keys", () => {
  const math = { benchmark: "MATH-500", name: CUT, unit: "%" };
  assert.notEqual(resultKey(math), resultKey({ ...math, benchmark: "IFEval" }));
});

test("directory benchmark highlight filters to preapproved primary scores and prefers Measured runs", () => {
  const mkVersion = (measures: object[], runs: Run[] = []) => ({
    label: "v1",
    claims: [{ record: { source: { url: "https://example.com" }, measures } }],
    runs,
    comparisons: [],
  }) as unknown as Version;

  // Auxiliary metrics like "Cut at the token limit" must never win over primary MATH-500 accuracy.
  const qwen08 = mkVersion([
    { benchmark: "MATH-500", metric: "accuracy", unit: "%", baseline: 7.9, patched: 48.1, better: "higher" },
    { benchmark: "MATH-500", name: CUT, unit: "%", baseline: 93.0, patched: 33.1, better: "lower" },
  ]);
  const hit = directoryBenchmarkHighlight([qwen08]);
  assert.equal(hit?.benchmark, "MATH-500");
  assert.equal(hit?.patched, 48.1);
  assert.equal(hit?.source, "author");

  // Obscure benchmarks return null instead of surfacing noisy pills.
  const obscure = mkVersion([
    { benchmark: "SmolDataEnvs (n=150)", metric: "pass@1", unit: "%", baseline: 62.0, patched: 78.0, better: "higher" },
    { benchmark: "SWE-smith next-action (n=40)", name: "Same tool", metric: "accuracy", unit: "%", baseline: 32.0, patched: 88.0, better: "higher" },
  ]);
  assert.equal(directoryBenchmarkHighlight([obscure]), null);

  // Well-known SWE-bench Pro beats obscure next-action claims on the same model.
  const swe = mkVersion([
    { benchmark: "SWE-bench Pro (hard 51)", metric: "pass@1", unit: "%", baseline: 39.2, patched: 64.7, better: "higher" },
  ]);
  assert.equal(directoryBenchmarkHighlight([obscure, swe])?.benchmark, "SWE-bench Pro (hard 51)");
});
