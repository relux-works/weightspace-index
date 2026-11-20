import assert from "node:assert/strict";
import { test } from "node:test";
import { groupEvidence, previewResult, type Result, type Pending } from "../src/lib/insight.ts";
import type { Run } from "../src/lib/types.ts";

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


test("a bench run reads as final answers right and its cut share, in percent, against its paired baseline", async () => {
  const { results, FINAL_ANSWERS, CUT } = await import("../src/lib/insight.ts");
  const protocol = { name: "bench/protocols/math500-n197.json", sha256: "a".repeat(64) };
  const score = (value: number, cut: number) => ({ metric: "math_verify", value, value_lm_eval: value + 0.1, cut_at_limit: cut, unit: "mean over items", denominator: 197 });
  const base = { url: "/base/", group: "g", record: { state: "COMPLETE", protocol, score: score(0.2, 0.8) } } as unknown as Run;
  const run = { url: "/run/", group: "g", paired: base, protocolLabel: "math500-n197", record: { state: "COMPLETE", protocol, score: score(0.273, 0.645) } } as unknown as Run;
  const v = { claims: [], comparisons: [], runs: [run] } as unknown as Parameters<typeof results>[0];
  const [head, cut] = results(v);
  assert.deepEqual([head.benchmark, head.metric, head.unit, head.base, head.patched, head.higherIsBetter], ["MATH-500", FINAL_ANSWERS, "%", 20, 27.3, true]);
  assert.deepEqual([cut.name, cut.unit, cut.base, cut.patched, cut.higherIsBetter], [CUT, "%", 80, 64.5, false]);
});
