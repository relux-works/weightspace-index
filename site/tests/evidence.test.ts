import assert from "node:assert/strict";
import { test } from "node:test";
import { groupEvidence, previewResult, type Result, type Pending } from "../src/lib/insight.ts";
import { bestSingle } from "../src/lib/combinations.ts";
import type { Run } from "../src/lib/types.ts";

test("source groups keep author progress and unpaired contributor runs with their own measurements", () => {
  const author = { source: "author" } as Result;
  const index = { source: "index" } as Result;
  const waiting = { source: "author" } as Pending;
  const run = { record: { performed_by: "third-party" } } as Run;
  const groups = groupEvidence([author, index], [waiting], [run]);
  assert.deepEqual(groups.map(({source, scores, progress, runs}) => [source, scores, progress, runs]), [
    ["author", [author], [waiting], []], ["index", [index], [], []], ["third-party", [], [], [run]],
  ]);
});

test("the preview favors independent evidence even if its result is flat or negative", () => {
  const author = { source: "author", base: 2, patched: 9 } as Result;
  for (const patched of [1, 2]) {
    const measured = { source: "third-party", base: 2, patched } as Result;
    assert.equal(previewResult([author, measured]), measured);
  }
  assert.equal(previewResult([author]), author);
  assert.equal(previewResult([]), undefined);
});

test("the best single comparison for a combination follows the metric direction", () => {
  assert.equal(bestSingle([7, 12], "lower"), 7);
  assert.equal(bestSingle([7, 12], "higher"), 12);
});

test("the matrix keeps every two-way study of the active releases under the same pair", async () => {
  const { pairStudies } = await import("../src/lib/combinations.ts");
  const member = (slug: string, isLatest = true) => ({ slug, release: { isLatest, status: "active" as const } });
  const a = member("a"), b = member("b");
  const studies = [
    { members: [a, b] }, { members: [b, a] },
    { members: [a, member("b", false)] }, { members: [a, b, member("c")] },
  ];
  assert.deepEqual(pairStudies(studies).get("a|b"), [0, 1]);
});
