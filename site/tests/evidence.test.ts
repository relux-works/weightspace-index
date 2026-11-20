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

test("the preview favors independent evidence even if its result is flat or negative", () => {
  const author = { source: "author", base: 2, patched: 9 } as Result;
  for (const patched of [1, 2]) {
    const measured = { source: "index", base: 2, patched } as Result;
    assert.equal(previewResult([author, measured]), measured);
  }
  assert.equal(previewResult([author]), author);
  assert.equal(previewResult([]), undefined);
});

