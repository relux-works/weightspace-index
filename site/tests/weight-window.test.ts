import assert from "node:assert/strict";
import test from "node:test";
import type { DeltaCell } from "../src/lib/insight.ts";
import { weightWindow } from "../src/lib/weight-window.ts";

const cell = (row: number, col: number): DeltaCell => ({ row, col, old: 0.25, new: 0.5 });

test("selects a contiguous window around the most sampled edits", () => {
  const first = cell(1, 1);
  const left = cell(1, 7);
  const right = cell(1, 9);
  const rightNext = cell(3, 9);
  const tensor = { cols: 10, columns: [1, 7, 9], cells: [[first, left, right], [null, null, rightNext]] };
  assert.deepEqual(weightWindow(tensor, 3), [[left, null, right], [null, null, rightNext]]);
});

test("retains all columns in a matrix narrower than the requested window", () => {
  const edit = cell(9, 1);
  assert.deepEqual(weightWindow({ cols: 2, columns: [1], cells: [[edit]] }, 4), [[null, edit]]);
});

test("skips matrices without an edited weight in the sampled rows", () => {
  assert.equal(weightWindow({ cols: 2, columns: [1], cells: [[null]] }, 2), null);
});
