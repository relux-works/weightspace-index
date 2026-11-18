import assert from "node:assert/strict";
import { test } from "node:test";
import { scoreAxis } from "../src/lib/score-axis.ts";

test("percentage comparisons retain a zero to one hundred scale", () => {
  const axis = scoreAxis([7, 49], "%");
  assert.equal(axis.position(7), 7);
  assert.equal(axis.position(49), 49);
});

test("signed scores share a zero origin and identical scores keep a finite scale", () => {
  const axis = scoreAxis([-10, 0, 10], "points");
  assert.equal(axis.position(-10), 0);
  assert.equal(axis.zero, 50);
  assert.equal(axis.position(10), 100);
  assert.equal(scoreAxis([0, 0], "points").position(0), 0);
});
