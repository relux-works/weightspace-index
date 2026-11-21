import { readCoverage } from "./catalog.ts";
import type { Run } from "./types.ts";

const MIN_COVERAGE = 0.25;

function summarize(outcomes: NonNullable<Run["outcomes"]>) {
  const cells = new Set<string>();
  const items = new Map<number, { correct: number; seeds: number }>();
  let cuts = 0;
  let hasFinishReasons = true;
  for (const { id, seed, correct, finish_reason } of outcomes) {
    if (typeof id !== "number" || !Number.isSafeInteger(id) || typeof seed !== "number" || !Number.isSafeInteger(seed) ||
        typeof correct !== "boolean") return;
    const cell = `${id}|${seed}`;
    if (cells.has(cell)) return;
    cells.add(cell);

    const item = items.get(id) ?? { correct: 0, seeds: 0 };
    item.correct += Number(correct);
    item.seeds++;
    items.set(id, item);
    if (finish_reason === "length") cuts++;
    if (typeof finish_reason !== "string") hasFinishReasons = false;
  }

  // Match bench/runs.py by weighting each item equally after averaging its filed seeds.
  const sum = [...items.values()].reduce((total, item) => total + item.correct / item.seeds, 0);
  return {
    cells,
    score: {
      value: sum / items.size,
      denominator: items.size,
      cut_at_limit: hasFinishReasons ? cuts / cells.size : undefined,
    },
  };
}

/** Derive a score for display from matched, already-graded outcomes. */
export function provisionalBenchScores(base: Run, patched: Run) {
  if (base.record.state !== "INCOMPLETE" || patched.record.state !== "INCOMPLETE") return;
  const coverage = readCoverage(patched.record.score.reason);
  const baseCoverage = readCoverage(base.record.score.reason);
  if (!coverage || !baseCoverage || coverage.filed / coverage.total < MIN_COVERAGE || coverage.filed === coverage.total ||
      coverage.filed !== baseCoverage.filed || coverage.total !== baseCoverage.total) return;
  if (base.outcomes?.length !== coverage.filed || patched.outcomes?.length !== coverage.filed) return;

  const before = summarize(base.outcomes);
  const after = summarize(patched.outcomes);
  if (!before || !after || [...before.cells].some(cell => !after.cells.has(cell))) return;
  return { coverage, base: before.score, patched: after.score };
}
