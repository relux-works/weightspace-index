// Rankings show recorded scores only. A ranking holds runs that share one
// group (profile, protocol, dataset and settings), one metric and unit, and
// one measured-by bucket (rule 9: index-performed and reported results stay
// apart). Only COMPLETE runs with a numeric score are ranked. No difference,
// interval or estimate is computed; combinations are never derived.
import { measuredBy, rankable, runLabel, type Group, type Run } from "./index.ts";

export interface Ranking {
  bucket: "index" | "reported";
  metric: string;
  unit: string;
  /** Highest recorded score first; ties keep record order. */
  runs: Run[];
  /** Active versions sharing the highest recorded score among patched runs. */
  best: Run[];
}

export const BUCKET_LABEL = {
  index: "Measured by the index",
  reported: "Reported by authors or third parties",
} as const;

export function rankings(g: Group): Ranking[] {
  const by = new Map<string, Ranking>();
  for (const run of g.runs) {
    if (!rankable(run)) continue;
    const s = run.record.score;
    const bucket = measuredBy(run);
    const key = `${bucket}|${s.metric}|${s.unit}`;
    let r = by.get(key);
    if (!r) by.set(key, (r = { bucket, metric: s.metric!, unit: s.unit!, runs: [], best: [] }));
    r.runs.push(run);
  }
  const out = [...by.values()].sort((a, b) => (a.bucket === b.bucket ? 0 : a.bucket === "index" ? -1 : 1));
  for (const r of out) {
    r.runs.sort((a, b) => b.record.score.value! - a.record.score.value!);
    const patched = r.runs.filter((x) => x.version && x.version.status === "active");
    if (patched.length) {
      const top = patched[0].record.score.value!;
      r.best = patched.filter((x) => x.record.score.value === top);
    }
  }
  return out;
}

export function barLabel(run: Run): string {
  if (!run.version) return "unpatched base";
  const v = run.version;
  return `${v.patch.slug} ${v.label}${v.status === "active" ? "" : ` (${v.status})`}`;
}

export { runLabel };
