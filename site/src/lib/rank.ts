// Rankings show recorded scores only. A ranking holds runs that share one
// group (profile, protocol, dataset and settings) and one metric and unit.
// Only COMPLETE runs with a numeric score are ranked. No difference,
// interval or estimate is computed; combinations are never derived.
import { rankable, runLabel, type Group, type Run } from "./index.ts";

export interface Ranking {
  metric: string;
  unit: string;
  /** Highest recorded score first; ties keep record order. */
  runs: Run[];
}

export function rankings(g: Group): Ranking[] {
  const by = new Map<string, Ranking>();
  for (const run of g.runs) {
    if (!rankable(run)) continue;
    const s = run.record.score;
    const key = `${s.metric}|${s.unit}`;
    let r = by.get(key);
    if (!r) by.set(key, (r = { metric: s.metric!, unit: s.unit!, runs: [] }));
    r.runs.push(run);
  }
  const out = [...by.values()];
  for (const r of out) r.runs.sort((a, b) => b.record.score.value! - a.record.score.value!);
  return out;
}

export function barLabel(run: Run): string {
  if (!run.version) return "unpatched base";
  const v = run.version;
  return `${v.patch.slug} ${v.label}${v.status === "active" ? "" : ` (${v.status})`}`;
}

export { runLabel };
