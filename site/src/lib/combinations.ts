import { fail, listDir, readRecord } from "./files.ts";
import type { Base, Version } from "./types.ts";

// A combination study is one reported benchmark of a stack: two or more exact
// patch versions applied together, each at a recorded scale. Any stack is
// valid; the site shows only stacks someone measured. A study that also files
// each member alone (same setup) can say whether the patches interact; one with
// only the joint score can only say how the stack compares with the base.

type Member = { slug: string; version: number; artifact: string; scale: number; alone?: number };
export type Direction = "higher" | "lower";
interface Study {
  contract_version: 1;
  profile: string;
  benchmark: string;
  metric: string;
  unit: string;
  denominator: number;
  direction: Direction;
  measured_by: "index" | "third-party" | "author";
  protocol: string;
  baseline: number;
  together: number;
  members: Member[];
}

export function validateStudy(input: unknown, versions: Version[]) {
  const s = input as Study;
  const invalid = () => fail("combination study needs a measured base and joint score, exact artifacts on a shared profile, and either every member's solo score or none");
  if (!s || s.contract_version !== 1 || ![s.benchmark, s.metric, s.unit, s.protocol, s.profile].every(x => typeof x === "string" && x.trim()) ||
      !["index", "third-party", "author"].includes(s.measured_by) ||
      !["higher", "lower"].includes(s.direction) || !Number.isSafeInteger(s.denominator) || s.denominator < 1 || !Number.isFinite(s.together) || !Number.isFinite(s.baseline) ||
      !Array.isArray(s.members) || s.members.length < 2) invalid();
  const solo = s.members.filter(m => m?.alone !== undefined).length;
  if (solo !== 0 && solo !== s.members.length) invalid();
  const seen = new Set<string>();
  const members = s.members.map(m => {
    const release = versions.find(v => v.n === m?.version && v.patch.slug === m.slug && v.profile.id === s.profile);
    const artifact = release?.files.find(f => f.name === m.artifact && /\.(safetensors|gguf)$/i.test(f.name));
    if (!release || !artifact || seen.has(m.slug) || !Number.isFinite(m.scale) || (solo && !Number.isFinite(m.alone))) return invalid();
    seen.add(m.slug);
    return { ...m, release, artifact };
  });
  return { ...s, members };
}

export function combinationStudies(base: Base, version?: Version) {
  const root = base.dir.slice(0, -(`models/${base.address}`.length));
  const dir = `${root}bench/combinations/${base.address}`;
  return listDir(dir).filter(name => name.endsWith(".json")).map(name => {
    const file = `${dir}/${name}`;
    return { ...validateStudy(readRecord(file), base.patches.flatMap(p => p.versions)), file };
  }).filter(study => !version || study.members.some(m => m.release === version));
}

/** Order-free key of a stack: its patch slugs, sorted. */
export const stackKey = (slugs: string[]) => [...slugs].sort().join("|");

// ── Reading one study ────────────────────────────────────────────────────────

/**
 * A study is read two ways. The headline answers the practitioner's question:
 * is the stack better than just applying its best patch alone? With solo
 * scores that is `best`; without them only the base is available (`base`).
 * The second reading (`sum`) asks whether the patches interact: does the stack
 * land where each patch adding its own change would put it?
 *
 *   best: above / level / below the best single patch (level = inside the noise)
 *   base: above / level / below the unpatched base
 *   sum:  help (beyond the sum) / adds (inside the noise) / clash (short of it) /
 *         unclear (the noise is wider than either patch's own effect)
 * A metric with no noise model gets `level` only on an exact tie.
 */
export type Standing = "above" | "level" | "below";
export type SumVerdict = "help" | "adds" | "clash" | "unclear";
export const SUM_LABEL: Record<SumVerdict, string> = {
  help: "Help each other",
  adds: "Add up",
  clash: "Get in each other's way",
  unclear: "Too close to call",
};

/** 95% half-width of a percentage measured on n items (binomial, normal approximation). */
export function pctNoise(p: number, n: number) {
  const q = Math.min(Math.max(p / 100, 0), 1);
  return 1.96 * 100 * Math.sqrt(q * (1 - q) / n);
}

/** A comparison of the stack with one reference, `gap` signed so that positive is better. */
export interface Comparison { ref: number; gap: number; noise: number | null }

export interface Reading {
  /** +1 when higher scores are better, −1 when lower ones are. */
  sign: 1 | -1;
  /** Changes from the base, in the metric's units. */
  alone: number[] | null;
  together: number;
  /** The headline comparison: with the best single patch, or with the base when no solo scores were filed. */
  lead: Comparison & { against: "best" | "base"; member: number | null; standing: Standing };
  /** With the sum of the solo changes; null without solo scores. */
  sum: (Comparison & { verdict: SumVerdict }) | null;
}

export function readStudy(s: { baseline: number; together: number; members: { alone?: number }[]; direction: Direction; unit: string; denominator: number }): Reading {
  const sign = s.direction === "higher" ? 1 : -1;
  const together = s.together - s.baseline;
  const solo = s.members.every(m => m.alone !== undefined) ? s.members.map(m => m.alone!) : null;
  const alone = solo && solo.map(a => a - s.baseline);
  // Independent-sample approximation; scoring every treatment on the same items
  // usually makes the real range narrower.
  const sd = (p: number) => s.unit === "%" ? pctNoise(p, s.denominator) / 1.96 : NaN;
  const range = (variance: number) => Number.isFinite(variance) ? 1.96 * Math.sqrt(variance) : null;
  const inside = (gap: number, noise: number | null) => noise === null ? gap === 0 : Math.abs(gap) <= noise;
  const standing = (gap: number, noise: number | null): Standing => inside(gap, noise) ? "level" : gap > 0 ? "above" : "below";

  if (!solo || !alone) {
    const noise = range(sd(s.together) ** 2 + sd(s.baseline) ** 2);
    const gap = sign * together;
    return { sign, alone: null, together, sum: null, lead: { against: "base", member: null, ref: 0, gap, noise, standing: standing(gap, noise) } };
  }
  const member = alone.reduce((b, d, i) => sign * d > sign * alone[b] ? i : b, 0);
  const bestNoise = range(sd(s.together) ** 2 + sd(solo[member]) ** 2);
  const bestGap = sign * (together - alone[member]);
  const expected = alone.reduce((sum, d) => sum + d, 0);
  const sumNoise = range(sd(s.together) ** 2 + solo.reduce((v, a) => v + sd(a) ** 2, 0) + ((solo.length - 1) * sd(s.baseline)) ** 2);
  const sumGap = sign * (together - expected);
  const effect = Math.max(...alone.map(Math.abs));
  const verdict: SumVerdict = inside(sumGap, sumNoise) ? (sumNoise !== null && sumNoise > effect ? "unclear" : "adds") : sumGap > 0 ? "help" : "clash";
  return {
    sign, alone, together,
    lead: { against: "best", member, ref: alone[member], gap: bestGap, noise: bestNoise, standing: standing(bestGap, bestNoise) },
    sum: { ref: expected, gap: sumGap, noise: sumNoise, verdict },
  };
}
