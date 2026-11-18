import { fail, listDir, readRecord } from "./files.ts";
import type { Base, Version } from "./types.ts";

type Member = { slug: string; version: number; artifact: string; scale: number; alone: number };
interface Study {
  contract_version: 1;
  profile: string;
  benchmark: string;
  metric: string;
  unit: string;
  denominator: number;
  direction: "higher" | "lower";
  measured_by: "index" | "third-party" | "author";
  protocol: string;
  baseline: number;
  together: number;
  members: Member[];
}

export function validateStudy(input: unknown, versions: Version[]) {
  const s = input as Study;
  const invalid = () => fail("combination study needs one measured score per treatment and exact artifacts on a shared profile");
  if (!s || s.contract_version !== 1 || ![s.benchmark, s.metric, s.unit, s.protocol, s.profile].every(x => typeof x === "string" && x.trim()) ||
      !["index", "third-party", "author"].includes(s.measured_by) ||
      !["higher", "lower"].includes(s.direction) || !Number.isSafeInteger(s.denominator) || s.denominator < 1 || !Number.isFinite(s.together) || !Number.isFinite(s.baseline) ||
      !Array.isArray(s.members) || s.members.length < 2) invalid();
  const seen = new Set<string>();
  const members = s.members.map(m => {
    const release = versions.find(v => v.n === m?.version && v.patch.slug === m.slug && v.profile.id === s.profile);
    const artifact = release?.files.find(f => f.name === m.artifact && /\.(safetensors|gguf)$/i.test(f.name));
    if (!release || !artifact || seen.has(m.slug) || !Number.isFinite(m.scale) || !Number.isFinite(m.alone)) return invalid();
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


export const pairKey = (a: string, b: string) => [a, b].sort().join("|");

/** The overview matrix describes pairs of currently downloadable releases. */
export function pairStudies(studies: { members: { slug: string; release: Pick<Version, "isLatest" | "status"> }[] }[]) {
  const pairs = new Map<string, number[]>();
  studies.forEach(({ members }, index) => {
    if (members.length !== 2 || !members.every(m => m.release.isLatest && m.release.status === "active")) return;
    const key = pairKey(members[0].slug, members[1].slug);
    const hits = pairs.get(key) ?? [];
    hits.push(index);
    pairs.set(key, hits);
  });
  return pairs;
}

export function bestSingle(alone: number[], direction: Study["direction"]) {
  return direction === "higher" ? Math.max(...alone) : Math.min(...alone);
}

export function jointOutcome(joint: number, alone: number[], direction: Study["direction"]) {
  const best = bestSingle(alone, direction);
  if (best === joint) return "Matches the best single patch";
  if (direction === "higher" ? joint < best : joint > best) return "Below the best single patch";
  return alone.length === 2 ? "Better than either patch alone" : "Better than each patch alone";
}
