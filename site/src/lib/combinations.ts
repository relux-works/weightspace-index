import { fail, listDir, readRecord } from "./files.ts";
import type { Base, Version } from "./types.ts";

// A combination study is one reported benchmark of a stack: two or more exact
// patch versions applied together, each at a recorded scale. Any stack is
// valid; the site shows only stacks someone measured, as extra rows of the
// model page's per-benchmark comparison. A study may also file each member's
// score alone under the same setup; those become that patch's row when the
// patch has no result of its own for the same benchmark.

type Member = { slug: string; version: number; artifact: string; scale: number; alone?: number };
type Direction = "higher" | "lower";
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
