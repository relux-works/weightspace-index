import type { FileEntry, Profile, Run, Version, Index } from "./types.ts";
import { seg } from "./config.ts";

/** Evidence states for search and badges, in the wording of docs/structure.md "Evidence states". */
export const NO_BENCHMARKS = "no benchmarks submitted";

export function evidenceOf(v: Version): string[] {
  if (!v.runs.length && !v.comparisons.length && !v.assessments.length) return [NO_BENCHMARKS];
  const out = new Set<string>();
  for (const r of v.runs) {
    out.add(r.record.state === "COMPLETE" ? "complete run" : "INCOMPLETE run");
    if (r.record.score.state === "UNKNOWN") out.add("UNKNOWN score");
  }
  for (const c of v.comparisons) out.add(c.record.state === "REFUSED" ? "REFUSED comparison" : "MATCHED comparison");
  for (const a of v.assessments) out.add(`${a.record.decision} assessment`);
  return [...out];
}

export function allVersions(index: Index): Version[] {
  return index.bases.flatMap((b) => b.patches.flatMap((p) => p.versions));
}

export function allRuns(index: Index): Run[] {
  return index.bases.flatMap((b) => b.runs);
}

/** A run can be ranked only when it is COMPLETE and its score has a value. */
export function rankable(run: Run): boolean {
  return run.record.state === "COMPLETE" && typeof run.record.score.value === "number";
}

/** Rule 9: index-performed results and reported ones are never ranked together. */
export function measuredBy(run: Run) {
  return run.record.performed_by;
}

export function runLabel(run: Run): string {
  return run.version ? `${run.version.patch.slug} ${run.version.label}` : "unpatched base";
}

/** A profile's files as recorded in base.json, with Hugging Face links at the pinned revision. */
export function profileFiles(profile: Profile): FileEntry[] {
  return profile.files.map((f) => ({
    name: f.path,
    sha256: f.sha256,
    size_bytes: f.size_bytes,
    digest: "recorded" as const,
    links: [
      {
        host: "Hugging Face" as const,
        href: `https://huggingface.co/${profile.source.repo_id}/resolve/${profile.source.revision}/${seg(f.path)}`,
      },
    ],
    view: `https://huggingface.co/${profile.source.repo_id}/blob/${profile.source.revision}/${seg(f.path)}`,
  }));
}

/** Runs that name this run as their paired baseline (bench files the pairing on both sides). */
export function pairedFrom(run: Run): Run[] {
  return run.base.runs.filter((r) => r.paired === run);
}
