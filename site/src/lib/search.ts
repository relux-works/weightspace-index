// Search entries, one per patch version, built from the records. The search
// page fetches them as search.json and filters in the browser. The homepage
// directory matches the same terms, so both searches agree.
import { formats, sourceLabels } from "./catalog.ts";
import { fmt, groupEvidence, pending, results } from "./insight.ts";
import { allVersions, type Index } from "./index.ts";
import type { Version } from "./types.ts";

export interface SearchEntry {
  url: string;
  base: string;
  profile: string;
  slug: string;
  title: string;
  summary: string;
  formats: string[];
  tags: string[];
  license: string;
  authors: string[];
  submitter: string;
  version: string;
  status: string;
  latest: boolean;
  benchmarks: string[];
  evidence: string[];
  /** How many versions the patch has, so "Latest" shows only where it tells something. */
  versionCount: number;
  /** One representative line per performer, as on the model page. */
  results: { source: string; text: string; value?: string }[];
  /** Lowercased text a result shows, so the page can tell when a hit came from elsewhere. */
  shown: string;
  /** Searchable fields a result does not show, named for the "Matched" line. */
  hidden: { label: string; text: string }[];
  /** Lowercased text the query words are matched against. */
  terms: string;
}

export const FACETS = [
  ["base", "Base"],
  ["profile", "Profile"],
  ["format", "Format"],
  ["tag", "Tag"],
  ["license", "License"],
  ["status", "Version status"],
  ["benchmark", "Benchmark"],
  ["evidence", "Evidence"],
] as const;

export type FacetKey = (typeof FACETS)[number][0];

/** Facets shown beside the search box; the rest sit behind "More filters". */
export const PRIMARY_FACETS: readonly FacetKey[] = ["base", "format"];

export interface FacetOption { value: string; label: string; count: number }

/**
 * Reader-facing wording for evidence states. The values stay as recorded
 * (docs/structure.md "Evidence states") so filters and shared URLs keep working.
 */
const EVIDENCE_LABELS: Record<string, string> = {
  "complete run": "Run complete",
  "INCOMPLETE run": "Run incomplete",
  "UNKNOWN score": "Score not established",
  "MATCHED comparison": "Matched comparison",
  "REFUSED comparison": "Comparison refused",
  "no benchmarks submitted": "No benchmarks submitted",
};
const ASSESSMENT_LABELS: Record<string, string> = {
  SUPPORTED: "Assessed: supported",
  REJECT: "Assessed: rejected",
  INCONCLUSIVE: "Assessed: inconclusive",
  UNSUPPORTED: "Assessed: unsupported",
  UNKNOWN: "Assessment not settled",
};

export function facetLabel(key: FacetKey, value: string): string {
  if (key === "evidence") {
    const assessment = /^(\w+) assessment$/.exec(value);
    return EVIDENCE_LABELS[value] ?? (assessment ? ASSESSMENT_LABELS[assessment[1]] : undefined) ?? value;
  }
  if (key === "status") return value.charAt(0).toUpperCase() + value.slice(1);
  return value;
}

/** The first score, else partial coverage, else a filed run, for each performer. */
export function resultLines(v: Version): SearchEntry["results"] {
  const label = sourceLabels;
  return groupEvidence(results(v), pending(v), v.runs).flatMap(({ source, scores, progress, runs }) => {
    const [s] = scores;
    if (s) return [{ source: label[source], text: `${s.benchmark} ${s.metric}`, value: `${fmt(s.base, s.unit)} → ${fmt(s.patched, s.unit)}` }];
    const [p] = progress;
    if (p) return [{ source: label[source], text: `${p.benchmark} score pending`, value: `${p.filed} of ${p.total} item–seed pairs` }];
    return runs.length ? [{ source: label[source], text: "Run filed, no matched score yet" }] : [];
  });
}

/**
 * Options per facet with how many versions carry each. A facet is left out when
 * no option could narrow the list: fewer than two values, or every value on
 * every version.
 */
export function facetOptions(entries: SearchEntry[]) {
  return FACETS.flatMap(([key, label]) => {
    const counts = new Map<string, number>();
    for (const e of entries) for (const v of new Set(facetValues(e, key))) counts.set(v, (counts.get(v) ?? 0) + 1);
    const options: FacetOption[] = [...counts].map(([value, count]) => ({ value, label: facetLabel(key, value), count })).sort((a, b) => a.label.localeCompare(b.label, "en", { sensitivity: "base" }));
    const narrows = options.some((o) => o.count < entries.length);
    return options.length < 2 || !narrows ? [] : [{ key, label, options, primary: PRIMARY_FACETS.includes(key) }];
  });
}

/** Everything a result shows about one version. */
export function versionTerms(v: Version): string {
  return [
    v.patch.base.address, v.profile.id, v.patch.kind, ...formats(v), v.patch.title, v.patch.summary, v.patch.slug,
    ...v.patch.tags, v.patch.license, ...v.patch.authors.map((a) => a.name), v.patch.submitter, v.label, v.status,
    ...v.runs.map((r) => r.protocolLabel), ...v.evidence, ...v.evidence.map((e) => facetLabel("evidence", e)),
  ].join(" ").toLowerCase();
}

/** Fields that are searched but not shown on a result, with a reader-facing label. */
export function hiddenFields(v: Version): SearchEntry["hidden"] {
  return [
    { label: "Authors", text: v.patch.authors.map((a) => a.name).join(", ") },
    { label: "Submitted by", text: v.patch.submitter },
    { label: "Evidence", text: v.evidence.map((e) => facetLabel("evidence", e)).join(", ") },
    { label: "Benchmark protocol", text: [...new Set(v.runs.map((r) => r.protocolLabel))].join(", ") },
    { label: "Patch kind", text: v.patch.kind },
    { label: "Slug", text: v.patch.slug },
  ].filter((f) => f.text);
}

export function searchEntries(index: Index): SearchEntry[] {
  return allVersions(index).map((v) => {
    const results = resultLines(v);
    const shown = [
      v.patch.title, v.patch.summary, v.label, v.status === "active" ? "" : v.status, v.patch.base.address, v.profile.id, ...formats(v), v.patch.license,
      ...v.patch.tags, ...results.flatMap((r) => [r.source, r.text, r.value ?? ""]),
    ].join(" ").toLowerCase();
    return {
      url: v.isLatest ? v.patch.url : v.url,
      base: v.patch.base.address,
      profile: v.profile.id,
      slug: v.patch.slug,
      title: v.patch.title,
      summary: v.patch.summary,
      formats: formats(v),
      tags: v.patch.tags,
      license: v.patch.license,
      authors: v.patch.authors.map((a) => a.name),
      submitter: v.patch.submitter,
      version: v.label,
      status: v.status,
      latest: v.isLatest,
      benchmarks: [...new Set(v.runs.map((r) => r.protocolLabel))],
      evidence: v.evidence,
      versionCount: v.patch.versions.length,
      results,
      shown,
      hidden: hiddenFields(v),
      terms: versionTerms(v),
    };
  });
}

/** Values of one facet for one entry. */
export function facetValues(e: SearchEntry, k: FacetKey): string[] {
  switch (k) {
    case "tag":
      return e.tags;
    case "benchmark":
      return e.benchmarks;
    case "evidence":
      return e.evidence;
    case "format":
      return e.formats;
    default:
      return [e[k]];
  }
}
