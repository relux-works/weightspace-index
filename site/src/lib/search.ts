// Search entries, one per patch version, built from the records. The search
// page fetches them as search.json and filters in the browser. The homepage
// directory matches the same terms, so both searches agree.
import { authorReport } from "../data/reports.ts";
import { formats } from "./catalog.ts";
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

/** Everything a result shows about one version, plus its short name and goal. */
export function versionTerms(v: Version): string {
  const report = authorReport(v);
  return [
    v.patch.base.address, v.profile.id, v.patch.kind, ...formats(v), v.patch.title, v.patch.summary, v.patch.slug,
    ...v.patch.tags, v.patch.license, ...v.patch.authors.map((a) => a.name), v.patch.submitter, v.label, v.status,
    ...v.runs.map((r) => r.protocolLabel), ...v.evidence, report?.name ?? "", report?.goal ?? "",
  ].join(" ").toLowerCase();
}

export function searchEntries(index: Index): SearchEntry[] {
  return allVersions(index).map((v) => ({
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
    terms: versionTerms(v),
  }));
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
