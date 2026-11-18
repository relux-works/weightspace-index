// Search entries, one per patch version, built from the records. The search
// page fetches them as search.json and filters in the browser.
import { allVersions, type Index } from "./index.ts";

export interface SearchEntry {
  url: string;
  base: string;
  profile: string;
  slug: string;
  title: string;
  summary: string;
  kind: string;
  tags: string[];
  license: string;
  authors: string[];
  submitter: string;
  version: string;
  status: string;
  latest: boolean;
  benchmarks: string[];
  evidence: string[];
}

export const FACETS = [
  ["base", "Base"],
  ["profile", "Profile"],
  ["kind", "Kind"],
  ["tag", "Tag"],
  ["license", "License"],
  ["status", "Version status"],
  ["benchmark", "Benchmark"],
  ["evidence", "Evidence"],
] as const;

export type FacetKey = (typeof FACETS)[number][0];

export function searchEntries(index: Index): SearchEntry[] {
  return allVersions(index).map((v) => ({
    url: v.isLatest ? v.patch.url : v.url,
    base: v.patch.base.address,
    profile: v.profile.id,
    slug: v.patch.slug,
    title: v.patch.title,
    summary: v.patch.summary,
    kind: v.patch.kind,
    tags: v.patch.tags,
    license: v.patch.license,
    authors: v.patch.authors.map((a) => a.name),
    submitter: v.patch.submitter,
    version: v.label,
    status: v.status,
    latest: v.isLatest,
    benchmarks: [...new Set(v.runs.map((r) => r.protocolLabel))],
    evidence: v.evidence,
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
    default:
      return [e[k]];
  }
}
