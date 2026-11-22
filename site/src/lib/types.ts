import type { SessionRecord } from "./session.ts";
import type { Sampling } from "./sampling.ts";

export type Locator =
  | { kind: "index-path"; path: string }
  | { kind: "hf"; repo_type: "model" | "dataset"; repo_id: string; revision: string; path: string };

export interface Link {
  host: "Hugging Face" | "GitHub";
  href: string;
}

export interface FileEntry {
  name: string;
  role?: "source" | "derived";
  format?: string;
  sha256: string;
  size_bytes: number;
  /** "recorded": digest stated by a record and, when hosted here, checked against the bytes. "computed": no record states one; computed from the bytes at the pinned commit. */
  digest: "recorded" | "computed";
  repoPath?: string;
  links: Link[];
  view?: string;
  derived_from?: string[];
  tool?: { name: string; version: string };
}

export interface Profile {
  id: string;
  source: { repo_id: string; revision: string };
  files: { path: string; sha256: string; size_bytes: number }[];
}

export interface Score {
  state?: "UNKNOWN";
  reason?: string;
  owner?: string;
  metric?: string;
  value?: number;
  /** bench runs: lm-eval's own score, beside the headline. */
  value_lm_eval?: number;
  /** bench runs: share of replies cut at the token limit. */
  cut_at_limit?: number;
  /** bench runs: share of replies that finished without an answer. */
  finished_without_answer?: number;
  /** bench runs: mean reply length in tokens. */
  mean_completion_tokens?: number;
  unit?: string;
  denominator?: number;
}

export interface RunRecord {
  contract_version: 1;
  profile: string;
  protocol: { name: string; sha256: string };
  dataset: { name: string; sha256: string };
  settings: Record<string, any>;
  state: "COMPLETE" | "INCOMPLETE";
  score: Score;
  outcomes: { sha256: string; size_bytes: number; locators: Locator[] };
  resources?: { sha256: string; size_bytes: number; locators: Locator[] };
}

export interface Session {
  name: string;
  dir: string;
  record: SessionRecord;
  files: FileEntry[];
}

export interface Run {
  id: string;
  dir: string;
  url: string;
  base: Base;
  version: Version | null;
  record: RunRecord;
  outcomes: Record<string, unknown>[] | null;
  outcomesFile: FileEntry;
  resourcesFile: FileEntry | null;
  sessions: Session[];
  paired: Run | null;
  group: Group;
  protocolLabel: string;
  settingsLabel: string;
  /** bench's own names, when the run was filed by bench/ (`--protocol`, `--settings`). */
  bench: { protocol: string; settings: string } | null;
  protocolFile: FileEntry | null;
  settingsFile: FileEntry | null;
  recordFile: FileEntry;
}

export interface Comparison {
  id: string;
  version: Version;
  record: any;
  file: FileEntry;
  baseline: Run;
  patched: Run;
  matched: FileEntry | null;
}

export interface Assessment {
  id: string;
  version: Version;
  record: any;
  file: FileEntry;
  comparisons: Comparison[];
  policy: FileEntry | null;
}

/** One measure the patch authors published, as recorded in a claim. */
export interface ClaimMeasure {
  benchmark: string;
  name?: string;
  metric?: string;
  unit: string;
  baseline: number;
  patched: number;
  better: "higher" | "lower";
  samples_per_item?: number;
}

/** `v<N>/claims/<id>.json`: aggregate results the patch authors published themselves. */
export interface Claim {
  id: string;
  version: Version;
  record: { source: { url: string; title: string }; affiliation?: string; conditions: string; sampling?: Sampling; measures: ClaimMeasure[] };
  file: FileEntry;
}

export interface TensorForms {
  /** Tensors stored as `#idx/#new/#old` trios: apply checks the saved old values. */
  sparse: string[];
  /** Tensors stored whole: apply needs `--expect-tensor-sha256` for each. */
  full: string[];
}

export interface Version {
  n: number;
  label: string;
  patch: Patch;
  dir: string;
  url: string;
  status: "active" | "deprecated" | "withdrawn";
  isLatest: boolean;
  changelog: string;
  profile: Profile;
  files: FileEntry[];
  record: FileEntry;
  runs: Run[];
  comparisons: Comparison[];
  assessments: Assessment[];
  claims: Claim[];
  /** Read from the safetensors headers of in-repository replacement sources; null when any source is hosted elsewhere. */
  tensors: TensorForms | null;
  evidence: string[];
}

export interface Patch {
  slug: string;
  base: Base;
  dir: string;
  url: string;
  title: string;
  summary: string;
  kind: "replacement" | "lora";
  authors: { name: string; url?: string }[];
  submitter: string;
  license: string;
  tags: string[];
  latest: number | null;
  versions: Version[];
  card: { repoPath: string; markdown: string; file: FileEntry } | null;
  notices: FileEntry[];
  record: FileEntry;
}

export interface Group {
  id: string;
  base: Base;
  profile: Profile;
  protocol: { name: string; sha256: string };
  dataset: { name: string; sha256: string };
  protocolLabel: string;
  settingsLabel: string;
  label: string;
  runs: Run[];
}

export interface Base {
  address: string;
  ns: string;
  model: string;
  dir: string;
  url: string;
  profiles: Profile[];
  baselineRuns: Run[];
  patches: Patch[];
  groups: Group[];
  runs: Run[];
  baseRecord: FileEntry;
}

export interface Index {
  commit: string;
  short: string;
  dirty: string[];
  example: boolean;
  collection: string;
  githubRepo: string;
  bases: Base[];
  benchAvailable: boolean;
  docsAt: (repoPath: string) => string;
}
