// Loads one index collection (the real one, or docs/example) into the shape
// the pages render. Every value comes from a file at the pinned commit; a
// missing or inconsistent record stops the build.
import { createHash } from "node:crypto";
import path from "node:path";
import {
  digestOf,
  exists,
  fail,
  isDir,
  joinRepo,
  listDir,
  pinCommit,
  readBytes,
  readRecord,
  readText,
  verifyDigest,
  type Pin,
} from "./files.ts";
import { checkSession, type SessionRecord } from "./session.ts";
import { COLLECTION, EXAMPLE, GITHUB_REPO, PIN_PATHS, PREFIX, href, seg } from "./config.ts";

export { href, GITHUB_REPO };

// ------------------------------------------------------------------ types

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
  unit?: string;
  denominator?: number;
}

export interface RunRecord {
  contract_version: 1;
  profile: string;
  performed_by: "index" | "author" | "third-party";
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

// ------------------------------------------------------------------ helpers

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") {
    const o = value as Record<string, unknown>;
    return `{${Object.keys(o)
      .sort()
      .map((k) => `${JSON.stringify(k)}:${canonical(o[k])}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}

function sha(text: string): string {
  return createHash("sha256").update(text).digest("hex");
}

function stem(p: string): string {
  return path.posix.basename(p).replace(/\.json$/, "");
}

/** Run IDs, slugs and labels made safe for an HTML id. */
function anchor(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

// ------------------------------------------------------------------ loader

let cached: Index | null = null;

export function loadIndex(): Index {
  if (!cached) cached = build();
  return cached;
}

function build(): Index {
  const example = EXAMPLE;
  const prefix = PREFIX;
  const pin: Pin = pinCommit(PIN_PATHS);
  const dev = Boolean(import.meta.env?.DEV);
  if (pin.dirty.length && !dev && process.env.WS_ALLOW_DIRTY !== "1") {
    fail(
      `index files differ from commit ${pin.commit.slice(0, 12)}; download links would point at other bytes. ` +
        `Commit or stash first:\n  ${pin.dirty.slice(0, 20).join("\n  ")}`,
    );
  }
  const commit = pin.commit;
  const raw = (p: string) => `https://raw.githubusercontent.com/${GITHUB_REPO}/${commit}/${seg(p)}`;
  const blob = (p: string) => `https://github.com/${GITHUB_REPO}/blob/${commit}/${seg(p)}`;

  function hf(loc: Extract<Locator, { kind: "hf" }>): string {
    const kind = loc.repo_type === "dataset" ? "datasets/" : "";
    return `https://huggingface.co/${kind}${loc.repo_id}/resolve/${loc.revision}/${seg(loc.path)}`;
  }

  /** A file described by a record: size and SHA-256 recorded, in-repository bytes checked. */
  function recorded(
    name: string,
    artifact: { sha256: string; size_bytes: number; locators: Locator[] },
    declaringDir: string,
    declaredIn: string,
    extra: Partial<FileEntry> = {},
  ): FileEntry {
    const links: Link[] = [];
    let repoPath: string | undefined;
    // Clients try Hugging Face first (structure.md, "Where files live").
    for (const loc of artifact.locators) if (loc.kind === "hf") links.push({ host: "Hugging Face", href: hf(loc) });
    for (const loc of artifact.locators) {
      if (loc.kind !== "index-path") continue;
      const p = joinRepo(declaringDir, loc.path);
      verifyDigest(p, artifact, declaredIn);
      repoPath ??= p;
      links.push({ host: "GitHub", href: raw(p) });
    }
    return {
      name,
      sha256: artifact.sha256,
      size_bytes: artifact.size_bytes,
      digest: "recorded",
      repoPath,
      links,
      view: repoPath ? blob(repoPath) : undefined,
      ...extra,
    };
  }

  /** A file no record describes (cards, notices, bench sessions): digest computed from the pinned bytes. */
  function computed(repoPath: string): FileEntry {
    const d = digestOf(repoPath);
    return {
      name: path.posix.basename(repoPath),
      sha256: d.sha256,
      size_bytes: d.size_bytes,
      digest: "computed",
      repoPath,
      links: [{ host: "GitHub", href: raw(repoPath) }],
      view: blob(repoPath),
    };
  }

  function optionalFile(repoPath: string): FileEntry | null {
    return exists(repoPath) && !isDir(repoPath) ? computed(repoPath) : null;
  }

  const modelsIndex = readRecord<{ bases: string[] }>(`${prefix}models/index.json`, "models index");
  const bases: Base[] = [];

  for (const address of modelsIndex.bases) {
    const [ns, model] = address.split("/");
    const dir = `${prefix}models/${ns}/${model}`;
    const idx = readRecord<any>(`${dir}/index.json`, "base index");
    if (idx.base !== address) fail(`${dir}/index.json names base ${idx.base}, expected ${address}`);
    const baseJson = readRecord<{ profiles: Profile[] }>(`${dir}/base.json`, "base record");
    const base: Base = {
      address,
      ns,
      model,
      dir,
      url: href(`models/${seg(address)}/`),
      profiles: baseJson.profiles,
      baselineRuns: [],
      patches: [],
      groups: [],
      runs: [],
      baseRecord: computed(`${dir}/base.json`),
    };
    const profileById = (id: string, where: string): Profile =>
      base.profiles.find((p) => p.id === id) ?? fail(`${where} names profile ${id}, which ${dir}/base.json does not define`);

    const runsByDir = new Map<string, Run>();
    const groups = new Map<string, Group>();

    function loadRun(runDir: string, id: string, version: Version | null, url: string): Run {
      const runPath = `${runDir}/run.json`;
      const record = readRecord<RunRecord>(runPath, "run record");
      const profile = profileById(record.profile, runPath);
      const outcomesFile = recorded("outcomes", record.outcomes, runDir, runPath);
      let outcomes: Record<string, unknown>[] | null = null;
      if (outcomesFile.repoPath) {
        outcomes = readText(outcomesFile.repoPath)
          .split("\n")
          .filter((l) => l.trim())
          .map((l, i) => {
            try {
              const o = JSON.parse(l);
              if (!o || typeof o !== "object" || Array.isArray(o)) throw new Error("not an object");
              return o;
            } catch (e) {
              fail(`${outcomesFile.repoPath} line ${i + 1} is not a JSON object (${(e as Error).message})`);
            }
          });
      }
      const resourcesFile = record.resources ? recorded("resources", record.resources, runDir, runPath) : null;
      const s = record.settings ?? {};

      // bench/ conventions inside the free-form settings object (bench/README.md, "merge").
      const settingsFileName = typeof s.file === "string" ? s.file : null;
      const settingsSha = typeof s.sha256 === "string" ? s.sha256 : null;
      const settingsKey = settingsFileName && settingsSha ? `file:${settingsFileName}@${settingsSha}` : `json:${canonical(s)}`;
      const settingsLabel = settingsFileName ? stem(settingsFileName) : `settings ${sha(canonical(s)).slice(0, 8)}`;
      const protoMatch = /^bench\/protocols\/([A-Za-z0-9._-]+)\.json$/.exec(record.protocol.name);
      const setMatch = settingsFileName ? /^bench\/settings\/([A-Za-z0-9._-]+)\.json$/.exec(settingsFileName) : null;
      const protocolLabel = protoMatch ? protoMatch[1] : record.protocol.name;
      const benchNames =
        protoMatch && setMatch && exists(record.protocol.name) && exists(settingsFileName!)
          ? { protocol: protoMatch[1], settings: setMatch[1] }
          : null;
      const protocolFile = protoMatch ? optionalFile(record.protocol.name) : null;
      const settingsFile = settingsFileName && setMatch ? optionalFile(settingsFileName) : null;
      if (settingsFile && settingsSha && settingsFile.sha256 !== settingsSha) {
        fail(`${runPath} records settings ${settingsFileName} sha256 ${settingsSha}, but the file has ${settingsFile.sha256}`);
      }

      // Bench sessions behind the run (bench/results/...); session.json has no index schema, so it is shape-checked here.
      const sessions: Session[] = [];
      if (s.sessions && typeof s.sessions === "object") {
        if (typeof s.results !== "string") fail(`${runPath} lists sessions but no settings.results folder`);
        for (const [name, summary] of Object.entries<any>(s.sessions)) {
          const sdir = joinRepo(s.results, name);
          if (!isDir(sdir)) fail(`${runPath} lists session ${name}, but ${sdir} is missing`);
          const rec = checkSession(`${sdir}/session.json`, readRecordLoose(`${sdir}/session.json`));
          if (rec.seed !== summary.seed) fail(`${sdir}/session.json seed ${rec.seed} differs from ${runPath} (${summary.seed})`);
          if (rec.items.length !== summary.items) {
            fail(`${sdir}/session.json has ${rec.items.length} items, ${runPath} says ${summary.items}`);
          }
          if (rec.profile !== record.profile) fail(`${sdir}/session.json profile ${rec.profile} differs from ${runPath}`);
          sessions.push({ name, dir: sdir, record: rec, files: listDir(sdir).map((f) => computed(`${sdir}/${f}`)) });
        }
      }

      const key = [record.profile, record.protocol.sha256, record.dataset.sha256, settingsKey].join("|");
      let group = groups.get(key);
      if (!group) {
        group = {
          id: anchor(`${protocolLabel}-${settingsLabel}-${record.profile}`) + "-" + sha(key).slice(0, 6),
          base,
          profile,
          protocol: record.protocol,
          dataset: record.dataset,
          protocolLabel,
          settingsLabel,
          label: `${protocolLabel} · ${settingsLabel} · ${record.profile}`,
          runs: [],
        };
        groups.set(key, group);
      }
      const run: Run = {
        id,
        dir: runDir,
        url,
        base,
        version,
        record,
        outcomes,
        outcomesFile: { ...outcomesFile, name: path.posix.basename(outcomesFile.repoPath ?? "outcomes") },
        resourcesFile,
        sessions,
        paired: null,
        group,
        protocolLabel,
        settingsLabel,
        bench: benchNames,
        protocolFile,
        settingsFile,
        recordFile: computed(runPath),
      };
      group.runs.push(run);
      runsByDir.set(runDir, run);
      base.runs.push(run);
      return run;
    }

    for (const r of idx.baseline_runs as { id: string }[]) {
      base.baselineRuns.push(loadRun(`${dir}/runs/${r.id}`, r.id, null, href(`models/${seg(address)}/runs/${seg(r.id)}/`)));
    }

    for (const p of idx.patches as any[]) {
      const pdir = `${dir}/${p.slug}`;
      const patchJson = readRecord<any>(`${pdir}/patch.json`, "patch record");
      for (const k of ["title", "summary", "kind", "license", "submitter"]) {
        if (patchJson[k] !== p[k]) fail(`${dir}/index.json is out of date for ${p.slug}.${k}; run weightspace generate`);
      }
      const cardPath = `${pdir}/README.md`;
      const patch: Patch = {
        slug: p.slug,
        base,
        dir: pdir,
        url: href(`models/${seg(address)}/${p.slug}/`),
        title: p.title,
        summary: p.summary,
        kind: p.kind,
        authors: p.authors,
        submitter: p.submitter,
        license: p.license,
        tags: p.tags,
        latest: p.latest ?? null,
        versions: [],
        card: exists(cardPath) ? { repoPath: cardPath, markdown: readText(cardPath), file: computed(cardPath) } : null,
        notices: listDir(`${pdir}/notices`).map((f) => computed(`${pdir}/notices/${f}`)),
        record: computed(`${pdir}/patch.json`),
      };
      base.patches.push(patch);

      for (const v of p.versions as any[]) {
        const vdir = `${pdir}/v${v.version}`;
        const vjson = readRecord<any>(`${vdir}/version.json`, "version record");
        const status = patchJson.versions.find((x: any) => x.version === v.version)?.status;
        if (status !== v.status) fail(`${pdir}/patch.json and ${dir}/index.json disagree on v${v.version} status`);
        const version: Version = {
          n: v.version,
          label: `v${v.version}`,
          patch,
          dir: vdir,
          url: href(`models/${seg(address)}/${p.slug}/v${v.version}/`),
          status: v.status,
          isLatest: p.latest === v.version,
          changelog: vjson.changelog,
          profile: profileById(vjson.target_profile, `${vdir}/version.json`),
          files: [],
          record: computed(`${vdir}/version.json`),
          runs: [],
          comparisons: [],
          assessments: [],
          tensors: null,
          evidence: [],
        };
        const formats = new Map<string, string>((v.files as any[]).map((f) => [f.name, f.format]));
        version.files = (vjson.files as any[]).map((f) =>
          recorded(f.name, f, vdir, `${vdir}/version.json`, {
            role: f.role,
            format: formats.get(f.name) ?? fail(`${dir}/index.json lacks file ${f.name} of ${p.slug} v${v.version}`),
            derived_from: f.derived_from,
            tool: f.tool,
          }),
        );
        if (patch.kind === "replacement") version.tensors = readTensorForms(version.files.filter((f) => f.role === "source"));

        for (const r of v.runs as { id: string }[]) {
          version.runs.push(
            loadRun(`${vdir}/runs/${r.id}`, r.id, version, href(`models/${seg(address)}/${p.slug}/v${v.version}/runs/${seg(r.id)}/`)),
          );
        }
        patch.versions.push(version);
      }
    }

    // Paired runs (bench files each patched run with its own paired baseline, settings.paired_with).
    for (const run of runsByDir.values()) {
      const pw = run.record.settings?.paired_with;
      if (pw === undefined) continue;
      if (typeof pw !== "string") fail(`${run.dir}/run.json: settings.paired_with is not a path`);
      run.paired = runsByDir.get(pw) ?? fail(`${run.dir}/run.json is paired with ${pw}, which is not a run of ${address}`);
      if (run.paired.group !== run.group) fail(`${run.dir} and its paired run ${pw} differ in profile, protocol, dataset or settings`);
    }

    // Comparisons and assessments: files under the version folder (not listed in index.json).
    for (const patch of base.patches) {
      for (const version of patch.versions) {
        const runId = (id: string, list: Run[], where: string) =>
          list.find((r) => r.id === id) ?? fail(`${where} names run ${id}, which does not exist`);
        for (const f of listDir(`${version.dir}/comparisons`)) {
          const fpath = `${version.dir}/comparisons/${f}`;
          if (!f.endsWith(".json")) fail(`unexpected file ${fpath}`);
          const record = readRecord<any>(fpath, "comparison record");
          version.comparisons.push({
            id: f.replace(/\.json$/, ""),
            version,
            record,
            file: computed(fpath),
            baseline: runId(record.baseline_run, base.baselineRuns, fpath),
            patched: runId(record.patched_run, version.runs, fpath),
            matched: record.matched_outcomes ? recorded("matched outcomes", record.matched_outcomes, `${version.dir}/comparisons`, fpath) : null,
          });
        }
        for (const f of listDir(`${version.dir}/assessments`)) {
          const fpath = `${version.dir}/assessments/${f}`;
          if (!f.endsWith(".json")) fail(`unexpected file ${fpath}`);
          const record = readRecord<any>(fpath, "assessment record");
          version.assessments.push({
            id: f.replace(/\.json$/, ""),
            version,
            record,
            file: computed(fpath),
            comparisons: (record.comparisons ?? []).map(
              (id: string) => version.comparisons.find((c) => c.id === id) ?? fail(`${fpath} names comparison ${id}, which does not exist`),
            ),
            policy: record.policy ? recorded("policy", record.policy, `${version.dir}/assessments`, fpath) : null,
          });
        }
        version.evidence = evidenceOf(version);
      }
    }

    base.groups = [...groups.values()];
    bases.push(base);
  }

  return {
    commit,
    short: commit.slice(0, 7),
    dirty: pin.dirty,
    example,
    collection: COLLECTION,
    githubRepo: GITHUB_REPO,
    bases,
    benchAvailable: !example && exists("bench/README.md"),
    docsAt: blob,
  };
}

function readRecordLoose(p: string): unknown {
  try {
    return JSON.parse(readText(p, "bench session"));
  } catch (e) {
    fail(`bench session is not valid JSON: ${p} (${(e as Error).message})`);
  }
}

/** Reads the tensor names from each in-repository replacement source's safetensors header. */
function readTensorForms(sources: FileEntry[]): TensorForms | null {
  const sparse = new Set<string>();
  const full = new Set<string>();
  for (const f of sources) {
    if (!f.repoPath || !f.name.endsWith(".safetensors")) return null;
    const bytes = readBytes(f.repoPath);
    if (bytes.length < 8) fail(`${f.repoPath} is too short to be safetensors`);
    const n = Number(bytes.readBigUInt64LE(0));
    if (n <= 0 || 8 + n > bytes.length) fail(`${f.repoPath} has an invalid safetensors header length`);
    let header: Record<string, unknown>;
    try {
      header = JSON.parse(bytes.subarray(8, 8 + n).toString("utf8"));
    } catch {
      fail(`${f.repoPath} has an unreadable safetensors header`);
    }
    for (const name of Object.keys(header)) {
      if (name === "__metadata__") continue;
      const m = /^(.*)#(idx|new|old)$/.exec(name);
      if (m) sparse.add(m[1]);
      else full.add(name);
    }
  }
  return { sparse: [...sparse].sort(), full: [...full].sort() };
}

/** Evidence states for search and badges, in the wording of docs/structure.md "Evidence states". */
export const NO_BENCHMARKS = "no benchmarks submitted";

function evidenceOf(v: Version): string[] {
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

// ------------------------------------------------------------------ queries

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
export function measuredBy(run: Run): "index" | "reported" {
  return run.record.performed_by === "index" ? "index" : "reported";
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
