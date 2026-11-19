// Reader-facing summaries of the records: before/after results per source,
// what a release edits, and how combinations behaved. Values come from the
// records unchanged; differences are shown next to the two values they come
// from and are never ranked across sources (rule 9).
import { activeVersion, footprint, readCoverage } from "./catalog.ts";
import { bestSingle, combinationStudies, jointOutcome } from "./combinations.ts";
import { exists, readBytes, readJson } from "./files.ts";
import { rankable } from "./queries.ts";
import type { Base, Run, Version } from "./types.ts";

export type Source = "author" | "index" | "third-party";

export interface Result {
  source: Source;
  benchmark: string;
  /** What the row measures when it is not the benchmark's own score (e.g. "Hit token limit"). */
  name?: string;
  metric: string;
  unit: string;
  base: number;
  patched: number;
  /** Higher is better unless the measure says otherwise (e.g. hitting the token limit). */
  higherIsBetter: boolean;
  n?: number;
  /** Answers drawn per problem, when the source states it. */
  samples?: number;
  /** Identifies the baseline measurement; equal keys mean one shared unpatched result. */
  baseKey: string;
  version: Version;
  href: string;
}

export interface Pending {
  source: Source;
  benchmark: string;
  filed: number;
  total: number;
  version: Version;
  href: string;
}

/** The matched unpatched baseline for a patched run, when the two are comparable. */
export function pairedBaseline(v: Version, run: Run): Run | undefined {
  const explicit = v.comparisons.find(c => c.patched === run && c.record.state === "MATCHED")?.baseline;
  const b = explicit ?? run.paired;
  if (!b || b.version || b.group !== run.group || !rankable(b) || !rankable(run)) return;
  const s = run.record.score, t = b.record.score;
  return s.metric === t.metric && s.unit === t.unit && s.denominator === t.denominator ? b : undefined;
}

function scale(unit: string, n: number) {
  return unit === "fraction" ? { unit: "%", value: Math.round(n * 1000) / 10 } : { unit, value: n };
}

/** The dataset's own name for a bench protocol (MATH-500, not math500-n197). */
function benchmarkName(run: Run) {
  const file = run.record.protocol.name;
  if (!/^bench\/protocols\/[A-Za-z0-9._-]+\.json$/.test(file) || !exists(file)) return run.protocolLabel;
  return readJson<{ dataset?: { repo_id?: string } }>(file).dataset?.repo_id?.split("/").pop() || run.protocolLabel;
}

/** The small print under a result's name: metric, item count and samples per problem. */
/** A result's row title: what it measures, or the benchmark when it is the benchmark's own score. */
export const resultTitle = (r: Pick<Result, "name" | "benchmark">) => r.name ?? r.benchmark;

/** The small print under a result's title: benchmark (when the title is not it), metric, item count, samples per problem. */
export const resultNote = (r: Pick<Result, "name" | "benchmark" | "metric" | "n" | "samples">) =>
  [r.name && r.benchmark, r.metric, r.n && `n = ${r.n}`, r.samples && `${r.samples} samples per problem`].filter(Boolean).join(" · ");

/** Every before/after pair for one release, author reports first. */
export function results(v: Version | undefined): Result[] {
  if (!v) return [];
  const out: Result[] = [];
  for (const c of v.claims) {
    const url = c.record.source.url;
    for (const m of c.record.measures) {
      out.push({
        source: "author", benchmark: m.benchmark, name: m.name, metric: m.metric ?? "", unit: m.unit, base: m.baseline, patched: m.patched,
        samples: m.samples_per_item, higherIsBetter: m.better === "higher",
        baseKey: `${url}#${m.benchmark}|${m.name ?? ""}|${m.metric ?? ""}@${m.baseline}`, version: v, href: url,
      });
    }
  }
  for (const run of v.runs) {
    const base = pairedBaseline(v, run);
    if (!base) continue;
    const s = run.record.score;
    const a = scale(s.unit!, base.record.score.value!), b = scale(s.unit!, s.value!);
    out.push({
      source: run.record.performed_by, benchmark: benchmarkName(run), metric: s.metric!, unit: b.unit,
      base: a.value, patched: b.value, higherIsBetter: true, n: s.denominator, baseKey: base.url, version: v, href: run.url,
    });
  }
  return out;
}

/** Independent runs that are filed but not yet scored. */
export function pending(v: Version | undefined): Pending[] {
  if (!v) return [];
  return v.runs.flatMap(run => {
    if (rankable(run)) return [];
    const c = readCoverage(run.record.score.reason);
    return c ? [{ source: run.record.performed_by, benchmark: benchmarkName(run), ...c, version: v, href: run.url }] : [];
  });
}

export const gain = (r: Pick<Result, "base" | "patched" | "higherIsBetter">) => (r.patched - r.base) * (r.higherIsBetter ? 1 : -1);

/** Preserve the performer on scores, partial coverage and underlying run records. */
export function groupEvidence(scores: Result[], progress: Pending[], runs: Run[]) {
  return (["author", "index", "third-party"] as const).map(source => ({
    source,
    scores: scores.filter(r => r.source === source),
    progress: progress.filter(r => r.source === source),
    runs: runs.filter(r => r.record.performed_by === source),
  }));
}

/** Pick a representative comparison without sorting different protocols by effect size. */
export function previewResult(rows: Result[]) {
  return rows.find(r => r.source === "index") ?? rows.find(r => r.source === "third-party") ?? rows[0];
}

export function signed(n: number, unit = "") {
  const v = Math.round(n * 10) / 10;
  const u = unit === "%" ? " pts" : unit ? ` ${unit}` : "";
  return `${v > 0 ? "+" : v < 0 ? "−" : "±"}${Math.abs(v)}${u}`;
}

export function fmt(n: number, unit: string) {
  return unit === "%" ? `${n}%` : `${n}${unit ? " " + unit : ""}`;
}

// ── What a release edits ─────────────────────────────────────────────────────

export interface EditStats {
  tensors: { name: string; short: string; layer: string; rows: number; cols: number; active: number; bins: number[]; columns: number[] }[];
  elements: number;
  tensorElements: number;
  modelParams: number | null;
  rank: number | null;
  replacementBytes: number | null;
  loraBytes: number | null;
  exact: boolean;
  maxUlp: number | null;
}

interface ReportTensor {
  shape: number[]; active_column_indices?: number[]; measured_active_elements?: number; all_elements?: number;
  rank?: number; max_ulp?: number; measured_bake_exact?: boolean;
}

/** Edit footprint from the release's own lora-report (selector-exact method only). */
export function editStats(v: Version | undefined): EditStats | null {
  if (!v) return null;
  const fp = footprint(v);
  if (!fp) return null;
  const report = readJson<{ tensors: Record<string, ReportTensor>; tensor_shapes?: Record<string, number[]> }>(fp.file.repoPath!);
  const entries = Object.entries(report.tensors);
  const tensors = fp.layers.map(l => {
    const shape = report.tensor_shapes?.[l.name] ?? [0, l.width];
    return {
      name: l.name, layer: l.layer, rows: shape[0], cols: shape[1], active: l.columns, bins: l.bins,
      columns: report.tensors[l.name]?.active_column_indices ?? [],
      short: l.name.replace(/^model\.(language_model\.)?/, "").replace(/\.weight$/, ""),
    };
  });
  const elements = entries.reduce((n, [, t]) => n + (t.measured_active_elements ?? 0), 0);
  const tensorElements = entries.reduce((n, [, t]) => n + (t.all_elements ?? 0), 0);
  const weights = v.profile.files.filter(f => f.path.endsWith(".safetensors"));
  const bytesPerParam = v.profile.id === "bf16" || v.profile.id === "f16" || v.profile.id === "fp16" ? 2 : v.profile.id === "f32" ? 4 : 0;
  const modelParams = bytesPerParam && weights.length ? Math.round(weights.reduce((n, f) => n + f.size_bytes, 0) / bytesPerParam) : null;
  const ranks = entries.map(([, t]) => t.rank).filter((r): r is number => typeof r === "number");
  const ulps = entries.map(([, t]) => t.max_ulp).filter((r): r is number => typeof r === "number");
  const source = v.files.find(f => f.role === "source" && f.name.endsWith(".safetensors"));
  const lora = v.files.find(f => /^(adapter|adapter_model)\.safetensors$/.test(f.name));
  return {
    tensors, elements, tensorElements, modelParams,
    rank: ranks.length ? Math.max(...ranks) : null,
    replacementBytes: v.patch.kind === "replacement" ? source?.size_bytes ?? null : null,
    loraBytes: lora?.size_bytes ?? null,
    exact: entries.every(([, t]) => t.measured_bake_exact === true),
    maxUlp: ulps.length ? Math.max(...ulps) : null,
  };
}

/** The release with the most benchmark evidence: scored results plus filed runs. Real releases before examples. */
export function mostBenchmarked(bases: Base[]) {
  const releases = bases.flatMap(b => b.patches.map(activeVersion)).filter((v): v is Version => !!v);
  const real = releases.filter(v => !v.dir.includes("docs/example/"));
  const weight = (v: Version) => results(v).length + v.runs.length;
  return (real.length ? real : releases).reduce<Version | undefined>((best, v) => (!best || weight(v) > weight(best) ? v : best), undefined);
}

export function compact(n: number) {
  if (n >= 1e9) return `${+(n / 1e9).toFixed(n >= 1e10 ? 0 : 2)}B`;
  if (n >= 1e6) return `${+(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)}M`;
  if (n >= 1e3) return `${+(n / 1e3).toFixed(n >= 1e4 ? 0 : 1)}K`;
  return String(n);
}

export function percent(part: number, whole: number) {
  const p = (100 * part) / whole;
  return p >= 1 ? `${+p.toFixed(1)}%` : p >= 0.01 ? `${+p.toFixed(3)}%` : `${+p.toPrecision(2)}%`;
}

// ── Combinations ─────────────────────────────────────────────────────────────

export type Verdict = "amplify" | "neutral" | "interfere";
export const VERDICT_LABEL: Record<Verdict, string> = {
  amplify: "Amplify",
  neutral: "No extra gain",
  interfere: "Interfere",
};

export function combos(base: Base, version?: Version) {
  return combinationStudies(base, version).map(s => {
    const outcome = jointOutcome(s.together, s.members.map(m => m.alone), s.direction);
    const verdict: Verdict = outcome.startsWith("Better") ? "amplify" : outcome.startsWith("Matches") ? "neutral" : "interfere";
    return { ...s, outcome, verdict, best: bestSingle(s.members.map(m => m.alone), s.direction) };
  });
}

// ── Real per-weight changes ──────────────────────────────────────────────────

export interface DeltaCell { row: number; col: number; old: number; new: number }
export interface DeltaTensor {
  name: string; short: string; rows: number; cols: number;
  /** Edited input columns, ascending. */
  columns: number[];
  /** Rows sampled evenly from the tensor, top to bottom. */
  sampleRows: number[];
  /** cells[r][c]: change at sampleRows[r], columns[c]; null where the value is unchanged. */
  cells: (DeltaCell | null)[][];
  /** |Δ| that maps to full colour (95th percentile of the tensor's changes). */
  scale: number;
  changed: number;
  /** Changed values as a share of the edited columns' cells. */
  coverage: number;
}

const bf16 = (u: number) => {
  const b = Buffer.alloc(4); b.writeUInt32LE(u << 16 >>> 0); return b.readFloatLE(0);
};

/**
 * The exact old and new values stored in a sparse replacement file
 * (`<tensor>#idx` / `#old` / `#new`), sampled into a small grid per tensor so
 * pages can draw how strongly each edited weight moves.
 */
export function editDeltas(v: Version | undefined, sample = 24): DeltaTensor[] {
  if (!v || v.patch.kind !== "replacement") return [];
  const src = v.files.find(f => f.role === "source" && f.repoPath && f.name.endsWith(".safetensors"));
  const stats = editStats(v);
  if (!src?.repoPath || !stats) return [];
  const buf = readBytes(src.repoPath, "replacement file");
  const n = Number(buf.readBigUInt64LE(0));
  const header = JSON.parse(buf.subarray(8, 8 + n).toString("utf8")) as Record<string, { dtype: string; shape: number[]; data_offsets: [number, number] }>;
  const body = buf.subarray(8 + n);
  const out: DeltaTensor[] = [];
  for (const t of stats.tensors) {
    const idx = header[`${t.name}#idx`], o = header[`${t.name}#old`], w = header[`${t.name}#new`];
    if (!idx || !o || !w || idx.dtype !== "I64" || o.dtype !== "BF16" || w.dtype !== "BF16") continue;
    const count = idx.shape[0];
    const byCell = new Map<number, [number, number]>();
    const mags: number[] = [];
    for (let i = 0; i < count; i++) {
      const flat = Number(body.readBigInt64LE(idx.data_offsets[0] + i * 8));
      const ov = bf16(body.readUInt16LE(o.data_offsets[0] + i * 2));
      const nv = bf16(body.readUInt16LE(w.data_offsets[0] + i * 2));
      byCell.set(flat, [ov, nv]);
      mags.push(Math.abs(nv - ov));
    }
    mags.sort((a, b) => a - b);
    const columns = t.columns.length ? t.columns.toSorted((a, b) => a - b)
      : [...new Set([...byCell.keys()].map(f => f % t.cols))].sort((a, b) => a - b);
    const rowsN = Math.min(sample, t.rows);
    const sampleRows = Array.from({ length: rowsN }, (_, r) => Math.floor(((r + 0.5) * t.rows) / rowsN));
    const cells = sampleRows.map(row => columns.map(col => {
      const hit = byCell.get(row * t.cols + col);
      return hit ? { row, col, old: hit[0], new: hit[1] } : null;
    }));
    out.push({
      name: t.name, short: t.short, rows: t.rows, cols: t.cols, columns, sampleRows, cells,
      scale: mags[Math.floor(mags.length * 0.95)] || mags[mags.length - 1] || 1,
      changed: count, coverage: columns.length ? count / (columns.length * t.rows) : 0,
    });
  }
  return out;
}

/** 0–1 colour strength for a change, linear in its size against the 95th percentile. */
export function strength(c: DeltaCell | null, scale: number) {
  return c ? Math.min(1, Math.abs(c.new - c.old) / scale) : 0;
}
