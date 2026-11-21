import { readJson } from "./files.ts";
import { href, seg } from "./config.ts";
import type { FileEntry, Patch, Run, Version } from "./types.ts";

export function activeVersion(patch: Patch) {
  return patch.versions.find(v => v.n === patch.latest && v.status === "active");
}

export function downloadUrl(file: FileEntry) {
  return file.repoPath ? href(`downloads/${file.sha256}/${seg(file.name)}`) : file.links[0].href;
}

export function formats(v?: Version): string[] {
  if (!v) return [];
  const names = v.files.map(f => f.name);
  const hasLora = v.patch.kind === "lora" || names.some(n => /^(adapter|adapter_model)\.safetensors$|\.gguf$/i.test(n));
  return [...(v.patch.kind === "replacement" ? ["Replacement"] : []), ...(hasLora ? ["LoRA"] : [])];
}

export function discoveryText(address: string, patches: Patch[], available: string[]) {
  return [address, ...available, ...patches.flatMap(p =>
    [p.title, p.summary, p.slug, ...p.tags, ...p.authors.map(a => a.name)])].join(" ").toLowerCase();
}

export interface Coverage { filed: number; total: number }

export function readCoverage(reason?: string): Coverage | null {
  const match = /^(\d+) of (\d+) item-seed pairs filed$/.exec(reason ?? "");
  if (!match) return null;
  const [, filed, total] = match.map(Number);
  return total > 0 && filed <= total ? { filed, total } : null;
}

/** Reported: the patch authors' own published figures. Measured: a run of the
 * index's own benchmark, filed with every request and reply (the receipts), by
 * the index team or any contributor. Neither word ranks one above the other. */
export const sourceLabels = {
  index: "Measured",
  author: "Reported",
};

export function columnBins(columns: number[], width: number, count: number) {
  const bins = Array<number>(count).fill(0);
  for (const col of columns) bins[Math.floor(col * count / width)]++;
  return bins;
}

interface TensorReport { shape: number[]; active_column_indices?: number[] }
export function footprint(v: Version) {
  const file = v.files.find(f => f.name === "lora-report.json" && f.repoPath);
  if (!file?.repoPath) return null;
  const report = readJson<{ method?: string; tensors?: Record<string, TensorReport> }>(file.repoPath);
  if (report.method !== "selector-exact") return null;
  const layers = Object.entries(report.tensors ?? {}).flatMap(([name, tensor]) => {
    const layer = /\.layers\.(\d+)\./.exec(name)?.[1];
    const cols = tensor.active_column_indices;
    const width = tensor.shape?.[1];
    if (!layer || !Array.isArray(cols) || !Number.isSafeInteger(width) || width <= 0 ||
        !cols.length || !cols.every(i => Number.isSafeInteger(i) && i >= 0 && i < width)) return [];
    return [{ name, layer, width, columns: cols.length, bins: columnBins(cols, width, 56) }];
  });
  return layers.length ? { layers, file } : null;
}
