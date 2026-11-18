// Repackage the canonical selector factors for PEFT. Renaming the tensor keys
// leaves the F32 payload untouched. Only the unit-scale, common-rank case has
// a portable interpretation across the three runtimes shown on the site.
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { mkdtempSync, mkdirSync, rmSync, utimesSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { gzipSync, gunzipSync } from "node:zlib";
import { exists, fail, readBytes, readJson } from "./files.ts";
import type { Profile, Version } from "./index.ts";

interface Tensor {
  shape: number[];
  dtype: string;
  data_offsets: [number, number];
}

export function makePeftFiles(factors: Buffer, source: Profile["source"]) {
  if (factors.length < 8) return null;
  const length = Number(factors.readBigUInt64LE(0));
  if (length < 2 || length > factors.length - 8) return null;
  const header: Record<string, Tensor> = JSON.parse(factors.subarray(8, 8 + length).toString("utf8"));
  const payload = factors.subarray(8 + length);
  const renamed: Record<string, Tensor> = {};
  const pairs = new Map<string, { A?: Tensor; B?: Tensor }>();
  for (const [name, tensor] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    const match = /^([a-zA-Z0-9_.]+)\.weight\.lora_([AB])$/.exec(name);
    if (!match || tensor.dtype !== "F32" || tensor.shape.length !== 2) return null;
    const [start, end] = tensor.data_offsets;
    const [rows, cols] = tensor.shape;
    if (!Number.isSafeInteger(rows) || !Number.isSafeInteger(cols) || rows < 1 || cols < 1 ||
        start < 0 || end > payload.length || end - start !== rows * cols * 4) return null;
    const [, module, side] = match;
    const pair = pairs.get(module) ?? {};
    pair[side as "A" | "B"] = tensor;
    pairs.set(module, pair);
    renamed[`base_model.model.${module}.lora_${side}.weight`] = tensor;
  }
  const ranks = new Set<number>();
  for (const { A, B } of pairs.values()) {
    if (!A || !B || A.shape[0] !== B.shape[1]) return null;
    ranks.add(A.shape[0]);
  }
  if (ranks.size !== 1) return null;
  const [rank] = ranks;
  const config = {
    base_model_name_or_path: source.repo_id, revision: source.revision, peft_type: "LORA",
    inference_mode: true, bias: "none", r: rank, lora_alpha: rank, lora_dropout: 0,
    target_modules: [...pairs.keys()].sort(),
  };
  const json = Buffer.from(JSON.stringify({ __metadata__: { format: "pt" }, ...renamed }));
  const pad = Buffer.alloc((8 - json.length % 8) % 8, 0x20);
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(json.length + pad.length));
  return { config, weights: Buffer.concat([prefix, json, pad, payload]) };
}

/** Use the same system tar required by the site snapshot check. Only regular
 * files enter the archive, with normalized timestamps and file permissions. */
export function tarGzip(files: Record<string, Buffer>): Buffer {
  const dir = mkdtempSync(path.join(tmpdir(), "ws-adapter-"));
  try {
    let total = 0;
    for (const [name, data] of Object.entries(files)) {
      if (!/^(?:notices\/)?[a-zA-Z0-9][\w.-]*$/.test(name)) throw new Error(`Unexpected package path ${name}`);
      const dest = path.join(dir, name);
      mkdirSync(path.dirname(dest), { recursive: true });
      writeFileSync(dest, data, { mode: 0o644 });
      utimesSync(dest, 0, 0);
      total += data.length + 1024;
    }
    const archive = execFileSync("tar", ["--format=ustar", "--owner=0", "--group=0", "--numeric-owner", "-cf", "-", "-C", dir, ...Object.keys(files).sort()], {
      maxBuffer: total + 1024 * 64,
      env: { ...process.env, COPYFILE_DISABLE: "1" },
    });
    return gzipSync(archive);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

export interface AdapterBundle {
  filename: string;
  bytes: Buffer;
  rank: number;
  gguf: boolean;
}

const cache = new WeakMap<Version, AdapterBundle | null>();

export function adapterBundle(v: Version): AdapterBundle | null {
  if (cache.has(v)) return cache.get(v)!;
  const bundle = buildBundle(v);
  cache.set(v, bundle);
  return bundle;
}

function buildBundle(v: Version): AdapterBundle | null {
  const factors = v.files.find(f => f.name === "adapter.safetensors" && f.role === "derived");
  const reportFile = v.files.find(f => f.name === "lora-report.json");
  if (!factors?.repoPath || !reportFile?.repoPath) return null;
  const report = readJson(reportFile.repoPath);
  if (report.tool?.name !== "weightspace derive-lora" || report.method !== "selector-exact" ||
      report.status !== "complete" || report.effective_scale !== 1 ||
      report.adapter_safetensors_sha256 !== factors.sha256) return null;
  const peft = makePeftFiles(readBytes(factors.repoPath), v.profile.source);
  if (!peft) return null;
  const files: Record<string, Buffer> = {
    "adapter_model.safetensors": peft.weights,
    "adapter_config.json": Buffer.from(JSON.stringify(peft.config, null, 2) + "\n"),
    "provenance.json": Buffer.from(JSON.stringify({
      patch: v.patch.slug, version: v.label, authors: v.patch.authors, license: v.patch.license,
      source_file: factors.name, source_sha256: factors.sha256, source: factors.links[0].href,
      modification: "PEFT tensor names and configuration added. F32 factor bytes preserved.",
    }, null, 2) + "\n"),
  };
  for (const notice of v.patch.notices) if (notice.repoPath) files[`notices/${notice.name}`] = readBytes(notice.repoPath);
  // A GGUF export is specific to the base architecture and exact factor bytes.
  const exported = `site/src/assets/runtime/${factors.sha256}`;
  let gguf = false;
  if (exists(exported + ".json")) {
    const manifest = readJson(exported + ".json");
    const base = v.profile.source;
    if (manifest.base === base.repo_id && manifest.revision === base.revision && manifest.source_sha256 === factors.sha256) {
      const bytes = gunzipSync(readBytes(exported + ".gguf.gz"));
      if (createHash("sha256").update(bytes).digest("hex") !== manifest.gguf_sha256) fail("GGUF export digest mismatch");
      files["adapter.gguf"] = bytes;
      files["gguf-provenance.json"] = readBytes(exported + ".json");
      gguf = true;
    }
  }
  return { filename: `${v.patch.slug}-${v.label}-lora.tar.gz`, bytes: tarGzip(files), rank: peft.config.r, gguf };
}
