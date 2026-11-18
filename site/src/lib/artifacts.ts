import path from "node:path";
import { GITHUB_REPO, seg } from "./config.ts";
import { digestOf, exists, fail, isDir, joinRepo, readBytes, verifyDigest } from "./files.ts";
import type { FileEntry, Link, Locator, TensorForms } from "./types.ts";

export function artifactFiles(commit: string) {
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

  return { blob, recorded, computed, optionalFile };
}

/** Reads the tensor names from each in-repository replacement source's safetensors header. */
export function readTensorForms(sources: FileEntry[]): TensorForms | null {
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
