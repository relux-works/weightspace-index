// Reading files from the index checkout. Every read goes through here so a
// missing file or an unsupported contract version stops the build.
import { createHash } from "node:crypto";
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { execFileSync } from "node:child_process";
import path from "node:path";

export const SUPPORTED_CONTRACT = 1;

/** Repository root of the index checkout (the parent of site/). */
export const REPO_ROOT = path.resolve(process.env.WS_REPO_ROOT ?? path.resolve(process.cwd(), ".."));

export class IndexError extends Error {
  constructor(message: string) {
    super(`[weightspace site] ${message}`);
  }
}

export function fail(message: string): never {
  throw new IndexError(message);
}

export function abs(repoPath: string): string {
  return path.join(REPO_ROOT, ...repoPath.split("/"));
}

export function exists(repoPath: string): boolean {
  return existsSync(abs(repoPath));
}

export function isDir(repoPath: string): boolean {
  return exists(repoPath) && statSync(abs(repoPath)).isDirectory();
}

export function readBytes(repoPath: string, what = "file"): Buffer {
  if (!exists(repoPath)) fail(`missing ${what}: ${repoPath}`);
  return readFileSync(abs(repoPath));
}

export function readText(repoPath: string, what = "file"): string {
  return readBytes(repoPath, what).toString("utf8");
}

export function readJson<T = any>(repoPath: string, what = "record"): T {
  const text = readText(repoPath, what);
  try {
    return JSON.parse(text) as T;
  } catch (e) {
    fail(`${what} is not valid JSON: ${repoPath} (${(e as Error).message})`);
  }
}

/** Reads an index record and stops on a contract version this site does not support. */
export function readRecord<T = any>(repoPath: string, what = "record"): T {
  const value = readJson<any>(repoPath, what);
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    fail(`${what} is not a JSON object: ${repoPath}`);
  }
  if (!("contract_version" in value)) fail(`${what} has no contract_version: ${repoPath}`);
  if (value.contract_version !== SUPPORTED_CONTRACT) {
    fail(
      `unsupported contract_version ${JSON.stringify(value.contract_version)} in ${repoPath}; ` +
        `this site reads contract version ${SUPPORTED_CONTRACT} only. Update the site before building.`,
    );
  }
  return value as T;
}

export function listDir(repoPath: string): string[] {
  if (!isDir(repoPath)) return [];
  return readdirSync(abs(repoPath)).filter((n) => !n.startsWith(".")).sort();
}

export interface Digest {
  sha256: string;
  size_bytes: number;
}

const digestCache = new Map<string, Digest>();

export function digestOf(repoPath: string): Digest {
  let d = digestCache.get(repoPath);
  if (!d) {
    const bytes = readBytes(repoPath);
    d = { sha256: createHash("sha256").update(bytes).digest("hex"), size_bytes: bytes.length };
    digestCache.set(repoPath, d);
  }
  return d;
}

/** Fails when an in-repository file differs from the size and SHA-256 its record states. */
export function verifyDigest(repoPath: string, expected: Digest, declaredIn: string): void {
  const got = digestOf(repoPath);
  if (got.size_bytes !== expected.size_bytes || got.sha256 !== expected.sha256) {
    fail(
      `${repoPath} does not match ${declaredIn}: recorded ${expected.size_bytes} bytes ` +
        `sha256 ${expected.sha256}, found ${got.size_bytes} bytes sha256 ${got.sha256}`,
    );
  }
}

/** Joins a record-relative path onto the folder of the declaring file, refusing escapes. */
export function joinRepo(dir: string, rel: string): string {
  const joined = path.posix.normalize(path.posix.join(dir, rel));
  if (joined.startsWith("..") || path.posix.isAbsolute(joined)) fail(`path escapes the repository: ${dir} + ${rel}`);
  return joined;
}

function git(args: string[]): string {
  return execFileSync("git", args, { cwd: REPO_ROOT, encoding: "utf8" }).trim();
}

/**
 * When a record first appeared in the index: the author time, in seconds, of
 * the commit that added it. A record not committed yet counts as the newest.
 */
export function firstAdded(repoPath: string): number {
  let out = "";
  try { out = git(["log", "--diff-filter=A", "--format=%at", "--", repoPath]); } catch { /* not a checkout */ }
  const oldest = out.split("\n").filter(Boolean).at(-1);
  return oldest ? Number(oldest) : Number.POSITIVE_INFINITY;
}

export interface Pin {
  commit: string;
  dirty: string[];
}

let pinned: Pin | null = null;

/** The index commit every page and download link is built from (structure.md, step 2). Resolved once per build. */
export function pinCommit(paths: string[]): Pin {
  pinned ??= resolvePin(paths);
  return pinned;
}

function resolvePin(paths: string[]): Pin {
  let commit: string;
  try {
    commit = git(["rev-parse", "HEAD"]);
  } catch {
    fail(`cannot resolve the index commit: ${REPO_ROOT} is not a git checkout`);
  }
  if (!/^[0-9a-f]{40}$/.test(commit)) fail(`unexpected commit id from git: ${commit}`);
  const status = git(["status", "--porcelain", "--untracked-files=all", "--", ...paths]);
  const dirty = status ? status.split("\n").map((l) => l.slice(3)) : [];
  return { commit, dirty };
}
