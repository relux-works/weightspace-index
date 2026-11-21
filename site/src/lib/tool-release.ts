// The published weightspace release the /weightspace/ page offers. The site is
// static and builds offline, so the release is recorded in
// src/data/tool-release.json and updated by hand after each tagged release
// (see site/README.md). Sizes and digests come from the release's assets.
import data from "../data/tool-release.json" with { type: "json" };

export type ToolOs = "macos" | "linux" | "windows";

export interface ToolAsset {
  target: string;
  os: ToolOs;
  label: string;
  arch: string;
  name: string;
  size: number;
  sha256: string;
  /** The unpacked binary, published from releases after v0.1.0-alpha.1. */
  binary?: ToolFile;
}

export interface ToolFile {
  name: string;
  size: number;
  sha256: string;
}

export interface ToolRelease {
  tag: string;
  toolVersion: string;
  published: string;
  assets: ToolAsset[];
}

const OS = new Set<string>(["macos", "linux", "windows"]);

/** Throws when the recorded release does not follow the release workflow's naming. */
export function checkRelease(release: ToolRelease): ToolRelease {
  if (!/^v\d+\.\d+\.\d+(-[\w.]+)?$/.test(release.tag)) throw new Error(`tool release: bad tag ${release.tag}`);
  if (!release.assets.length) throw new Error("tool release: no assets");
  for (const a of release.assets) {
    const ext = a.os === "windows" ? "zip" : "tar.gz";
    if (!OS.has(a.os)) throw new Error(`tool release: unknown os ${a.os}`);
    if (a.name !== `weightspace-${release.tag}-${a.target}.${ext}`) throw new Error(`tool release: ${a.name} does not match its tag and target`);
    checkFile(a);
    if (a.binary) {
      const want = `weightspace-${release.tag}-${a.target}${a.os === "windows" ? ".exe" : ""}`;
      if (a.binary.name !== want) throw new Error(`tool release: ${a.binary.name} should be ${want}`);
      checkFile(a.binary);
    }
  }
  return release;
}

function checkFile(f: ToolFile) {
  if (!/^[0-9a-f]{64}$/.test(f.sha256)) throw new Error(`tool release: bad sha256 for ${f.name}`);
  if (!Number.isInteger(f.size) || f.size <= 0) throw new Error(`tool release: bad size for ${f.name}`);
}

/** What the page offers for a platform: the ready-to-run binary when the release has one, else the archive. */
export const download = (asset: ToolAsset): ToolFile & { ready: boolean } =>
  asset.binary ? { ...asset.binary, ready: true } : { name: asset.name, size: asset.size, sha256: asset.sha256, ready: false };

export const toolRelease: ToolRelease = checkRelease(data as ToolRelease);

export const releasePage = (repo: string, release: ToolRelease) =>
  `https://github.com/${repo}/releases/tag/${release.tag}`;

export const assetUrl = (repo: string, release: ToolRelease, file: { name: string }) =>
  `https://github.com/${repo}/releases/download/${release.tag}/${file.name}`;

/** The folder each archive unpacks to. */
export const assetDir = (release: ToolRelease, asset: ToolAsset) => `weightspace-${release.tag}-${asset.target}`;
