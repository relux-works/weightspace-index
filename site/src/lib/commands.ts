// Copyable commands. Each one is a command documented in this repository,
// filled in from the records; nothing is written per page. `source` names the
// document the command form comes from, and the page links to it.
import type { Base, FileEntry, Index, Profile, Run, Version } from "./index.ts";

export interface Command {
  text: string;
  source: { label: string; path: string; anchor?: string };
}

const DOCS = {
  hf: { label: "docs/benchmarks/qwen3.5-0.8b-m1-pro.md", path: "docs/benchmarks/qwen3.5-0.8b-m1-pro.md" },
  verify: { label: "README.md, “Use a patch”", path: "README.md", anchor: "use-a-patch" },
  apply: { label: "tool/README.md, “Apply a replacement patch”", path: "tool/README.md", anchor: "apply-a-replacement-patch-patch-users" },
  bench: { label: "bench/README.md", path: "bench/README.md", anchor: "running" },
  benchStatus: { label: "bench/README.md", path: "bench/README.md", anchor: "status-merging-checking-comparing" },
  install: { label: "tool/README.md, “From source”", path: "tool/README.md", anchor: "from-source" },
  replacement: { label: "tool/README.md, “Build a replacement patch”", path: "tool/README.md", anchor: "build-a-replacement-patch-patch-authors" },
};

/** The tool's own documented forms, with the README's placeholder paths. Used by /weightspace/. */
export function toolInstall(githubRepo: string): Command {
  const dir = githubRepo.split("/").pop();
  return {
    text: `git clone https://github.com/${githubRepo}\ncd ${dir}/tool\ncargo build --locked --release --bin weightspace`,
    source: DOCS.install,
  };
}

export const toolCommands = {
  apply: {
    text: "weightspace apply \\\n  --patch ./my-patch.safetensors \\\n  --model ./my-model \\\n  --out ./my-model-patched",
    source: DOCS.apply,
  },
  replacement: {
    text: "weightspace replacement \\\n  --base ./base.safetensors \\\n  --edited ./edited.safetensors \\\n  --out ./my-patch.safetensors",
    source: DOCS.replacement,
  },
} satisfies Record<string, Command>;

/** The folder a profile is downloaded into, used by every later command. */
export function profileDir(base: Base, profile: Profile): string {
  return `${base.model}-${profile.id}`;
}

/** Downloads the profile's repository at its pinned revision (the whole snapshot, as bench does). */
export function hfDownload(base: Base, profile: Profile): Command {
  return {
    text: `hf download ${profile.source.repo_id} \\\n  --revision ${profile.source.revision} \\\n  --local-dir ${profileDir(base, profile)}`,
    source: DOCS.hf,
  };
}

/** Prints a file's SHA-256 and byte count, to compare with the record. */
export function verify(filePath: string): Command {
  return { text: `shasum -a 256 ${filePath}\nwc -c < ${filePath}`, source: DOCS.verify };
}

export interface ApplyPlan {
  command: Command | null;
  /** Why the command is missing or will not run as shown. */
  notes: string[];
  needsDigests: string[];
}

export function apply(v: Version): ApplyPlan {
  const patch = v.patch;
  const notes: string[] = [];
  if (patch.kind !== "replacement") {
    return {
      command: null,
      needsDigests: [],
      notes: ["A LoRA is loaded or merged by your runtime; the index adds no command of its own. See the runtime notes in docs/workflows.md."],
    };
  }
  const sources = v.files.filter((f) => f.role === "source");
  if (sources.length !== 1) {
    return {
      command: null,
      needsDigests: [],
      notes: [`This version has ${sources.length} source files; weightspace apply takes one patch file per copy.`],
    };
  }
  const src: FileEntry = sources[0];
  const dir = profileDir(patch.base, v.profile);
  const out = `${dir}-${patch.slug}-${v.label}`;
  const text = `weightspace apply \\\n  --patch ${src.name} \\\n  --model ${dir} \\\n  --out ${out}`;
  let needsDigests: string[] = [];
  if (v.tensors === null) {
    notes.push(
      "The source file is hosted outside this repository, so the site cannot read which tensors it stores whole. " +
        "If apply asks for --expect-tensor-sha256, take the digest from the patch author, never from your own model.",
    );
  } else if (v.tensors.full.length) {
    needsDigests = v.tensors.full;
    notes.push(
      "This patch stores whole tensors. apply refuses them until you add --expect-tensor-sha256 NAME=HEX for each one below. " +
        "The index does not record these base-tensor digests; take them from the patch author, never from your own model.",
    );
  }
  return { command: { text, source: DOCS.apply }, notes, needsDigests };
}

export interface BenchCommands {
  protocol: string;
  settings: string;
  run: Command;
  status: Command;
  compare: Command;
}

function benchFor(v: Version, protocol: string, settings: string): BenchCommands {
  const args = `--entry ${v.dir} --protocol ${protocol} --settings ${settings}`;
  return {
    protocol,
    settings,
    run: {
      text: `python -m bench run \\\n  --entry ${v.dir} \\\n  --protocol ${protocol} \\\n  --settings ${settings} \\\n  --hardware "<your hardware, vLLM version>"`,
      source: DOCS.bench,
    },
    status: { text: `python -m bench status ${args}`, source: DOCS.benchStatus },
    compare: { text: `python -m bench compare ${args}`, source: DOCS.benchStatus },
  };
}

/**
 * bench commands for a version: one set per protocol and settings already
 * used on the version's target profile of this base (a settings file applies
 * only to the base whose card it cites).
 */
export function benchCommands(index: Index, v: Version): BenchCommands[] {
  if (!index.benchAvailable) return [];
  const seen = new Map<string, BenchCommands>();
  for (const r of v.patch.base.runs) {
    if (!r.bench || r.record.profile !== v.profile.id) continue;
    const key = `${r.bench.protocol}|${r.bench.settings}`;
    if (!seen.has(key)) seen.set(key, benchFor(v, r.bench.protocol, r.bench.settings));
  }
  return [...seen.values()];
}

/** Commands for the version a run belongs to (a paired baseline belongs to its patched run's version). */
export function benchForRun(index: Index, run: Run): BenchCommands | null {
  if (!index.benchAvailable || !run.bench) return null;
  const v = run.version ?? run.paired?.version ?? null;
  return v ? benchFor(v, run.bench.protocol, run.bench.settings) : null;
}
