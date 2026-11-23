// Copyable commands. Each one is a command documented in this repository,
// filled in from the records; nothing is written per page. `source` names the
// document the command form comes from, and the page links to it.
import type { Base, FileEntry, Index, Profile, Run, Version } from "./index.ts";
import { assetDir, assetUrl, download, type ToolAsset, type ToolRelease } from "./tool-release.ts";

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
  release: { label: "tool/README.md, “From a release archive”", path: "tool/README.md", anchor: "from-a-release-archive" },
  install: { label: "tool/README.md, “From source”", path: "tool/README.md", anchor: "from-source" },
  replacement: { label: "tool/README.md, “Build a replacement patch”", path: "tool/README.md", anchor: "build-a-replacement-patch-patch-authors" },
};

/** The tool's own documented forms, with the README's placeholder paths. Used by /weightspace/. */
/**
 * Downloads one platform's release file, checks it against the recorded SHA-256
 * before anything runs, and puts the binary on PATH. Uses the bare binary when
 * the release publishes one, else unpacks the archive.
 */
export function toolDownload(githubRepo: string, release: ToolRelease, asset: ToolAsset): Command {
  const file = download(asset);
  const url = assetUrl(githubRepo, release, file);
  const dir = assetDir(release, asset);
  const lines = asset.os === "windows"
    ? [
        `Invoke-WebRequest ${url} -OutFile ${file.name}`,
        `if ((Get-FileHash ${file.name}).Hash -ne '${file.sha256.toUpperCase()}') { throw 'checksum mismatch' }`,
        ...(file.ready
          ? [`# Rename ${file.name} to weightspace.exe and move it to a folder on your PATH`]
          : [`Expand-Archive ${file.name} -DestinationPath .`, `# Move ${dir}\\weightspace.exe to a folder on your PATH`]),
      ]
    : [
        `curl -fLO ${url}`,
        `echo '${file.sha256}  ${file.name}' | ${asset.os === "linux" ? "sha256sum -c" : "shasum -a 256 -c"}`,
        ...(file.ready
          ? [`sudo install -m 755 ${file.name} /usr/local/bin/weightspace`]
          : [`tar -xzf ${file.name}`, `sudo install ${dir}/weightspace /usr/local/bin/`]),
      ];
  return { text: lines.join("\n"), source: DOCS.release };
}

export function toolInstall(githubRepo: string): Command {
  const dir = githubRepo.split("/").pop();
  return {
    text: `git clone https://github.com/${githubRepo}\ncd ${dir}/tool\ncargo install --path . --locked`,
    source: DOCS.install,
  };
}

export function toolHomebrew(): Command {
  return {
    text: "brew install relux-works/tap/weightspace\n# Installs both `weightspace` and the `wsp` short alias",
    source: DOCS.release,
  };
}

export function toolApt(): Command {
  return {
    text: [
      "curl -fsSL https://relux-works.github.io/apt/relux-works.gpg | sudo tee /usr/share/keyrings/relux-works.gpg >/dev/null",
      'echo "deb [signed-by=/usr/share/keyrings/relux-works.gpg] https://relux-works.github.io/apt stable main" | sudo tee /etc/apt/sources.list.d/relux-works.list',
      "sudo apt update && sudo apt install weightspace",
    ].join("\n"),
    source: DOCS.release,
  };
}

export const toolCommands = {
  tui: {
    text: "# Launch interactive TUI (detects local models & engines)\nwsp\n\n# Or use hf-style subcommands (no auth required)\nwsp search qwen\nwsp models\nwsp run Qwen/Qwen3.8-27B/harness-hub-pi --engine llama.cpp",
    source: DOCS.apply,
  },
  apply: {
    text: "weightspace apply \\\n  --patch ./my-patch.safetensors \\\n  --model ./my-model \\\n  --out ./my-model-patched",
    source: DOCS.apply,
  },
  stack: {
    text: "# Stack multiple replacements with an explicit conflict policy\nweightspace apply \\\n  --patch ./patch-a.safetensors \\\n  --patch ./patch-b.safetensors \\\n  --on-conflict add \\\n  --model ./my-model \\\n  --in-place\n\n# Restore byte-identical original weights from .weightspace-backup/\nweightspace restore --model ./my-model",
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
