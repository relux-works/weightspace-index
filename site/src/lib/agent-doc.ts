// Generates machine-readable Markdown documentation for AI agents (/agent.md)
// and a concise llmstxt.org index (/llms.txt) directly from the live index.
import type { Index, Patch, Version } from "./types.ts";
import { activeVersion, downloadUrl } from "./catalog.ts";
import { humanBytes } from "./format.ts";
import { href } from "./config.ts";
import { runtimeGuide } from "./runtime.ts";

function fmtVal(value: number, unit: string): string {
  if (unit === "%" || unit === "proportion") {
    const num = unit === "proportion" ? value * 100 : value;
    return `${num.toFixed(1)}%`;
  }
  return `${value} ${unit}`;
}

export const SITE_ORIGIN = (process.env.WS_SITE_ORIGIN ?? "https://weightspaceindex.com").replace(/\/+$/, "");

function absUrl(relPath: string): string {
  const path = relPath.startsWith("/") ? relPath : href(relPath);
  return `${SITE_ORIGIN}${path}`;
}

function formatClaimLine(version: Version): string[] {
  const lines: string[] = [];
  for (const claim of version.claims) {
    const src = claim.record.affiliation
      ? `${claim.record.affiliation} — ${claim.record.source.title}`
      : claim.record.source.title;
    for (const m of claim.record.measures) {
      const label = m.name ? `${m.benchmark} (${m.name})` : m.benchmark;
      const delta = m.patched - m.baseline;
      const sign = delta > 0 ? "+" : "";
      const unit = m.unit === "%" ? "pp" : ` ${m.unit}`;
      lines.push(
        `  - Reported (${src}): ${label}: ${m.baseline}${m.unit} -> ${m.patched}${m.unit} (${sign}${delta.toFixed(1)}${unit}, better: ${m.better})`,
      );
    }
  }
  return lines;
}

function formatMeasuredLine(version: Version): string[] {
  const lines: string[] = [];
  for (const run of version.runs) {
    const score = run.record.score;
    if (score.state === "UNKNOWN" || score.value === undefined) continue;
    const baseScore = run.paired?.record.score;
    const baseStr =
      baseScore && baseScore.value !== undefined
        ? `${fmtVal(baseScore.value, baseScore.unit ?? "%")} -> `
        : "";
    lines.push(
      `  - Measured [${run.record.state}] (${run.protocolLabel}): ${baseStr}${fmtVal(score.value, score.unit ?? "%")} (${run.id})`,
    );
  }
  return lines;
}

function formatPatchSection(patch: Patch): string {
  const v = activeVersion(patch);
  const lines: string[] = [];
  lines.push(`#### \`${patch.base.address}/${patch.slug}\` — ${patch.title}`);
  lines.push("");
  lines.push(`- **Kind:** \`${patch.kind}\``);
  lines.push(`- **Summary:** ${patch.summary}`);
  lines.push(`- **License:** \`${patch.license}\``);
  if (patch.tags.length > 0) {
    lines.push(`- **Tags:** ${patch.tags.map((t) => `\`${t}\``).join(", ")}`);
  }
  lines.push(`- **Authors:** ${patch.authors.map((a) => (a.url ? `[${a.name}](${a.url})` : a.name)).join(", ")}`);
  lines.push(`- **Page:** ${absUrl(patch.url)}`);

  if (!v) {
    lines.push("- **Latest active version:** none");
    return lines.join("\n");
  }

  lines.push(`- **Latest active version:** \`${v.label}\` (target profile \`${v.profile.id}\` @ \`${v.profile.source.repo_id}@${v.profile.source.revision}\`)`);
  lines.push("- **Artifacts:**");
  for (const f of v.files) {
    const url = downloadUrl(f);
    const fullUrl = url.startsWith("http") ? url : absUrl(url);
    lines.push(`  - \`${f.name}\` (${f.role ?? "file"}, ${humanBytes(f.size_bytes)}, \`sha256:${f.sha256}\`): ${fullUrl}`);
  }

  const evidence = [...formatClaimLine(v), ...formatMeasuredLine(v)];
  if (evidence.length > 0) {
    lines.push("- **Benchmarks:**");
    lines.push(...evidence);
  }

  const guide = runtimeGuide(v);
  if (guide.tabs.length > 0) {
    lines.push(`- **Supported runtimes:** ${guide.tabs.map((t) => t.name).join(", ")}`);
  }

  return lines.join("\n");
}

export function renderLlmsTxt(index: Index): string {
  const totalPatches = index.bases.reduce((sum, b) => sum + b.patches.length, 0);
  const lines: string[] = [
    "# Weightspace Index",
    "",
    "> Open catalog of verified weight patches (bit-exact replacements and native LoRA adapters) for open-weight AI models, pinned by exact base checkpoint revision, byte size, and SHA-256 digest.",
    "",
    `- Snapshot commit: \`${index.githubRepo}@${index.commit}\` (contract v1)`,
    `- Catalog size: ${index.bases.length} base model(s), ${totalPatches} patch(es)`,
    `- Full self-contained agent manual: ${absUrl("agent.md")}`,
    `- Structured search index (JSON): ${absUrl("search.json")}`,
    "",
    "## Core Documentation",
    "",
    `- [Complete Agent Guide (Markdown)](${absUrl("agent.md")}): Everything an AI agent needs — catalog, SHA-256 digests, download URLs, CLI commands, custom registries, and submission rules`,
    `- [Models Directory](${absUrl("")}): Interactive catalog of base models, patches, and benchmark evidence`,
    `- [How It Works](${absUrl("how-it-works/")}): Direct bit-exact replacements vs LoRA adapters, compute/precision/size tradeoffs`,
    `- [CLI Tool (weightspace / wsp)](${absUrl("weightspace/")}): Download, verify, apply, stack patches, and launch inference engines`,
    `- [All Benchmark Runs](${absUrl("compare/")}): Reported vs independently measured benchmark evidence`,
    `- [Repository Structure & Contract](${index.docsAt("docs/structure.md")}): Schema, layout, and integrity guarantees`,
    "",
    "## Base Models",
    "",
  ];

  for (const base of index.bases) {
    const activePatches = base.patches.filter((p) => activeVersion(p) !== undefined);
    lines.push(
      `- [${base.address}](${absUrl(base.url)}): ${activePatches.length} active patch(es) — ${activePatches.map((p) => `${p.slug} (${p.kind})`).join(", ") || "no active patches"}`,
    );
  }

  lines.push("");
  lines.push("## Policy & Trust Boundaries");
  lines.push("");
  lines.push("- Only weight patches (sparse/targeted replacements) and native LoRA adapters are indexed — never full-weight fine-tunes.");
  lines.push("- The default Relux Works registry excludes abliterated/uncensored patches; third-party/community registries can be added in `wsp` with zero authentication.");
  lines.push("- Reported author claims never transfer between formats (replacement <-> derived LoRA) and are strictly separated from Measured runs.");
  lines.push("");

  return lines.join("\n");
}

export function renderAgentMarkdown(index: Index): string {
  const totalPatches = index.bases.reduce((sum, b) => sum + b.patches.length, 0);
  const lines: string[] = [
    "# Weightspace Index — Complete Agent Reference",
    "",
    `> Built from \`${index.githubRepo}@${index.commit}\` (contract version 1).  `,
    `> Live site: ${SITE_ORIGIN} · Concise index: ${absUrl("llms.txt")} · JSON index: ${absUrl("search.json")}`,
    "",
    "## What Weightspace Index Gives You",
    "",
    "Instead of downloading multiple 15–70 GB full model checkpoints for different coding harnesses, reasoning styles, or tool-use behaviors, download a **single pinned base model once** and apply small, SHA-256-verified **weight patches**:",
    "",
    "1. **Bit-exact replacement patches (`kind: \"replacement\"`)** — bake targeted weight changes once into a copy of an unquantized checkpoint via `weightspace apply`. Zero runtime overhead; exact bit-for-bit match.",
    "2. **Native LoRA adapters (`kind: \"lora\"`)** — attach dynamically in `llama.cpp`, `vLLM`, or `Transformers/PEFT` (including quantized GGUF bases), and stack multiple adapters at runtime.",
    "",
    "Every entry is pinned to an exact upstream base `repo_id`, 40-character Git/HF `revision`, byte size, and `sha256` digest. Author-reported benchmarks (**Reported**) are never mixed with independent reproductions (**Measured**).",
    "",
    "## Machine-Readable Endpoints",
    "",
    "| Endpoint | Purpose |",
    "|---|---|",
    `| \`${absUrl("agent.md")}\` | This single-file Markdown manual with the full live catalog |`,
    `| \`${absUrl("llms.txt")}\` | Compact llmstxt.org summary & links |`,
    `| \`${absUrl("search.json")}\` | Structured JSON array of all models, patches, tags, and URLs |`,
    `| \`https://raw.githubusercontent.com/${index.githubRepo}/${index.commit}/models/index.json\` | Canonical base-model address list (Git/HTTP registry entry point) |`,
    `| \`https://raw.githubusercontent.com/${index.githubRepo}/${index.commit}/lookup/by-repo/<ns>/<repo>.json\` | Fast lookup by Hugging Face base \`repo_id\` |`,
    `| \`https://raw.githubusercontent.com/${index.githubRepo}/${index.commit}/lookup/by-digest/<aa>/<sha256>.json\` | Fast lookup by local weight file SHA-256 |`,
    "",
    "## CLI Quickstart (`weightspace` / `wsp`)",
    "",
    "Download binaries from " + absUrl("weightspace/") + " (macOS Apple Silicon, Linux x86_64, Windows x86_64).",
    "",
    "```sh",
    "# Bake a bit-exact replacement into a new model directory (original untouched)",
    "weightspace apply \\",
    "  --patch ./replacements.safetensors \\",
    "  --model ./base-model \\",
    "  --out ./patched-model",
    "",
    "# Run a GGUF LoRA in llama.cpp",
    "llama-server -m ./base.gguf --lora ./adapter.gguf",
    "",
    "# Serve a PEFT LoRA in vLLM",
    "vllm serve <base-repo-id> --revision <40-char-rev> --enable-lora --lora-modules <slug>=./<lora-dir>",
    "```",
    "",
    "## Custom & Private Git Registries",
    "",
    "The Weightspace Index format is pure Git + static JSON — it is **not** locked to GitHub. Any Git host (self-hosted GitLab, Gitea, Forgejo, Bitbucket, internal mirrors, local folders) or static HTTPS server serving `models/index.json` and `lookup/` works as a registry.",
    "",
    "Configure registries in `~/.config/weightspace/registries.toml`:",
    "",
    "```toml",
    'default = "official"',
    "",
    "[[registries]]",
    'name = "official"',
    'url = "https://github.com/relux-works/weightspace-index.git"',
    "priority = 100",
    "",
    "[[registries]]",
    'name = "internal-gitlab"',
    'url = "ssh://git@gitlab.internal.example/ml/weightspace-index.git"',
    'pin = "v2026.11"',
    "priority = 200",
    "```",
    "",
    "## Live Catalog",
    "",
    `Total: **${index.bases.length} base model(s)**, **${totalPatches} patch(es)**.`,
    "",
  ];

  for (const base of index.bases) {
    lines.push(`### Base Model: \`${base.address}\``);
    lines.push("");
    lines.push(`- **Page:** ${absUrl(base.url)}`);
    lines.push(`- **Profiles:** ${base.profiles.map((p) => `\`${p.id}\` (\`${p.source.repo_id}@${p.source.revision.slice(0, 12)}\`, ${p.files.length} file(s))`).join("; ")}`);
    lines.push("");
    for (const patch of base.patches) {
      lines.push(formatPatchSection(patch));
      lines.push("");
    }
  }

  lines.push("## Submitting a New Patch or Model");
  lines.push("");
  lines.push(`1. Every submission is a Git pull request adding authored records under \`models/<ns>/<model>/<patch-slug>/\` (see ${index.docsAt("CONTRIBUTING.md")}).`);
  lines.push("2. Only targeted weight patches (`replacement`) and native LoRA adapters (`lora`) are accepted — full-weight fine-tunes are out of scope.");
  lines.push("3. **Abliterated / uncensored patches:** The official Relux Works index does not host abliterated patches. Publish them in a third-party/community index repository (any Git host) using the exact same contract v1 layout, and connect that registry via `registries.toml`.");
  lines.push("4. Before opening a PR, run from `tool/`:");
  lines.push("");
  lines.push("```sh");
  lines.push("cargo run --locked -- generate ..");
  lines.push("cargo test --locked");
  lines.push("cargo run --locked -- check ..");
  lines.push("```");
  lines.push("");

  return lines.join("\n");
}
