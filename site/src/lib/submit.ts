// Pure contract-v1 submission validation and record generation for /submit/.
// Runs identically in unit tests and in the browser wizard.

export interface SubmissionInput {
  abliterated: boolean;
  communityRepoUrl: string;
  baseRepoId: string;
  baseRevision: string;
  profileId: string;
  slug: string;
  title: string;
  summary: string;
  kind: "lora" | "replacement";
  authorName: string;
  authorUrl?: string;
  submitter: string;
  license: string;
  tags: string;
  changelog: string;
  sourceRepoId: string;
  sourceRevision: string;
  sourceFileName: string;
  sourceSha256: string;
  sourceSizeBytes: number;
  configSha256?: string;
  configSizeBytes?: number;
  ggufRepoId?: string;
  ggufRevision?: string;
  ggufFileName?: string;
  ggufSha256?: string;
  ggufSizeBytes?: number;
  claimSourceUrl?: string;
  claimSourceTitle?: string;
  claimConditions?: string;
  claimBenchmark?: string;
  claimBaseline?: number;
  claimPatched?: number;
  claimUnit?: string;
  claimBetter?: "higher" | "lower";
}

export interface GeneratedFile {
  path: string;
  content: string;
}

export interface SubmissionBundle {
  errors: string[];
  targetRepoUrl: string;
  isCommunityTarget: boolean;
  files: GeneratedFile[];
  registryCommand: string | null;
  cliCommands: string;
  prUrl: string | null;
}

const REPO_ID_RE = /^[A-Za-z0-9_-][A-Za-z0-9_.-]*\/[A-Za-z0-9_-][A-Za-z0-9_.-]*$/;
const REV_RE = /^[a-f0-9]{40}$/;
const SHA_RE = /^[a-f0-9]{64}$/;
const SLUG_RE = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const ID_RE = /^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$/;
const HANDLE_RE = /^[A-Za-z0-9](?:-?[A-Za-z0-9])*$/;
const OFFICIAL_REPO = "https://github.com/relux-works/weightspace-index";

export function parseTags(raw: string): string[] {
  return raw
    .split(/[,\s]+/)
    .map((t) => t.trim().toLowerCase())
    .filter(Boolean);
}

export function validateSubmission(input: SubmissionInput): string[] {
  const errors: string[] = [];
  if (!REPO_ID_RE.test(input.baseRepoId.trim())) {
    errors.push("Base model must be in namespace/model format (e.g. Qwen/Qwen3.8-27B).");
  }
  if (!REV_RE.test(input.baseRevision.trim())) {
    errors.push("Base revision must be a 40-character lowercase hex commit SHA.");
  }
  if (!ID_RE.test(input.profileId.trim())) {
    errors.push("Profile ID must be lowercase alphanumeric (e.g. bf16, mxfp4).");
  }
  if (!SLUG_RE.test(input.slug.trim())) {
    errors.push("Patch slug must be lowercase kebab-case (e.g. harness-hub-pi).");
  }
  if (!input.title.trim()) errors.push("Patch title is required.");
  if (!input.summary.trim()) errors.push("One-line summary is required.");
  if (!input.authorName.trim()) errors.push("At least one author name is required.");
  if (!HANDLE_RE.test(input.submitter.trim())) {
    errors.push("Submitter handle must be a valid GitHub/Git handle.");
  }
  if (!input.license.trim()) errors.push("License identifier is required (e.g. Apache-2.0).");
  if (!REPO_ID_RE.test(input.sourceRepoId.trim())) {
    errors.push("Upstream artifact Hugging Face repo_id is required (namespace/repo).");
  }
  if (!REV_RE.test(input.sourceRevision.trim())) {
    errors.push("Artifact revision must be a 40-character lowercase hex commit SHA.");
  }
  if (!input.sourceFileName.trim()) {
    errors.push("Primary artifact filename is required.");
  } else if (input.kind === "replacement" && !input.sourceFileName.trim().endsWith(".safetensors")) {
    errors.push("Replacement source must be a .safetensors file.");
  } else if (
    input.kind === "lora" &&
    input.sourceFileName.trim() !== "adapter_model.safetensors" &&
    !input.sourceFileName.trim().endsWith(".gguf")
  ) {
    errors.push("LoRA source must be adapter_model.safetensors (PEFT) or a .gguf adapter.");
  }
  if (!SHA_RE.test(input.sourceSha256.trim())) {
    errors.push("Primary artifact SHA-256 must be 64 lowercase hex characters.");
  }
  if (!Number.isInteger(input.sourceSizeBytes) || input.sourceSizeBytes < 1) {
    errors.push("Primary artifact byte size must be a positive integer.");
  }
  if (input.kind === "lora" && input.sourceFileName.trim() === "adapter_model.safetensors") {
    if (!SHA_RE.test((input.configSha256 ?? "").trim())) {
      errors.push("PEFT LoRA submissions require adapter_config.json SHA-256 (64 lowercase hex chars).");
    }
    if (!Number.isInteger(input.configSizeBytes) || (input.configSizeBytes ?? 0) < 1) {
      errors.push("PEFT LoRA submissions require adapter_config.json byte size.");
    }
  }
  if (input.abliterated) {
    const url = input.communityRepoUrl.trim();
    if (url && /^https?:\/\/github\.com\/relux-works\/weightspace-index(?:\.git)?\/?$/i.test(url)) {
      errors.push(
        "Abliterated / uncensored patches cannot target relux-works/weightspace-index. Enter a third-party community index repository URL.",
      );
    }
  }
  for (const tag of parseTags(input.tags)) {
    if (!SLUG_RE.test(tag)) {
      errors.push(`Tag "${tag}" must be lowercase kebab-case.`);
    }
  }
  if (input.claimSourceUrl?.trim()) {
    if (!input.claimSourceUrl.trim().startsWith("https://")) {
      errors.push("Claim source URL must start with https://.");
    }
    if (!input.claimBenchmark?.trim()) {
      errors.push("Claim benchmark name is required when a claim source URL is given.");
    }
    if (typeof input.claimBaseline !== "number" || Number.isNaN(input.claimBaseline)) {
      errors.push("Claim baseline score must be a number.");
    }
    if (typeof input.claimPatched !== "number" || Number.isNaN(input.claimPatched)) {
      errors.push("Claim patched score must be a number.");
    }
  }
  return errors;
}

export function buildSubmissionBundle(input: SubmissionInput): SubmissionBundle {
  const errors = validateSubmission(input);
  const baseRepo = input.baseRepoId.trim() || "Qwen/Qwen3.8-27B";
  const slug = input.slug.trim() || "my-patch";
  const tags = parseTags(input.tags);
  if (input.abliterated && !tags.includes("abliterated")) {
    tags.push("abliterated");
  }

  const isCommunityTarget = input.abliterated || Boolean(input.communityRepoUrl.trim());
  const targetRepoUrl = isCommunityTarget
    ? input.communityRepoUrl.trim() || "https://github.com/<community-org>/weightspace-community-index"
    : OFFICIAL_REPO;

  const patchDir = `models/${baseRepo}/${slug}`;
  const patchJson = {
    contract_version: 1,
    title: input.title.trim(),
    summary: input.summary.trim(),
    kind: input.kind,
    authors: [
      input.authorUrl?.trim()
        ? { name: input.authorName.trim(), url: input.authorUrl.trim() }
        : { name: input.authorName.trim() },
    ],
    submitter: input.submitter.trim(),
    license: input.license.trim(),
    tags,
    versions: [{ version: 1, status: "active" }],
  };

  const versionFiles: Record<string, unknown>[] = [
    {
      name: input.sourceFileName.trim(),
      role: "source",
      sha256: input.sourceSha256.trim(),
      size_bytes: input.sourceSizeBytes,
      locators: [
        {
          kind: "hf",
          repo_type: "model",
          repo_id: input.sourceRepoId.trim(),
          revision: input.sourceRevision.trim(),
          path: input.sourceFileName.trim(),
        },
      ],
    },
  ];

  if (input.kind === "lora" && input.sourceFileName.trim() === "adapter_model.safetensors") {
    versionFiles.push({
      name: "adapter_config.json",
      role: "source",
      sha256: (input.configSha256 ?? "").trim(),
      size_bytes: input.configSizeBytes ?? 0,
      locators: [
        {
          kind: "hf",
          repo_type: "model",
          repo_id: input.sourceRepoId.trim(),
          revision: input.sourceRevision.trim(),
          path: "adapter_config.json",
        },
      ],
    });
  }

  if (input.ggufRepoId?.trim() && input.ggufFileName?.trim() && input.ggufSha256?.trim()) {
    versionFiles.push({
      name: input.ggufFileName.trim(),
      role: "derived",
      sha256: input.ggufSha256.trim(),
      size_bytes: input.ggufSizeBytes ?? 0,
      locators: [
        {
          kind: "hf",
          repo_type: "model",
          repo_id: input.ggufRepoId.trim(),
          revision: (input.ggufRevision ?? input.sourceRevision).trim(),
          path: input.ggufFileName.trim(),
        },
      ],
      derived_from: [input.sourceSha256.trim()],
      tool: { name: "llama.cpp convert_lora_to_gguf", version: "published-upstream" },
    });
  }

  const versionJson = {
    contract_version: 1,
    changelog: input.changelog.trim() || `Initial import of ${input.sourceRepoId.trim()}.`,
    target_profile: input.profileId.trim(),
    files: versionFiles,
  };

  const files: GeneratedFile[] = [
    { path: `${patchDir}/patch.json`, content: JSON.stringify(patchJson, null, 2) + "\n" },
    { path: `${patchDir}/v1/version.json`, content: JSON.stringify(versionJson, null, 2) + "\n" },
    {
      path: `${patchDir}/README.md`,
      content: [
        `# ${input.title.trim()}`,
        "",
        input.summary.trim(),
        "",
        "## Upstream",
        "",
        `- Base: \`${baseRepo}@${input.baseRevision.trim()}\``,
        `- Artifact: [${input.sourceRepoId.trim()}](https://huggingface.co/${input.sourceRepoId.trim()})`,
        "",
      ].join("\n"),
    },
  ];

  if (input.claimSourceUrl?.trim() && input.claimBenchmark?.trim()) {
    const claimJson = {
      contract_version: 1,
      source: {
        url: input.claimSourceUrl.trim(),
        title: input.claimSourceTitle?.trim() || `${input.sourceRepoId.trim()} model card`,
      },
      conditions: input.claimConditions?.trim() || "Author-reported paired evaluation against the pinned base model.",
      measures: [
        {
          benchmark: input.claimBenchmark.trim(),
          unit: input.claimUnit?.trim() || "%",
          baseline: input.claimBaseline ?? 0,
          patched: input.claimPatched ?? 0,
          better: input.claimBetter ?? "higher",
        },
      ],
    };
    files.push({
      path: `${patchDir}/v1/claims/author-report.json`,
      content: JSON.stringify(claimJson, null, 2) + "\n",
    });
  }

  const cloneUrl = targetRepoUrl.endsWith(".git") ? targetRepoUrl : `${targetRepoUrl}.git`;
  const registryCommand = isCommunityTarget
    ? `wsp registry add community ${cloneUrl}\nwsp registry pin community main`
    : null;

  const cliCommands = [
    `git clone ${cloneUrl}`,
    `cd $(basename ${targetRepoUrl})`,
    `git checkout -b add/${slug}`,
    `# Write the generated files under ${patchDir}/ and copy upstream license into ${patchDir}/notices/`,
    `(cd tool && cargo run --locked -- generate .. && cargo run --locked -- check ..)`,
    `git add models/ lookup/`,
    `git commit -S -m "models: add ${baseRepo}/${slug}"`,
    `git push -u origin add/${slug}`,
  ].join("\n");

  const ghMatch = /^https:\/\/github\.com\/([^/]+\/[^/.]+?)(?:\.git)?\/?$/i.exec(targetRepoUrl);
  let prUrl: string | null = null;
  if (ghMatch && !targetRepoUrl.includes("<community-org>")) {
    const title = encodeURIComponent(`Add ${baseRepo}/${slug}: ${input.title.trim()}`);
    const body = encodeURIComponent(
      [
        `### Submission`,
        `- Base: \`${baseRepo}@${input.baseRevision.trim()}\` (profile \`${input.profileId.trim()}\`)`,
        `- Patch: \`${slug}\` (\`${input.kind}\`)`,
        `- Upstream: https://huggingface.co/${input.sourceRepoId.trim()} (rev \`${input.sourceRevision.trim()}\`)`,
        `- License: \`${input.license.trim()}\``,
        `- Abliterated: ${input.abliterated ? "yes (community index)" : "no"}`,
        "",
        "Generated files from weightspaceindex.com/submit/:",
        ...files.map((f) => `- \`${f.path}\``),
      ].join("\n"),
    );
    prUrl = `https://github.com/${ghMatch[1]}/issues/new?title=${title}&body=${body}`;
  }

  return {
    errors,
    targetRepoUrl,
    isCommunityTarget,
    files,
    registryCommand,
    cliCommands,
    prUrl,
  };
}
