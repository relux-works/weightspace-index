import test from "node:test";
import assert from "node:assert/strict";
import { buildSubmissionBundle, validateSubmission, type SubmissionInput } from "../src/lib/submit.ts";

const VALID_HEX_40 = "a".repeat(40);
const VALID_HEX_64 = "b".repeat(64);
const VALID_CFG_64 = "c".repeat(64);

function sample(overrides: Partial<SubmissionInput> = {}): SubmissionInput {
  return {
    abliterated: false,
    communityRepoUrl: "",
    baseRepoId: "Qwen/Qwen3.8-27B",
    baseRevision: VALID_HEX_40,
    profileId: "bf16",
    slug: "custom-agent-lora",
    title: "Custom agent LoRA",
    summary: "Improves tool call formatting.",
    kind: "lora",
    authorName: "Example Lab",
    submitter: "ivanopcode",
    license: "Apache-2.0",
    tags: "coding-agent, lora",
    changelog: "Initial import.",
    sourceRepoId: "example/custom-agent-lora",
    sourceRevision: VALID_HEX_40,
    sourceFileName: "adapter_model.safetensors",
    sourceSha256: VALID_HEX_64,
    sourceSizeBytes: 12345678,
    configSha256: VALID_CFG_64,
    configSizeBytes: 1024,
    ...overrides,
  };
}

test("valid PEFT LoRA submission generates contract v1 files targeting official repo", () => {
  const bundle = buildSubmissionBundle(sample());
  assert.deepEqual(bundle.errors, []);
  assert.equal(bundle.isCommunityTarget, false);
  assert.equal(bundle.targetRepoUrl, "https://github.com/relux-works/weightspace-index");
  assert.equal(bundle.registryCommand, null);
  assert.ok(bundle.prUrl?.startsWith("https://github.com/relux-works/weightspace-index/issues/new"));
  assert.equal(bundle.files.length, 3);
  const patch = JSON.parse(bundle.files[0].content);
  assert.equal(patch.contract_version, 1);
  assert.equal(patch.kind, "lora");
  const version = JSON.parse(bundle.files[1].content);
  assert.equal(version.files.length, 2);
});

test("abliterated checkbox blocks official Relux Works repo and routes to community index", () => {
  const blocked = validateSubmission(
    sample({
      abliterated: true,
      communityRepoUrl: "https://github.com/relux-works/weightspace-index",
    }),
  );
  assert.ok(blocked.some((e) => e.includes("relux-works/weightspace-index")));

  const community = buildSubmissionBundle(
    sample({
      abliterated: true,
      communityRepoUrl: "https://gitlab.example.org/community/weightspace-abliterated",
    }),
  );
  assert.deepEqual(community.errors, []);
  assert.equal(community.isCommunityTarget, true);
  assert.ok(community.registryCommand?.includes("wsp registry add community"));
  const patch = JSON.parse(community.files[0].content);
  assert.ok(patch.tags.includes("abliterated"));
});

test("submission rejects malformed digests and revisions", () => {
  const errors = validateSubmission(sample({ baseRevision: "short", sourceSha256: "not-a-sha" }));
  assert.ok(errors.length >= 2);
});
