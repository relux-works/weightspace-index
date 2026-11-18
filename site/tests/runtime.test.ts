import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { test } from "node:test";
import { gunzipSync } from "node:zlib";
import { makePeftFiles, tarGzip } from "../src/lib/runtime-package.ts";
import { runtimeGuide } from "../src/lib/runtime.ts";
import type { FileEntry, Version } from "../src/lib/index.ts";

const moduleName = "model.language_model.layers.2.mlp.down_proj";
function factors(rankB = 2) {
  const header = {
    [moduleName + ".weight.lora_A"]: { dtype: "F32", shape: [2, 3], data_offsets: [0, 24] },
    [moduleName + ".weight.lora_B"]: { dtype: "F32", shape: [3, rankB], data_offsets: [24, 24 + rankB * 12] },
  };
  const json = Buffer.from(JSON.stringify(header));
  const length = Buffer.alloc(8);
  length.writeBigUInt64LE(BigInt(json.length));
  return Buffer.concat([length, json, Buffer.alloc(24 + rankB * 12, 8)]);
}
const source = { repo_id: "org/Model", revision: "abcd" };
test("PEFT conversion preserves the exact factor data and configures unit scaling on the edited modules", () => {
  const peft = makePeftFiles(factors(), source)!;
  assert.deepEqual(peft.config.target_modules, [moduleName]);
  assert.equal(peft.config.lora_alpha / peft.config.r, 1);
  assert.equal(peft.config.revision, "abcd");
  const length = Number(peft.weights.readBigUInt64LE());
  assert.equal(length % 8, 0);
  const header = JSON.parse(peft.weights.subarray(8, 8 + length).toString());
  assert.deepEqual(header["base_model.model." + moduleName + ".lora_B.weight"].shape, [3, 2]);
  assert.deepEqual(peft.weights.subarray(8 + length), Buffer.alloc(48, 8));
});
test("unmatched factor ranks do not produce a runtime bundle", () => {
  assert.equal(makePeftFiles(factors(1), source), null);
});
test("the compressed package can be extracted with tar and preserves binary files", () => {
  const input = Buffer.from([0, 255, 1, 128]);
  const tar = tarGzip({ "adapter_model.safetensors": input, "notices/ATTRIBUTION.md": Buffer.from("credit") });
  assert.deepEqual(execFileSync("tar", ["-xzOf", "-", "adapter_model.safetensors"], { input: tar }), input);
  assert.equal(execFileSync("tar", ["-xzOf", "-", "notices/ATTRIBUTION.md"], { input: tar }).toString(), "credit");
});
test("download archives omit the build user's identity", () => {
  const header = gunzipSync(tarGzip({ "adapter_config.json": Buffer.from("{}") }));
  assert.equal(parseInt(header.toString("ascii", 108, 116), 8), 0);
  assert.equal(parseInt(header.toString("ascii", 116, 124), 8), 0);
});
function file(name: string): FileEntry {
  return { name, role: "source", sha256: "b".repeat(64), size_bytes: 30, digest: "recorded", links: [{ host: "Hugging Face", href: "https://huggingface.co/" + name }] };
}
function version(files: FileEntry[]): Version {
  return { patch: { kind: "lora", slug: "patch", base: { address: source.repo_id, model: "Model" } }, profile: { id: "bf16", source, files: [{path:"model.safetensors"}] }, label: "v1", files } as Version;
}
test("native PEFT instructions acquire both companion files and pin the base revision", () => {
  const guide = runtimeGuide(version([file("adapter_model.safetensors"), file("adapter_config.json")]));
  const vllm = guide.tabs.find(t => t.name === "vLLM")!;
  assert.match(vllm.command, /adapter_config.json/);
  assert.match(vllm.command, /adapter_model.safetensors/);
  assert.match(vllm.command, /--revision abcd/);
  assert.match(vllm.command, /--max-lora-rank/);
});
test("a GGUF-only adapter has a llama.cpp command and no PEFT load command", () => {
  const guide = runtimeGuide(version([file("adapter.gguf")]));
  assert.equal(guide.download?.name, "adapter.gguf");
  assert.match(guide.tabs.find(t => t.name === "llama.cpp")!.command, /--lora/);
  assert.ok(!guide.tabs.some(t => t.name === "vLLM"));
});
test("unpaired weights are downloadable as files but never get runtime commands", () => {
  assert.deepEqual(runtimeGuide(version([file("adapter_model.safetensors")])).tabs, []);
});
test("a GGUF-only base profile does not suggest loading an HF checkpoint", () => {
  const v = version([file("adapter_model.safetensors"), file("adapter_config.json")]);
  v.profile.files = [{ path: "base-q4.gguf", size_bytes: 24, sha256: "c".repeat(64) }];
  assert.deepEqual(runtimeGuide(v).tabs, []);
});
