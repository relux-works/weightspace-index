// The build must stop on a missing record, invalid JSON, or a contract
// version this site does not read.
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

const root = mkdtempSync(path.join(tmpdir(), "weightspace-site-test-"));
process.env.WS_REPO_ROOT = root;
const { readRecord, IndexError } = await import("../src/lib/files.ts");

const put = (name: string, text: string) => writeFileSync(path.join(root, name), text);

test("a missing record stops the build", () => {
  assert.throws(() => readRecord("absent.json"), (e: unknown) => e instanceof IndexError && /absent\.json/.test(e.message));
});

test("invalid JSON stops the build", () => {
  put("broken.json", "{");
  assert.throws(() => readRecord("broken.json"), IndexError);
});

test("an unsupported contract_version stops the build with a clear error", () => {
  put("v2.json", JSON.stringify({ contract_version: 2 }));
  assert.throws(() => readRecord("v2.json"), (e: unknown) => e instanceof IndexError && /unsupported contract_version 2/.test(e.message));
});

test("a record without contract_version stops the build", () => {
  put("none.json", JSON.stringify({}));
  assert.throws(() => readRecord("none.json"), IndexError);
});

test("contract version 1 is read", () => {
  put("v1.json", JSON.stringify({ contract_version: 1, x: 1 }));
  assert.deepEqual(readRecord("v1.json"), { contract_version: 1, x: 1 });
});
