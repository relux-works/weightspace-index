import assert from "node:assert/strict";
import { test } from "node:test";
import { downloadCommand, localRecipe } from "../src/lib/recipe.ts";

const file = { href: "/catalog/asset", name: "adapter's file", sha256: "a".repeat(64) };
test("one-paste downloads resolve the hosting origin and save the exact local filename", () => {
  assert.equal(downloadCommand(file, "https://test.invalid"), "curl -fL 'https://test.invalid/catalog/asset' -o 'adapter'\\''s file'");
  assert.match(downloadCommand({ ...file, href: "https://host.invalid/file" }, "http://localhost"), /https:\/\/host\.invalid\/file/);
});

test("the offline recipe verifies the downloaded file before preparing or launching it", () => {
  const recipe = localRecipe({ name: "Runtime", setup: "prepare-base", prepare: "unpack-adapter", command: "launch-adapter" }, file);
  assert.ok(recipe.startsWith("sh <<'WEIGHTSPACE'\nset -e\n"));
  assert.ok(recipe.endsWith("\nWEIGHTSPACE"));
  assert.match(recipe, /shasum -a 256 -c -/);
  assert.ok(recipe.indexOf("shasum") < recipe.indexOf("prepare-base"));
  assert.ok(recipe.indexOf("prepare-base") < recipe.indexOf("launch-adapter"));
  assert.ok(recipe.indexOf("prepare-base") < recipe.indexOf("unpack-adapter"));
  assert.ok(recipe.indexOf("unpack-adapter") < recipe.indexOf("launch-adapter"));
  assert.ok(!recipe.includes("curl"));
});
