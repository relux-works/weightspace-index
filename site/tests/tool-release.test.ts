import assert from "node:assert/strict";
import { test } from "node:test";
import { toolDownload } from "../src/lib/commands.ts";
import { assetUrl, checkRelease, toolRelease } from "../src/lib/tool-release.ts";

test("the recorded release follows the release workflow's archive names", () => {
  assert.doesNotThrow(() => checkRelease(toolRelease));
  const bad = { ...toolRelease, assets: [{ ...toolRelease.assets[0], name: "weightspace.tar.gz" }] };
  assert.throws(() => checkRelease(bad), /does not match/);
});

test("release downloads point at the tagged GitHub asset", () => {
  const asset = toolRelease.assets[0];
  assert.equal(assetUrl("o/r", toolRelease, asset), `https://github.com/o/r/releases/download/${toolRelease.tag}/${asset.name}`);
});

// The same platforms with a bare binary, as releases after v0.1.0-alpha.1 publish them.
const withBinaries = {
  ...toolRelease,
  assets: toolRelease.assets.map(a => ({
    ...a,
    binary: { name: `weightspace-${toolRelease.tag}-${a.target}${a.os === "windows" ? ".exe" : ""}`, size: 1, sha256: "b".repeat(64) },
  })),
};

test("bare binaries must be named like their archive without the extension", () => {
  assert.doesNotThrow(() => checkRelease(withBinaries));
  const bad = { ...withBinaries, assets: [{ ...withBinaries.assets[0], binary: { ...withBinaries.assets[0].binary, name: "weightspace" } }] };
  assert.throws(() => checkRelease(bad), /should be/);
});

test("every install command checks the digest before the file is unpacked or installed", () => {
  for (const release of [toolRelease, withBinaries]) {
    for (const asset of release.assets) {
      const file = asset.binary ?? asset;
      const lines = toolDownload("o/r", release, asset).text.split("\n");
      assert.ok(lines[0].includes(file.name), `${asset.os}: downloads ${file.name}`);
      const check = lines.findIndex(l => l.toLowerCase().includes(file.sha256));
      const next = lines.findIndex(l => /^(tar|Expand-Archive|sudo install|# Rename)/.test(l));
      assert.ok(check > 0 && next > check, `${asset.os}: digest check comes first`);
    }
  }
});
