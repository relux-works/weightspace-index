import test from "node:test";
import assert from "node:assert/strict";
import { loadIndex } from "../src/lib/index.ts";
import { renderAgentMarkdown, renderLlmsTxt } from "../src/lib/agent-doc.ts";

test("llms.txt lists live bases, commit and agent.md pointer", () => {
  const index = loadIndex();
  const txt = renderLlmsTxt(index);
  assert.match(txt, /^# Weightspace Index/m);
  assert.ok(txt.includes(index.commit), "includes pinned commit");
  assert.ok(txt.includes("/agent.md"), "links to agent.md");
  assert.ok(txt.includes("/search.json"), "links to search.json");
  for (const base of index.bases) {
    assert.ok(txt.includes(base.address), `lists base ${base.address}`);
  }
});

test("agent.md is a self-contained catalog with digests, registries and policies", () => {
  const index = loadIndex();
  const md = renderAgentMarkdown(index);
  assert.match(md, /^# Weightspace Index — Complete Agent Reference/m);
  assert.ok(md.includes(index.commit), "includes pinned commit");
  assert.ok(md.includes("registries.toml"), "documents custom/private registries");
  assert.ok(md.includes("Abliterated"), "documents abliterated community registry policy");
  for (const base of index.bases) {
    assert.ok(md.includes(base.address), `includes base ${base.address}`);
    for (const patch of base.patches) {
      assert.ok(md.includes(`${base.address}/${patch.slug}`), `includes patch ${patch.slug}`);
      const active = patch.versions.find((v) => v.status === "active");
      if (active) {
        for (const f of active.files) {
          assert.ok(md.includes(`sha256:${f.sha256}`), `includes digest for ${f.name}`);
        }
      }
    }
  }
});
