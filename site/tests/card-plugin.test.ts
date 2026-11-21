// Patch cards are untrusted: check that the card plugins neutralize raw HTML,
// images and unsafe links, and map relative links onto site pages.
import assert from "node:assert/strict";
import { test } from "node:test";
import { pathToFileURL } from "node:url";
import path from "node:path";
import { markdownToHtml } from "satteri";
import { cardHast, cardMdast } from "../src/lib/card-plugin.ts";
import { REPO_ROOT } from "../src/lib/files.ts";

const card = "models/Qwen/Qwen3.5-0.8B/rikiyat-thinking-loops/README.md";
const render = (md: string, file = card) =>
  markdownToHtml(md, {
    fileURL: pathToFileURL(path.join(REPO_ROOT, file)),
    mdastPlugins: [cardMdast],
    hastPlugins: [cardHast],
  }).html;

test("raw HTML is shown as text, never rendered", () => {
  const html = render(`<script>alert(1)</script>\n\n<div onclick="x()">block</div>\n\nInline <img src=x onerror=alert(1)> and <b>b</b>.`);
  assert.doesNotMatch(html, /<script|<div|<img|<b>/);
  assert.match(html, /&lt;script&gt;/);
  assert.match(html, /&lt;img src=x onerror=alert\(1\)&gt;/);
});

test("images are never loaded", () => {
  const html = render(`![pixel](https://tracker.invalid/p.png) and ![ref][r]\n\n[r]: https://tracker.invalid/q.png`);
  assert.doesNotMatch(html, /<img/);
  assert.match(html, /image: pixel/);
});

test("unsafe schemes and missing targets lose their link", () => {
  const html = render(`[js](javascript:alert(1)) [data][d] [missing](nope/file.md)\n\n[d]: data:text/html,hi`);
  assert.doesNotMatch(html, /javascript:|data:text|nope\/file/);
});

test("relative links go to site pages or to GitHub at the pinned commit", () => {
  const html = render(`[v](v1/version.json) [base](../base.json) [doc](../../../../docs/structure.md#patch-forms) [ext](https://huggingface.co/x)`);
  assert.match(html, /<a href="\/models\/Qwen\/Qwen3.5-0.8B\/rikiyat-thinking-loops\/v1\/">v<\/a>/);
  assert.match(html, /href="\/models\/Qwen\/Qwen3.5-0.8B\/"/);
  assert.match(html, /href="https:\/\/github.com\/[^"]+\/blob\/[0-9a-f]{40}\/docs\/structure.md#patch-forms"/);
  assert.match(html, /href="https:\/\/huggingface.co\/x" target="_blank" rel="noopener noreferrer nofollow"/);
});

test("the card title heading is dropped and the rest move one level down", () => {
  const html = render(`# Qwen3.5-0.8B, edited by an LLM with no training (Weight-Space Programming)\n\n## Base model\n\ntext`);
  assert.doesNotMatch(html, /<h1/);
  assert.match(html, /<h3[^>]*>Base model<\/h3>/);
});

test("benchmark notes use repository-relative links and the same safe Markdown as cards", () => {
  const note = "bench/notes/example/Model/patch/v1.md";
  const target = `../../../../../${card.replace("README.md", "v1/version.json")}`;
  const html = render(`## Settings\n\n[release](${target}) [source](https://example.invalid/study) ![image](https://example.invalid/image.png) <b>markup</b>`, note);
  assert.match(html, /<h3[^>]*>Settings<\/h3>/);
  assert.match(html, /href="\/models\/Qwen\/Qwen3.5-0.8B\/rikiyat-thinking-loops\/v1\/"/);
  assert.match(html, /href="https:\/\/example.invalid\/study" target="_blank" rel="noopener noreferrer nofollow"/);
  assert.doesNotMatch(html, /<img|<b>/);
});
