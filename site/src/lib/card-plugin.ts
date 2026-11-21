// Sätteri plugins for repository Markdown shown on patch pages.
// - raw HTML is shown as text, never rendered;
// - images are never loaded (that would be a third-party request); they
//   become links;
// - relative links go to the matching site page, or to the file on GitHub at
//   the pinned commit; links to files that do not exist lose their link;
// - the card's own title heading is dropped (the page shows it) and the other
//   headings move one level down under the page's heading.
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineHastPlugin, defineMdastPlugin } from "satteri";
import { DEMO, GITHUB_REPO, PIN_PATHS, PREFIX, href, seg } from "./config.ts";
import { REPO_ROOT, exists, isDir, pinCommit, readRecord } from "./files.ts";
import { externalLinkAttrs } from "./links.ts";


const CARD = /^models\/[^/]+\/[^/]+\/[^/]+\/README\.md$/;
const BENCHMARK_NOTE = /^bench\/notes\/[^/]+\/[^/]+\/[^/]+\/v[1-9][0-9]*\.md$/;
const SAFE_ABS = /^(https?:\/\/|mailto:)/i;
const HAS_SCHEME = /^[a-z][a-z0-9+.-]*:/i;

/** Resolves links in a patch card or benchmark note to their published targets. */
export function resolveCardLink(cardPath: string, target: string): string | null {
  const t = target.trim();
  if (!t) return null;
  if (HAS_SCHEME.test(t) || t.startsWith("//")) return SAFE_ABS.test(t) ? t : null;
  if (t.startsWith("#")) return t;
  const [rel, frag] = splitFragment(t);
  let p: string;
  try {
    p = path.posix.normalize(path.posix.join(path.posix.dirname(cardPath), decodeURI(rel)));
  } catch {
    return null;
  }
  if (p.startsWith("../") || p === ".." || path.posix.isAbsolute(p)) return null;
  p = p.replace(/\/$/, "");
  if (!exists(p)) return null;
  const page = sitePage(p);
  if (page) return page + frag;
  const { commit } = pinCommit(PIN_PATHS);
  const view = isDir(p) ? "tree" : "blob";
  return `https://github.com/${GITHUB_REPO}/${view}/${commit}/${seg(p)}${frag}`;
}

function splitFragment(t: string): [string, string] {
  const i = t.indexOf("#");
  return i < 0 ? [t, ""] : [t.slice(0, i), t.slice(i)];
}

/** The site page for an index path, following the fixed layout of docs/structure.md. */
function sitePage(p: string): string | null {
  const models = DEMO && p.startsWith("docs/example/") ? "docs/example/models/" : `${PREFIX}models/`;
  if (!p.startsWith(models)) return null;
  const s = p.slice(models.length).split("/");
  const page = (n: number) => href(`models/${seg(s.slice(0, n).join("/"))}/`);
  const last = s.length;
  const is = (n: number, files: string[]) => last === n || (last === n + 1 && files.includes(s[n]));
  if (last < 2) return null;
  if (is(2, ["base.json", "README.md", "index.json"])) return page(2);
  if (s[2] === "runs") return last >= 4 && is(4, ["run.json"]) ? page(4) : null;
  if (is(3, ["patch.json", "README.md"])) return page(3);
  if (!/^v[1-9][0-9]*$/.test(s[3] ?? "")) return null;
  if (is(4, ["version.json"])) return page(4);
  if (s[4] === "runs" && is(6, ["run.json"])) return page(6);
  return null;
}

function markdownPathOf(fileURL: URL | undefined): string | null {
  if (!fileURL) return null;
  const rel = path.relative(REPO_ROOT, fileURLToPath(fileURL)).split(path.sep).join("/");
  const prefix = DEMO && rel.startsWith("docs/example/") ? "docs/example/" : PREFIX;
  if (!rel.startsWith(prefix)) return null;
  const inCollection = rel.slice(prefix.length);
  return CARD.test(inCollection) || BENCHMARK_NOTE.test(inCollection) ? rel : null;
}

export const cardMdast = (ctx: { fileURL: URL | undefined }) => {
  const markdown = markdownPathOf(ctx.fileURL);
  if (!markdown) return null;
  const patchJson = path.posix.join(path.posix.dirname(markdown), "patch.json");
  const title: string | undefined = markdown.endsWith("/README.md") && exists(patchJson) ? readRecord<any>(patchJson, "patch record").title : undefined;
  const resolve = (url: string) => resolveCardLink(markdown, url);
  let seenHeading = false;
  const inlineParents = new Set(["paragraph", "heading", "tableCell", "emphasis", "strong", "delete", "link", "linkReference"]);
  return defineMdastPlugin({
    name: "weightspace-card",
    html(node, v) {
      const parent = v.parent(node);
      const inline = parent ? inlineParents.has(parent.type) : false;
      v.replaceNode(node, inline ? { type: "text", value: node.value } : { type: "code", value: node.value, lang: null, meta: null });
    },
    image(node, v) {
      const label = { type: "text" as const, value: `image: ${node.alt || node.url}` };
      const url = resolve(node.url);
      v.replaceNode(node, url ? { type: "link", url, title: null, children: [label] } : label);
    },
    imageReference(node, v) {
      v.replaceNode(node, { type: "text", value: `image: ${node.alt ?? node.label ?? ""}` });
    },
    link(node, v) {
      const url = resolve(node.url);
      if (url) v.setField(node, "url", url);
      else v.replaceNode(node, { type: "text", value: v.textContent(node) });
    },
    definition(node, v) {
      v.setField(node, "url", resolve(node.url) ?? "#");
    },
    heading(node, v) {
      const first = !seenHeading;
      seenHeading = true;
      if (first && node.depth === 1 && title && v.textContent(node).trim() === title.trim()) {
        v.removeNode(node);
        return;
      }
      v.setField(node, "depth", Math.min(6, node.depth + 1) as 1 | 2 | 3 | 4 | 5 | 6);
    },
  });
};

/** External web links in repository Markdown open in a new tab. */
export const cardHast = (ctx: { fileURL: URL | undefined }) =>
  markdownPathOf(ctx.fileURL)
    ? defineHastPlugin({
        name: "weightspace-card-links",
        element: {
          filter: ["a"],
          visit(node, v) {
            const h = node.properties?.href;
            const { target } = typeof h === "string" ? externalLinkAttrs(h) : {};
            if (target) {
              v.setProperty(node, "target", target);
              v.setProperty(node, "rel", ["noopener", "noreferrer", "nofollow"]);
              v.setProperty(node, "className", ["ext"]);
            }
          },
        },
      })
    : null;
