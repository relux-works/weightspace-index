import type { APIRoute } from "astro";
import { allVersions, loadIndex } from "../../../lib/index.ts";
import { readBytes } from "../../../lib/files.ts";

export function getStaticPaths() {
  const paths = allVersions(loadIndex()).flatMap(v => v.files.filter(f => f.repoPath).map(file => ({
    params: { digest: file.sha256, name: file.name }, props: { path: file.repoPath! }
  })));
  return [...new Map(paths.map(p => [p.params.digest + "/" + p.params.name, p])).values()];
}

export const GET: APIRoute = ({ props }) => new Response(new Uint8Array(readBytes(props.path)), {
  headers: { "Content-Type": "application/octet-stream" }
});
