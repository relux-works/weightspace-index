import type { APIRoute } from "astro";
import { allVersions, loadIndex } from "../../lib/index.ts";
import { adapterBundle } from "../../lib/runtime-package.ts";

export function getStaticPaths() {
  return allVersions(loadIndex()).flatMap(version => {
    const bundle = adapterBundle(version);
    if (!bundle) return [];
    const asset = `${version.patch.base.address}/${bundle.filename}`;
    return [{ params: { asset }, props: { version } }];
  });
}

export const GET: APIRoute = ({ props }) => {
  const bytes = adapterBundle(props.version)!.bytes;
  return new Response(new Uint8Array(bytes), { headers: { "Content-Type": "application/gzip" } });
};
