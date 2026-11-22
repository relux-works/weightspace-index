import { loadIndex } from "../lib/index.ts";
import { renderLlmsTxt } from "../lib/agent-doc.ts";

export function GET() {
  const index = loadIndex();
  return new Response(renderLlmsTxt(index), {
    headers: { "content-type": "text/plain; charset=utf-8" },
  });
}
