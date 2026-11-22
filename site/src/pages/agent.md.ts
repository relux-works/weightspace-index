import { loadIndex } from "../lib/index.ts";
import { renderAgentMarkdown } from "../lib/agent-doc.ts";

export function GET() {
  const index = loadIndex();
  return new Response(renderAgentMarkdown(index), {
    headers: { "content-type": "text/markdown; charset=utf-8" },
  });
}
