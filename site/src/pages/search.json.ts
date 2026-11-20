import { loadIndex } from "../lib/index.ts";
import { searchEntries } from "../lib/search.ts";

export function GET() {
  const index = loadIndex();
  return new Response(JSON.stringify({ commit: index.commit, entries: searchEntries(index) }), {
    headers: { "content-type": "application/json" },
  });
}
