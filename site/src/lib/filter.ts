/** Lowercased query words; a row matches when its text contains every word. */
export function queryWords(query: string): string[] {
  return query.trim().toLowerCase().split(/\s+/).filter(Boolean);
}

export function matchesAll(text: string, words: string[]): boolean {
  return words.every(w => text.includes(w));
}

/** Filter the full directory before taking a page so later entries remain discoverable. */
export function filteredPage(rows: string[], query: string, limit: number) {
  const words = queryWords(query);
  const matches = rows.flatMap((text, i) => matchesAll(text, words) ? [i] : []);
  return { visible: matches.slice(0, limit), matched: matches.length };
}
