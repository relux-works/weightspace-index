/** Filter the full directory before taking a page so later entries remain discoverable. */
export function filteredPage(rows: string[], query: string, limit: number) {
  const words = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const matches = rows.flatMap((text, i) => words.every(w => text.includes(w)) ? [i] : []);
  return { visible: matches.slice(0, limit), matched: matches.length };
}
