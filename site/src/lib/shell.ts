// A small POSIX-shell highlighter for the copyable recipes. It only wraps
// tokens in spans; the text content stays byte-for-byte what gets copied.
const esc = (s: string) => s.replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
const span = (cls: string, t: string) => `<span class="sh-${cls}">${esc(t)}</span>`;

const TOKEN = /(\s+)|('[^']*'|"(?:\\.|[^"\\])*")|(\$\([^)]*\)|\$\{?\w+\}?)|(&&|\|\||\||;|\\$|<<-?\s*'?\w+'?|>>?)|([^\s'"|;&<>]+)|([\s\S])/g;

function line(text: string, expectCmd: boolean): string {
  let html = "";
  for (const m of text.matchAll(TOKEN)) {
    const [t, ws, str, variable, op, word] = m;
    if (ws) html += ws;
    else if (str) { html += span("str", t); expectCmd = false; }
    else if (variable) html += span("var", t);
    else if (op) { html += span("op", t); if (/^(&&|\|\||\||;)$/.test(op)) expectCmd = true; }
    else if (word && expectCmd && !/^\w+=/.test(word)) { html += span("cmd", t); expectCmd = false; }
    else if (word?.startsWith("-") && word.length > 1) html += span("flag", t);
    else if (word && /^(https?:\/\/)/.test(word)) html += span("url", t);
    else html += esc(t);
  }
  return html;
}

export function highlight(src: string): string {
  const out: string[] = [];
  let heredoc: string | null = null;
  let expectCmd = true;
  for (const text of src.split("\n")) {
    if (heredoc !== null) {
      out.push(text.trim() === heredoc ? span("op", text) : span("here", text));
      if (text.trim() === heredoc) { heredoc = null; expectCmd = true; }
      continue;
    }
    if (/^\s*#/.test(text)) { out.push(span("com", text)); expectCmd = true; continue; }
    const doc = /<<-?\s*'?(\w+)'?/.exec(text);
    out.push(line(text, expectCmd));
    if (doc) heredoc = doc[1];
    const trimmed = text.trimEnd();
    expectCmd = !trimmed.endsWith("\\") || /(&&|\|\|)\s*\\?$/.test(trimmed);
  }
  return out.join("\n");
}
