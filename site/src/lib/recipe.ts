import type { RuntimeTab } from "./runtime.ts";

export interface RecipeFile {
  name: string;
  href: string;
  sha256: string;
}

const quote = (s: string) => "'" + s.replaceAll("'", "'\\''") + "'";

export function downloadCommand(file: Pick<RecipeFile, "href" | "name">, origin: string) {
  return "curl -fL " + quote(new URL(file.href, origin).href) + " -o " + quote(file.name);
}

/** The static version starts with the file from the download button. */
export function localRecipe(tab: Pick<RuntimeTab, "name" | "setup" | "setupLabel" | "prepare" | "command">, file?: RecipeFile | null) {
  const lines = ["sh <<'WEIGHTSPACE'", "set -e"];
  if (file) lines.push("# Verify the downloaded file", "printf '%s  %s\\n' " + quote(file.sha256) + " " + quote(file.name) + " | shasum -a 256 -c -");
  if (tab.setup) lines.push("", "# " + (tab.setupLabel ?? "Prepare the pinned base model"), tab.setup);
  if (tab.prepare) lines.push("", tab.name === "Weightspace" ? "# Fetch and verify base weights" : "# Prepare the adapter", tab.prepare);
  lines.push("", "# " + (tab.name === "Weightspace" ? "Apply the replacement" : "Run with the adapter"), tab.command, "WEIGHTSPACE");
  return lines.join("\n");
}
