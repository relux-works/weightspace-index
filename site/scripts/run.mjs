#!/usr/bin/env node
// Entry point for every site command (see package.json scripts).
//   build | dev  run Astro over the index checkout. Non-dev builds refuse index
//                files that differ from HEAD (src/lib/index.ts). Set
//                WS_RUN_INDEX_CHECK=1 to also run `weightspace check` via Cargo
//                on a `git archive` snapshot of HEAD first (schema, layout,
//                references, digests, generated files; CI runs this in checks.yml).
//   preview      serves the last build.
//   test         runs the site's own tests.
// Astro's telemetry is always disabled.
import { spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const [cmd, ...rest] = process.argv.slice(2);
const env = { ...process.env, ASTRO_TELEMETRY_DISABLED: "1" };
const astro = fileURLToPath(new URL("../node_modules/astro/bin/astro.mjs", import.meta.url));

/** Runs a command; returns its exit status (1 when it could not start). */
function status(bin, args, cwd) {
  const r = spawnSync(bin, args, { stdio: "inherit", env, cwd });
  if (r.error) {
    console.error(`[weightspace site] could not run ${bin}: ${r.error.message}`);
    return 1;
  }
  return r.status ?? 1;
}

function run(bin, args, cwd) {
  const code = status(bin, args, cwd);
  if (code !== 0) process.exit(code);
}

function checkSnapshot() {
  const repo = fileURLToPath(new URL("../../", import.meta.url));
  const tmp = mkdtempSync(path.join(tmpdir(), "weightspace-site-check-"));
  const tree = path.join(tmp, "tree");
  mkdirSync(tree);
  console.log("[weightspace site] weightspace check on a snapshot of HEAD");
  const code =
    status("git", ["archive", "--format=tar", "-o", path.join(tmp, "head.tar"), "HEAD"], repo) ||
    status("tar", ["-xf", path.join(tmp, "head.tar"), "-C", tree]) ||
    status("cargo", ["run", "--quiet", "--locked", "--", "check", tree], path.join(repo, "tool"));
  rmSync(tmp, { recursive: true, force: true });
  if (code !== 0) process.exit(code);
}

switch (cmd) {
  case "build":
  case "dev":
    if (env.WS_RUN_INDEX_CHECK === "1") {
      checkSnapshot();
    }
    run(process.execPath, [astro, cmd, ...rest]);
    break;
  case "preview":
    run(process.execPath, [astro, "preview", ...rest]);
    break;
  case "test":
    run(process.execPath, ["--test", "tests/*.test.ts"]);
    break;
  default:
    console.error("usage: node scripts/run.mjs build|dev|preview|test");
    process.exit(2);
}
