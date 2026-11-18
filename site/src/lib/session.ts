// bench/results/**/session.json has no definition in schema/records.schema.json
// (it is a bench file, not an index record). The site shape-checks the
// fields it shows, so a missing or malformed field stops the build instead of
// rendering a blank.
import { fail } from "./files.ts";

export interface SessionHalf {
  served_model: string;
  model_files: Record<string, string>;
  server: { max_model_len: number; version: string; managed: boolean; args: string[] };
  seconds: number;
}

export interface SessionRecord {
  protocol_sha256: string;
  settings_sha256: string;
  profile: string;
  seed: number;
  items: number[];
  base: SessionHalf;
  patched: SessionHalf;
  harness: { packages: Record<string, string>; task_config_sha256: string };
  concurrency: number;
  timeout: number;
  performed_by: "index" | "author" | "third-party";
  hardware: string;
  started: string;
}

const HEX64 = /^[0-9a-f]{64}$/;

export function checkSession(p: string, v: any): SessionRecord {
  const bad = (field: string, want: string): never => fail(`${p}: ${field} must be ${want}`);
  const obj = (x: unknown, field: string) => (x && typeof x === "object" && !Array.isArray(x) ? x : bad(field, "an object"));
  const str = (x: unknown, field: string) => (typeof x === "string" && x.length > 0 ? x : bad(field, "a nonempty string"));
  const int = (x: unknown, field: string) => (Number.isInteger(x) && (x as number) >= 0 ? x : bad(field, "a non-negative integer"));
  const hex = (x: unknown, field: string) => (typeof x === "string" && HEX64.test(x) ? x : bad(field, "a SHA-256 hex digest"));

  obj(v, "the file");
  hex(v.protocol_sha256, "protocol_sha256");
  hex(v.settings_sha256, "settings_sha256");
  str(v.profile, "profile");
  int(v.seed, "seed");
  if (!Array.isArray(v.items) || !v.items.length) bad("items", "a nonempty array");
  v.items.forEach((x: unknown, i: number) => int(x, `items[${i}]`));
  for (const half of ["base", "patched"]) {
    const h = obj(v[half], half) as any;
    str(h.served_model, `${half}.served_model`);
    const files = obj(h.model_files, `${half}.model_files`) as Record<string, unknown>;
    for (const [k, d] of Object.entries(files)) hex(d, `${half}.model_files.${k}`);
    const s = obj(h.server, `${half}.server`) as any;
    int(s.max_model_len, `${half}.server.max_model_len`);
    str(s.version, `${half}.server.version`);
    if (typeof s.managed !== "boolean") bad(`${half}.server.managed`, "a boolean");
    if (!Array.isArray(s.args) || s.args.some((a: unknown) => typeof a !== "string")) bad(`${half}.server.args`, "an array of strings");
    if (typeof h.seconds !== "number" || h.seconds < 0) bad(`${half}.seconds`, "a non-negative number");
  }
  const harness = obj(v.harness, "harness") as any;
  const pkgs = obj(harness.packages, "harness.packages") as Record<string, unknown>;
  for (const [k, ver] of Object.entries(pkgs)) str(ver, `harness.packages.${k}`);
  hex(harness.task_config_sha256, "harness.task_config_sha256");
  int(v.concurrency, "concurrency");
  int(v.timeout, "timeout");
  if (!["index", "author", "third-party"].includes(v.performed_by)) bad("performed_by", "index, author or third-party");
  str(v.hardware, "hardware");
  if (typeof v.started !== "string" || Number.isNaN(Date.parse(v.started))) bad("started", "an ISO 8601 time");
  return v as SessionRecord;
}
