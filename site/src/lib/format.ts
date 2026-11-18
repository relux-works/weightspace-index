// Display formatting only: no value is changed, rounded away or inferred here.
import type { Score } from "./index.ts";

const UNITS = ["B", "KB", "MB", "GB", "TB"];

/** "1.56 MB"; the exact byte count is always shown next to it. */
export function humanBytes(n: number): string {
  let v = n;
  let i = 0;
  while (v >= 1000 && i < UNITS.length - 1) {
    v /= 1000;
    i++;
  }
  return i === 0 ? `${n} B` : `${v.toFixed(v >= 100 ? 0 : v >= 10 ? 1 : 2)} ${UNITS[i]}`;
}

export function exactBytes(n: number): string {
  return `${n.toLocaleString("en-US")} bytes`;
}

/** A recorded score exactly as stored: value, metric, unit and denominator. */
export function scoreText(s: Score): string {
  if (s.state === "UNKNOWN") return "UNKNOWN";
  return `${s.value} ${s.metric} (${s.unit}, n = ${s.denominator})`;
}

export function shortHash(h: string, n = 12): string {
  return h.length > n ? `${h.slice(0, n)}…` : h;
}

export function plural(n: number, one: string, many = `${one}s`): string {
  return `${n} ${n === 1 ? one : many}`;
}

export function utc(iso: string): string {
  return iso.replace("T", " ").replace(/\.\d+Z$|Z$/, " UTC");
}

export function duration(seconds: number): string {
  const s = Math.round(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  return h ? `${h} h ${m} min` : m ? `${m} min ${r} s` : `${r} s`;
}

const TONES: Record<string, string> = {
  COMPLETE: "ok",
  "complete run": "ok",
  info: "info",
  SUPPORTED: "ok",
  active: "ok",
  MATCHED: "info",
  latest: "accent",
  INCOMPLETE: "warn",
  UNKNOWN: "warn",
  INCONCLUSIVE: "neutral",
  deprecated: "neutral",
  REFUSED: "bad",
  REJECT: "bad",
  UNSUPPORTED: "bad",
  withdrawn: "bad",
};

/** Colour family for a state word; anything unlisted is neutral. */
export function tone(state: string): string {
  const key = Object.keys(TONES).find((k) => state === k || state.startsWith(`${k} `));
  return key ? TONES[key] : "neutral";
}
