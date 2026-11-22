// Inference settings behind a result, in one vocabulary for both sources: a
// claim's `sampling` (what the authors' source states) and a bench run's
// `settings.gen_kwargs`. The set of parameters is small and fixed, so results
// side by side can show exactly the settings that differ between them.

export interface Sampling {
  temperature?: number;
  top_p?: number;
  top_k?: number;
  min_p?: number;
  presence_penalty?: number;
  repetition_penalty?: number;
  max_tokens?: number;
  thinking?: boolean;
}

export type SamplingKey = keyof Sampling;

/** Display order, and how each setting reads on a chart row. */
const PARAMS: { key: SamplingKey; short: string; long: string }[] = [
  { key: "temperature", short: "t", long: "temperature" },
  { key: "top_p", short: "top-p", long: "top-p" },
  { key: "top_k", short: "top-k", long: "top-k" },
  { key: "min_p", short: "min-p", long: "min-p" },
  { key: "presence_penalty", short: "presence", long: "presence penalty" },
  { key: "repetition_penalty", short: "repetition", long: "repetition penalty" },
  { key: "max_tokens", short: "max", long: "max new tokens" },
  { key: "thinking", short: "thinking", long: "thinking" },
];

const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : undefined);

/** A claim's stated settings, as recorded. */
export function claimSampling(record: { sampling?: Sampling }): Sampling | undefined {
  return record.sampling && Object.keys(record.sampling).length ? record.sampling : undefined;
}

/** A run's settings, from bench's lm-eval `gen_kwargs`. Runs filed another way have none. */
export function runSampling(settings: Record<string, any> | undefined): Sampling | undefined {
  const g = settings?.gen_kwargs;
  if (!g || typeof g !== "object") return;
  const thinking = g.chat_template_kwargs?.enable_thinking;
  const s: Sampling = {
    temperature: g.do_sample === false ? 0 : num(g.temperature),
    top_p: num(g.top_p),
    top_k: num(g.top_k),
    min_p: num(g.min_p),
    presence_penalty: num(g.presence_penalty),
    repetition_penalty: num(g.repetition_penalty),
    max_tokens: num(g.max_gen_toks),
    thinking: typeof thinking === "boolean" ? thinking : undefined,
  };
  for (const k of Object.keys(s) as SamplingKey[]) if (s[k] === undefined) delete s[k];
  return Object.keys(s).length ? s : undefined;
}

/**
 * The settings that differ among results shown together: a setting counts when
 * at least two of them state it with different values. A setting only one side
 * states is unknown on the other, so it is never shown as a difference.
 */
export function samplingDiff(list: (Sampling | undefined)[]): SamplingKey[] {
  return PARAMS.map(p => p.key).filter(k => {
    const stated = list.flatMap(s => (s?.[k] === undefined ? [] : [s[k]]));
    return new Set(stated).size > 1;
  });
}

function value(key: SamplingKey, v: number | boolean) {
  if (typeof v === "boolean") return v ? "on" : "off";
  // Sampling floats keep one decimal so 1 reads as the setting 1.0.
  if (key === "temperature" || key.endsWith("_p") || key.endsWith("_penalty")) return Number.isInteger(v) ? v.toFixed(1) : String(v);
  return v.toLocaleString("en-US");
}

export interface SettingPart { key: SamplingKey; short: string; long: string; value: string }

/** One result's values for the differing settings, in display order. */
export function settingParts(s: Sampling | undefined, keys: SamplingKey[]): SettingPart[] {
  if (!s) return [];
  return PARAMS.filter(p => keys.includes(p.key) && s[p.key] !== undefined)
    .map(p => ({ key: p.key, short: p.short, long: p.long, value: value(p.key, s[p.key]!) }));
}

/** Plain text of those values, e.g. "temperature 0.6", for titles and screen readers. */
export const settingText = (parts: SettingPart[]) => parts.map(p => `${p.long} ${p.value}`).join(", ");
