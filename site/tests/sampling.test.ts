import assert from "node:assert/strict";
import { test } from "node:test";
import { runSampling, samplingDiff, settingParts, settingText } from "../src/lib/sampling.ts";

const bench = {
  gen_kwargs: {
    do_sample: true, temperature: 1.0, top_p: 0.95, top_k: 20, min_p: 0.0, presence_penalty: 1.5,
    repetition_penalty: 1.0, max_gen_toks: 32768, chat_template_kwargs: { enable_thinking: true }, until: [],
  },
};

test("a bench run's gen_kwargs map onto the shared settings", () => {
  assert.deepEqual(runSampling(bench), {
    temperature: 1, top_p: 0.95, top_k: 20, min_p: 0, presence_penalty: 1.5, repetition_penalty: 1, max_tokens: 32768, thinking: true,
  });
  assert.equal(runSampling({ gen_kwargs: { do_sample: false, temperature: 0.7 } })?.temperature, 0);
  assert.equal(runSampling({}), undefined);
  assert.equal(runSampling(undefined), undefined);
});

test("only settings both sides state and disagree on count as a difference", () => {
  const reported = { temperature: 0.6, max_tokens: 32768, thinking: true };
  const measured = runSampling(bench)!;
  // top-p, presence penalty and the rest are unknown for the authors: never shown.
  assert.deepEqual(samplingDiff([reported, measured]), ["temperature"]);
  assert.deepEqual(samplingDiff([reported, undefined]), []);
  assert.deepEqual(samplingDiff([measured, { ...measured }]), []);
  // A later rerun at the authors' temperature still differs from the index's own run.
  assert.deepEqual(samplingDiff([reported, measured, { ...measured, temperature: 0.6 }]), ["temperature"]);
});

test("values read as settings: one decimal for temperature, switches as on/off", () => {
  const parts = settingParts({ temperature: 1, max_tokens: 32768, thinking: false }, ["temperature", "max_tokens", "thinking"]);
  assert.deepEqual(parts.map(p => `${p.short} = ${p.value}`), ["t = 1.0", "max = 32,768", "thinking = off"]);
  assert.equal(settingText(parts.slice(0, 1)), "temperature 1.0");
  assert.deepEqual(settingParts(undefined, ["temperature"]), []);
});
