// Deterministic textures for illustrative weight matrices. Real per-weight
// changes come from editDeltas() in insight.ts; these only draw the backdrop
// and the explainer's made-up examples.
const hash = (i: number, j: number, s: number) => ((Math.sin(i * 12.9898 + j * 78.233 + s * 37.719) * 43758.5453) % 1 + 1) % 1;

/** Magnitude of an illustrative weight, 0–1, skewed toward small values like real weights. */
export const weight = (i: number, j: number, s = 0) => hash(i, j, s) ** 1.6;

/** Strength of an illustrative change, 0 for a value the edit leaves alone. */
export const change = (i: number, j: number, s = 0) => {
  const n = hash(i + 31, j + 17, s + 5);
  return n < 0.14 ? 0 : 0.2 + 0.8 * n ** 0.8;
};
