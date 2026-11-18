/** One scale per unit, with zero included so lengths are comparable. */
export function scoreAxis(values: number[], unit: string) {
  const low = Math.min(0, ...values);
  const high = Math.max(unit === "%" ? 100 : 0, ...values);
  const position = (v: number) => 100 * (v - low) / (high - low || 1);
  const zero = position(0);
  const bar = (v: number) => `left:${Math.min(zero, position(v))}%;width:${Math.abs(position(v) - zero)}%`;
  return { position, zero, bar };
}
