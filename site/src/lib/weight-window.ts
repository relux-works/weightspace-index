import type { DeltaTensor } from "./insight.ts";

/** A contiguous column window through the sampled rows, with as many edited weights as possible. */
export function weightWindow(tensor: Pick<DeltaTensor, "cols" | "columns" | "cells">, maxColumns: number) {
  const width = Math.min(maxColumns, tensor.cols);
  const columnIndex = new Map(tensor.columns.map((col, index) => [col, index]));
  const counts = new Map(tensor.columns.map((col, index) => [
    col, tensor.cells.filter(row => row[index] !== null).length,
  ]));
  const count = (col: number) => counts.get(col) ?? 0;

  let total = 0;
  for (let col = 0; col < width; col++) total += count(col);
  let best = total;
  let first = 0;

  for (let start = 1; start <= tensor.cols - width; start++) {
    total += count(start + width - 1) - count(start - 1);
    if (total > best) {
      best = total;
      first = start;
    }
  }

  if (!best || width < 1) return null;
  return tensor.cells.map(row => Array.from({ length: width }, (_, col) => row[columnIndex.get(first + col) ?? -1] ?? null));
}
