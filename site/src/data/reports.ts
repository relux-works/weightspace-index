// Editorial abstracts are scoped to one source artifact. New releases need their own citation.
import type { Version } from "../lib/types.ts";

interface AuthorReport {
  name: string;
  goal: string;
  /** What the reporters measured, in one or two plain sentences. */
  scope: string;
  /** Everyone the study credits, in its byline order; the patch authors when absent. */
  by?: string[];
  affiliation?: string;
  url: string;
  studyUrl?: string;
  studyTitle?: string;
  measures: { label: string; base: number; edited: number }[];
}

const reports: Record<string, AuthorReport> = {
  "62c555179d5133102991bb00278d8babdba59cca673e65386e3ea37d0eae4fd9": {
    name: "Thinking loops",
    goal: "Targets repetitive reasoning on math problems by editing two MLP layers.",
    scope: "They measured their own edited checkpoint as a text-only export, with thinking on and a 32,768-token limit.",
    by: ["Rikiya Takehi", "Jayden Teoh", "Alex Zhang", "Omar Khattab"],
    affiliation: "MIT CSAIL",
    url: "https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming/blob/d1671a178705bcab7644806d23c535517f35723b/README.md",
    studyUrl: "https://rikiyat.github.io/blog/weight-space-programming/",
    studyTitle: "Sparks of Weight-Space Programming",
    measures: [
      { label: "MATH-500 accuracy", base: 7.9, edited: 48.1 },
      { label: "GSM8K accuracy", base: 13.8, edited: 67.1 },
      { label: "IFEval score", base: 21, edited: 62.3 },
      { label: "Hit token limit", base: 93, edited: 33.1 },
    ],
  },
};

export function authorReport(version?: Version) {
  if (version?.profile.source.repo_id !== "Qwen/Qwen3.5-0.8B" ||
      version.profile.source.revision !== "2fc06364715b967f1860aea9cf38778875588b17") return;
  const source = version.files.find(f => f.role === "source");
  return source ? reports[source.sha256] : undefined;
}
