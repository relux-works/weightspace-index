// Each recipe starts in the directory containing the selected download.
import { createHash } from "node:crypto";
import { downloadUrl } from "./catalog.ts";
import { href, seg } from "./config.ts";
import type { FileEntry, Version } from "./types.ts";
import { adapterBundle } from "./runtime-package.ts";

interface Download { href: string; name: string; sizeBytes: number; label: string; sha256: string }
export interface RuntimeTab { name: string; note: string; command: string; docs: string; setup?: string; setupLabel?: string; prepare?: string }
export interface RuntimeGuide { download: Download | null; tabs: RuntimeTab[] }

const LLAMA_DOCS = "https://github.com/ggml-org/llama.cpp/blob/ece963f41b0b02d7a0d61436ae365762c073a4c8/tools/server/README.md";
const MAX_RANKS = [1, 8, 16, 32, 64, 128, 256, 320, 512];
const CLONE = "git clone --depth 1 --branch b10450 https://github.com/ggml-org/llama.cpp.git llama.cpp";
const DEPS = "python3 -m pip install ./llama.cpp/gguf-py torch transformers sentencepiece protobuf";
const quote = (s: string) => "'" + s.replaceAll("'", "'\\''") + "'";

function link(file: FileEntry, label: string): Download {
  return { href: downloadUrl(file), name: file.name, sizeBytes: file.size_bytes, label, sha256: file.sha256 };
}

function llamaTab(v: Version, prepare: string, adapter: string, packaged: boolean): RuntimeTab {
  const { repo_id, revision } = v.profile.source;
  const gguf = v.profile.files.find(f => f.path.toLowerCase().endsWith(".gguf"));
  const base = gguf ? `./base/${gguf.path}` : "./base.gguf";
  const download = `hf download ${repo_id} --revision ${revision} --local-dir ./base`;
  const setup = gguf ? download : [download, CLONE, DEPS,
    "python3 llama.cpp/convert_hf_to_gguf.py ./base --outtype bf16 --outfile ./base.gguf"].join("\n");
  return {
    name: "llama.cpp", docs: LLAMA_DOCS, setup,
    note: packaged
      ? "Use the included F32 GGUF adapter with a base GGUF from the pinned revision. The bundle includes the llama.cpp load-check details."
      : "Use a base GGUF from the pinned revision with the GGUF adapter.",
    prepare, command: `llama-server -m ${base} --lora ${adapter}`,
  };
}

function peftTabs(v: Version, prepare: string, dir: string, rank?: number): RuntimeTab[] {
  const { repo_id, revision } = v.profile.source;
  const moduleName = `${v.patch.slug}-${v.label}`;
  // An externally hosted config can declare a per-module rank pattern.
  const chooseRank = `import json; c=json.load(open("${dir}/adapter_config.json")); r=max([c["r"], *(c.get("rank_pattern") or {}).values()]); print(next(n for n in ${JSON.stringify(MAX_RANKS)} if n >= r))`;
  const cap = rank === undefined ? `"$(python3 -c '${chooseRank}')"` : MAX_RANKS.find(r => r >= rank);
  const tabs: RuntimeTab[] = [];
  if (cap) {
    tabs.push({
      name: "vLLM", docs: "https://docs.vllm.ai/en/stable/features/lora/",
      note: `Select ${moduleName} in the request’s model field. vLLM serving has not been tested for this adapter.`,
      prepare, command: [`vllm serve ${repo_id} --revision ${revision} \\`, `  --enable-lora --max-lora-rank ${cap} \\`, `  --lora-modules ${moduleName}=${dir}`].join("\n"),
    });
  }
  tabs.push({
    name: "Transformers", docs: "https://huggingface.co/docs/peft/quicktour",
    note: "Load the pinned checkpoint and attach the adapter with Transformers, PyTorch and PEFT installed.",
    prepare, command: ["python3 - <<'PY'", "import transformers", "from peft import PeftModel", "",
      `repo, rev = "${repo_id}", "${revision}"`,
      "config = transformers.AutoConfig.from_pretrained(repo, revision=rev)",
      "model_class = getattr(transformers, config.architectures[0])",
      "base = model_class.from_pretrained(repo, revision=rev, dtype=\"auto\")",
      `model = PeftModel.from_pretrained(base, "${dir}")`,
      "tok = transformers.AutoTokenizer.from_pretrained(repo, revision=rev)",
      "tokens = tok(\"What is 2 + 2?\", return_tensors=\"pt\")",
      "print(tok.decode(model.generate(**tokens, max_new_tokens=12)[0]))", "PY"].join("\n"),
  });
  return tabs;
}

export function runtimeGuide(v: Version): RuntimeGuide {
  const bundle = adapterBundle(v);
  const peft = v.files.find(f => f.name === "adapter_model.safetensors");
  const config = v.files.find(f => f.name === "adapter_config.json");
  const gguf = v.files.find(f => f.name.toLowerCase().endsWith(".gguf"));
  const dir = `./${v.patch.slug}-${v.label}-lora`;
  let download: Download | null = null;
  let prepare = "";
  if (bundle) {
    download = {
      name: bundle.filename, sizeBytes: bundle.bytes.length, label: bundle.gguf ? "PEFT + GGUF" : "PEFT bundle",
      href: href(`adapters/${seg(v.patch.base.address)}/${seg(bundle.filename)}`),
      sha256: createHash("sha256").update(bundle.bytes).digest("hex"),
    };
    prepare = `mkdir -p ${dir}\ntar -xzf ${bundle.filename} -C ${dir}`;
  } else if (peft && config) {
    download = link(peft, "PEFT weights");
    prepare = [`mkdir -p ${dir}`, `cp ./adapter_model.safetensors ${dir}/`,
      `curl -fL ${quote(config.links[0].href)} -o ${dir}/adapter_config.json`].join("\n");
  }
  const hfWeights = v.profile.files.some(f => f.path.endsWith(".safetensors") || f.path.endsWith(".bin"));
  const tabs = prepare && hfWeights ? peftTabs(v, prepare, dir, bundle?.rank) : [];
  if (gguf) {
    download ??= link(gguf, "GGUF adapter");
    const fetch = download.name === gguf.name ? "" : `curl -fL ${quote(gguf.links[0].href)} -o ${gguf.name}`;
    tabs.splice(1, 0, llamaTab(v, fetch, "./" + gguf.name, false));
  } else if (bundle?.gguf) {
    tabs.splice(1, 0, llamaTab(v, prepare, dir + "/adapter.gguf", true));
  } else if (prepare && hfWeights) {
    // A PEFT-only release needs a model-aware conversion before llama.cpp.
    const convert = `python3 llama.cpp/convert_lora_to_gguf.py --base ./base --outtype f32 --outfile ./adapter.gguf ${dir}`;
    const tab = llamaTab(v, prepare + "\n" + convert, "./adapter.gguf", false);
    tab.note = "Prepare the base GGUF and llama.cpp converter below. Then convert the PEFT adapter and start the server.";
    tabs.splice(1, 0, tab);
  }
  const raw = peft ?? v.files.find(f => f.name === "adapter.safetensors" && f.role === "derived");
  if (!download && raw) download = link(raw, "LoRA weights");
  return { download, tabs };
}
