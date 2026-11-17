# Read exactly one completed reference profile.
def nonempty: type == "string" and length > 0 and (contains("REQUIRED") | not);
def digest: type == "string" and test("^[a-f0-9]{64}\\z");

length == 1 and (.[0] |
  .reference_config == "qwen35-0.8b-vllm-metal-bf16-2x12-v1" and
  .quantization == "bf16" and
  all(.[]; . != null and . != "" and (tostring | contains("REQUIRED") | not)) and
  all(.base_weights_digest, .tokenizer_digest, .template_digest; digest) and
  all(.patch_digest, .conversion_tool_versions, .patch_quantization_order,
      .engine_version_observed, .plugin_version_observed, .machine_layout_evidence; nonempty))
