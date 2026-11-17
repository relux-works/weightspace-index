# Stream one item per native row. An unknown row invalidates the attempt.
def key: [.task, .index] | tojson;
def digest: type == "string" and test("^[a-f0-9]{64}\\z");
INDEX($units[0][]; key) as $expected |
inputs as $row |
$expected[[$row.native_task, $row.doc_id] | tojson] as $unit |
if $unit == null then error("unexpected native task or doc_id")
elif (all($row.prompt_hash, $row.doc_hash; digest) | not) then error("invalid native hash")
else $unit + {
  prompt_sha256: $row.prompt_hash, doc_sha256: $row.doc_hash, finish_reason: null,
  prompt_tokens: null, completion_tokens: null, total_tokens: null
} end
