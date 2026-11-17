# Check each JSON value independently so a native JSONL log can be streamed.
def sensitive_key:
  ascii_downcase | gsub("-"; "_") |
  test("(^|_)(api_key|apikey|access_token|auth_token|refresh_token|authorization|password|passwd|secret|private_key|client_secret|token|secret_key|api_token|bearer_token|session_token|hf_token|github_token|credentials|credential|privatekey|clientsecret)$");

all(inputs | .. | objects | to_entries[] | select(.key | sensitive_key);
    .value == null or .value == "" or .value == false or .value == [] or .value == {})
