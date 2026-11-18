"""Test-only OpenAI-shaped server; never loads a model or dataset."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--variant", choices=["base", "edited"], default="base")
parser.add_argument("--mode", default="complete")
args = parser.parse_args()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, status, body):
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        self.reply(200, {"data": [{"id": "bench-model"}]})

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        # This asserts the actual stock backend wire contract, not an adapter mock.
        assert data["model"] == "bench-model" and data["max_tokens"] == 32768
        assert data["temperature"] == 0.6 and data["top_p"] == 0.95
        assert data["top_k"] == 20 and data["min_p"] == 0.0
        assert data["stop"] == ["<|im_end|>"]
        assert "Authorization" not in self.headers
        prompt = data["messages"][-1]["content"]
        if args.mode == "error":
            self.reply(503, {"error": "fixture failure"})
            return
        answer = "B" if "fixture-choice" in prompt else "2"
        if args.variant == "edited":
            answer = "A" if answer == "B" else "3"
        content = f"The answer is ({answer})." if "fixture-choice" in prompt else f"Final Answer: The final answer is ${answer}$. I hope it is correct."
        if args.mode == "incomplete" and "fixture-second" in prompt:
            content = ""
        self.reply(200, {"choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]})


HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
