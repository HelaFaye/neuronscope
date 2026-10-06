#!/usr/bin/env python3
"""Stand-in for llama-server in tests: same CLI shape, same HTTP surface."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

p = argparse.ArgumentParser()
p.add_argument("-m")
p.add_argument("--port", type=int)
p.add_argument("--host", default="127.0.0.1")
p.add_argument("--alias", default="")
p.add_argument("--mmproj", default="")
a, _ = p.parse_known_args()


class H(BaseHTTPRequestHandler):
    def log_message(self, *x):
        pass

    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            return self._send({"status": "ok"})
        self._send({"data": [{"id": a.alias}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("/embeddings"):
            # deterministic bag-of-words vectors: similar wording -> similar vectors
            import hashlib
            import re
            out = []
            for i, t in enumerate(body["input"] if isinstance(body["input"], list) else [body["input"]]):
                v = [0.0] * 64
                for w in re.findall(r"[a-z]+", t.lower()):
                    v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
                out.append({"index": i, "embedding": v})
            return self._send({"data": out, "model": a.alias})
        has_img = any(isinstance(m.get("content"), list) for m in body.get("messages", []))
        ctx = any(m.get("role") == "system" and "[1] (" in str(m.get("content")) for m in body.get("messages", []))
        text = f"model={a.alias} mmproj={'yes' if a.mmproj else 'no'} image={'yes' if has_img else 'no'}"
        if ctx:
            text += " ctx=yes"
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for piece in [text[:10], text[10:]]:
                self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'content': piece}}]})}\n\n".encode())
            final = {"choices": [{"delta": {}, "finish_reason": "stop"}],
                     "timings": {"predicted_n": 7, "predicted_per_second": 42.0}}
            self.wfile.write(f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n".encode())
            return
        self._send({"choices": [{"message": {"role": "assistant", "content": text}}],
                    "model": a.alias})


ThreadingHTTPServer((a.host, a.port), H).serve_forever()
