"""Minimal token-authenticated control server that runs on the RunPod pod.

Started by the pod's docker command. Reached through RunPod's HTTPS proxy
(https://<pod>-8000.proxy.runpod.net). Standard library only.

  POST /exec?name=N      body: shell command; runs detached, log in /workspace/logs/N.log
  GET  /status?name=N    {"running", "returncode", "tail"}
  GET  /file?path=P      raw bytes
  POST /put?path=P       body: raw bytes written to P
"""
import json
import os
import subprocess
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOKEN = os.environ["CTL_TOKEN"]
LOGS = "/workspace/logs"
os.makedirs(LOGS, exist_ok=True)
PROCS = {}


class H(BaseHTTPRequestHandler):
    def _ok(self, body, ctype="application/json", code=200):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _auth(self):
        if self.headers.get("X-Token") != TOKEN:
            self._ok({"error": "auth"}, code=403)
            return None
        u = urllib.parse.urlparse(self.path)
        return u.path, dict(urllib.parse.parse_qsl(u.query))

    def _body(self):
        return self.rfile.read(int(self.headers.get("Content-Length", 0)))

    def do_GET(self):
        a = self._auth()
        if not a:
            return
        path, q = a
        if path == "/status":
            p = PROCS.get(q.get("name"))
            log = os.path.join(LOGS, q.get("name", "x") + ".log")
            tail = ""
            if os.path.exists(log):
                with open(log, "rb") as f:
                    f.seek(max(0, os.path.getsize(log) - int(q.get("n", 4000))))
                    tail = f.read().decode("utf-8", "replace")
            self._ok({"running": bool(p and p.poll() is None),
                      "returncode": p.poll() if p else None, "tail": tail})
        elif path == "/file":
            try:
                with open(q["path"], "rb") as f:
                    self._ok(f.read(), "application/octet-stream")
            except OSError as e:
                self._ok({"error": str(e)}, code=404)
        elif path == "/ping":
            self._ok({"ok": True})
        else:
            self._ok({"error": "no route"}, code=404)

    def do_POST(self):
        a = self._auth()
        if not a:
            return
        path, q = a
        if path == "/exec":
            cmd = self._body().decode()
            name = q.get("name", "job")
            log = open(os.path.join(LOGS, name + ".log"), "ab")
            PROCS[name] = subprocess.Popen(["bash", "-lc", cmd], stdout=log, stderr=subprocess.STDOUT,
                                           cwd="/workspace", start_new_session=True)
            self._ok({"pid": PROCS[name].pid})
        elif path == "/put":
            os.makedirs(os.path.dirname(q["path"]) or ".", exist_ok=True)
            with open(q["path"], "wb") as f:
                f.write(self._body())
            self._ok({"ok": True})
        else:
            self._ok({"error": "no route"}, code=404)

    def log_message(self, *a):
        pass


ThreadingHTTPServer(("0.0.0.0", 8000), H).serve_forever()
