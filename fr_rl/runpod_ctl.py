"""Drive one RunPod pod from this machine.

  python3 -m fr_rl.runpod_ctl balance
  python3 -m fr_rl.runpod_ctl create [GPU_TYPE_ID]
  python3 -m fr_rl.runpod_ctl ping
  python3 -m fr_rl.runpod_ctl put LOCAL REMOTE
  python3 -m fr_rl.runpod_ctl exec NAME 'shell command'
  python3 -m fr_rl.runpod_ctl status NAME [N_BYTES]
  python3 -m fr_rl.runpod_ctl get REMOTE LOCAL
  python3 -m fr_rl.runpod_ctl terminate

Every create/terminate and every balance read is appended to
results/fr_rl/pod_ledger.jsonl, so spend comes from the account balance,
not from an estimate. Pod id and token live in refs_work/pod.json (not in git).
"""
from __future__ import annotations

import base64
import json
import os
import secrets
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "refs_work" / "pod.json"
LEDGER = ROOT / "results" / "fr_rl" / "pod_ledger.jsonl"
API = "https://api.runpod.io/graphql"
# A widely cached image boots in seconds; pip wheels bring their own CUDA
# runtime, so only the host driver matters (allowedCudaVersions below).
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


def gql(query: str) -> dict:
    req = urllib.request.Request(API, data=json.dumps({"query": query}).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + os.environ["RUNPOD_KEY"],
                                          "User-Agent": "curl/8.5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    if d.get("errors"):
        raise RuntimeError(d["errors"])
    return d["data"]


def ledger(event: str, **kw):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    row = {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": event, **kw}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row))


def balance() -> float:
    b = gql("query { myself { clientBalance currentSpendPerHr } }")["myself"]
    ledger("balance", client_balance=round(b["clientBalance"], 4), spend_per_hr=b["currentSpendPerHr"])
    return b["clientBalance"]


def state() -> dict:
    return json.loads(STATE.read_text())


def create(gpu: str = "NVIDIA GeForce RTX 3090", clouds: str = "COMMUNITY", max_hr: float = 0.30):
    token = secrets.token_hex(16)
    server = base64.b64encode((ROOT / "fr_rl" / "pod_server.py").read_bytes()).decode()
    cmd = f"bash -c 'echo {server} | base64 -d > /srv.py && python3 /srv.py'"
    for cloud in clouds.split(","):
        try:
            d = gql(f"""mutation {{ podFindAndDeployOnDemand(input: {{
                cloudType: {cloud}, gpuCount: 1, gpuTypeId: "{gpu}", name: "fr-rl",
                imageName: "{IMAGE}", containerDiskInGb: 60, volumeInGb: 0,
                minVcpuCount: 2, minMemoryInGb: 20,
                ports: "8000/http", dockerArgs: {json.dumps(cmd)},
                env: [{{key: "CTL_TOKEN", value: "{token}"}}] }})
                {{ id costPerHr machine {{ gpuDisplayName }} }} }}""")["podFindAndDeployOnDemand"]
            break
        except RuntimeError as e:
            print(cloud, "failed:", e)
    else:
        raise SystemExit("no pod")
    STATE.parent.mkdir(exist_ok=True)
    if d["costPerHr"] > max_hr:  # never keep a pod above the agreed price
        pid = d["id"]
        gql(f'mutation {{ podTerminate(input: {{podId: "{pid}"}}) }}')
        ledger("create_rejected_price", pod=d["id"], cost_per_hr=d["costPerHr"], cloud=cloud)
        raise SystemExit("too expensive, terminated")
    STATE.write_text(json.dumps({"id": d["id"], "token": token}))
    ledger("create", pod=d["id"], cost_per_hr=d["costPerHr"], gpu=d["machine"]["gpuDisplayName"],
           cloud=cloud, image=IMAGE)


def _url(path: str) -> str:
    return f"https://{state()['id']}-8000.proxy.runpod.net{path}"


def call(method: str, path: str, data: bytes | None = None, timeout=300) -> bytes:
    req = urllib.request.Request(_url(path), data=data, method=method,
                                 headers={"X-Token": state()["token"], "User-Agent": "curl/8.5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def terminate():
    pid = state()["id"]
    gql(f'mutation {{ podTerminate(input: {{podId: "{pid}"}}) }}')
    ledger("terminate", pod=pid)


def main(a):
    cmd = a[1]
    if cmd == "balance":
        balance()
    elif cmd == "create":
        create(*a[2:3]) if len(a) < 4 else create(a[2], a[3], float(a[4]))
    elif cmd == "ping":
        print(call("GET", "/ping", timeout=20).decode())
    elif cmd == "put":
        print(call("POST", "/put?path=" + a[3], Path(a[2]).read_bytes()).decode())
    elif cmd == "exec":
        print(call("POST", "/exec?name=" + a[2], a[3].encode()).decode())
    elif cmd == "status":
        d = json.loads(call("GET", f"/status?name={a[2]}&n={a[3] if len(a) > 3 else 3000}"))
        print(d["tail"])
        print("RUNNING" if d["running"] else f"EXITED rc={d['returncode']}")
    elif cmd == "get":
        Path(a[3]).parent.mkdir(parents=True, exist_ok=True)
        Path(a[3]).write_bytes(call("GET", "/file?path=" + a[2], timeout=600))
        print("ok", a[3])
    elif cmd == "speedgate":
        # Any failure (no boot within the wait, slow download) terminates the pod.
        min_mbps = float(a[2]) if len(a) > 2 else 20.0
        wait_s = float(a[3]) if len(a) > 3 else 420
        try:
            t0 = time.time()
            while True:
                try:
                    call("GET", "/ping", timeout=15)
                    break
                except Exception:
                    if time.time() - t0 > wait_s:
                        raise RuntimeError(f"no boot within {wait_s}s")
                    time.sleep(10)
            boot_s = round(time.time() - t0)
            call("POST", "/put?path=/workspace/pod_gate.sh", (ROOT / "fr_rl" / "pod_gate.sh").read_bytes())
            call("POST", "/exec?name=speedgate", b"bash /workspace/pod_gate.sh")
            time.sleep(24)
            tail = json.loads(call("GET", "/status?name=speedgate"))["tail"].strip().splitlines()
            print(tail[-1])
            tail = tail[0]
            mbps = float(tail or 0) / 1e6
            ledger("speedgate", pod=state()["id"], boot_wait_s=boot_s, mb_per_s=round(mbps, 1), min_mb_per_s=min_mbps)
            if mbps < min_mbps:
                raise RuntimeError("slow download")
        except BaseException as e:
            ledger("speedgate_fail", pod=state()["id"], reason=str(e)[:200])
            terminate()
            raise SystemExit(1)
    elif cmd == "terminate":
        terminate()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
