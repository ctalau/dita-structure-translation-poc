#!/usr/bin/env python3
"""Dollar watchdog for the one-pod French typography budget.

The allocations are the contract in configs/fr_style.yaml. The hourly rate is
the community RTX 3090 rate recorded in docs/experiments/03-en-fr-violations.md
($0.22/hr). This module can terminate a pod. Nothing in the CPU preparation
calls RunPod.
"""
from __future__ import annotations

import os
import time
import urllib.request

HOURLY_USD = 0.22
HOURLY_USD_SOURCE = "docs/experiments/03-en-fr-violations.md recorded community RTX 3090 rate"
TOTAL_USD = 5.00
ALLOCATION_USD = {
    "pilot": 0.50,
    "sft": 1.80,
    "grpo": 1.30,
    "eval": 0.70,
    "contingency": 0.40,
    "storage": 0.30,
}


def allocation_total() -> float:
    return round(sum(ALLOCATION_USD.values()), 2)


def credentials_from_env() -> tuple[str, str]:
    """Pod id and API key for a future run. CPU preparation leaves both empty."""
    return os.environ.get("RUNPOD_POD_ID", ""), os.environ.get("RUNPOD_API_KEY", "")


class BudgetWatchdog:
    """Stop a stage when its dollar cap is reached and terminate the pod."""

    def __init__(self, stage: str, pod_id: str, api_key: str, hourly_usd: float = HOURLY_USD,
                 clock=None, terminate=None, started_at: float | None = None):
        if stage not in ALLOCATION_USD:
            raise KeyError(stage)
        self.stage = stage
        self.cap_usd = ALLOCATION_USD[stage]
        self.pod_id = pod_id
        self.api_key = api_key
        self.hourly_usd = hourly_usd
        self.clock = time.monotonic if clock is None else clock
        self.terminate = terminate_runpod_pod if terminate is None else terminate
        self.started_at = self.clock() if started_at is None else started_at
        self.terminated = False

    def elapsed_s(self) -> float:
        return max(0.0, self.clock() - self.started_at)

    def spent_usd(self) -> float:
        return self.hourly_usd * (self.elapsed_s() / 3600.0)

    def over_cap(self) -> bool:
        return self.spent_usd() >= self.cap_usd

    def poll(self) -> None:
        if self.terminated or not self.over_cap():
            return
        self.terminate(self.pod_id, self.api_key)
        self.terminated = True
        raise SystemExit(
            f"budget watchdog terminated pod {self.pod_id} at stage {self.stage} "
            f"after ${self.spent_usd():.2f} (cap ${self.cap_usd:.2f})"
        )


def terminate_runpod_pod(pod_id: str, api_key: str, opener=None) -> None:
    """DELETE the pod. Callers must pass an opener in tests. Do not call this from CPU prep."""
    if not pod_id:
        raise ValueError("pod_id is required")
    if not api_key:
        raise ValueError("api_key is required")
    request = urllib.request.Request(
        f"https://rest.runpod.io/v1/pods/{pod_id}",
        method="DELETE",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    open_url = urllib.request.urlopen if opener is None else opener
    open_url(request)
