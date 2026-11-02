"""Synthetic probe: ask the demo 8 fixed questions and score each answer with goldpass.

Privacy: answer, question and citation text is read only in memory to score the answer, then
dropped. A `ProbeRun` holds scores, statuses, latencies and trace ids only.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx
from goldpass.golden import GoldenItem, load_golden
from goldpass.mapping import FieldMap, MappingError, to_record
from goldpass.metrics import score_item

from rail_rag_monitor.stats import percentile

PROBE_GOLDEN = Path("probe/golden.jsonl")

PACE_S = 13.0  # between /ask calls (the demo rate-limits)
TIMEOUT_S = 120.0
RETRIES_503 = 3
BACKOFF_503_S = 15.0


@dataclass
class QuestionResult:
    pass_: bool | None
    refused: bool | None
    grounding: bool | None
    status: str
    latency_s: float | None
    cold: bool
    trace_id: str | None

    def to_dict(self) -> dict:
        return {
            "pass": self.pass_,
            "refused": self.refused,
            "grounding": self.grounding,
            "status": self.status,
            "latency_s": self.latency_s,
            "cold": self.cold,
            "trace_id": self.trace_id,
        }


@dataclass
class ProbeRun:
    run_id: str
    started_at: str
    commit: str
    kind: str
    demo_version: str | None
    results: dict[str, QuestionResult] = field(default_factory=dict)
    n_pass: int = 0
    n_scored: int = 0
    n_errors: int = 0
    warm_p50_s: float | None = None
    warm_p95_s: float | None = None
    cold_latency_s: float | None = None

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "commit": self.commit,
            "kind": self.kind,
            "demo_version": self.demo_version,
            "results": {qid: r.to_dict() for qid, r in self.results.items()},
            "n_pass": self.n_pass,
            "n_scored": self.n_scored,
            "n_errors": self.n_errors,
            "warm_p50_s": self.warm_p50_s,
            "warm_p95_s": self.warm_p95_s,
            "cold_latency_s": self.cold_latency_s,
        }


def question_pass(expected: str, refused: bool | None, grounding: bool | None) -> bool | None:
    """Per-question verdict; None when the answer was not scored (e.g. an HTTP error)."""
    if refused is None:
        return None
    if expected == "refuse":
        return refused is True
    return refused is False and grounding is True


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _demo_version(client: httpx.Client, base: str) -> str | None:
    try:
        resp = client.get(f"{base}/health", timeout=TIMEOUT_S)
        body = resp.json() if resp.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        return None
    version = body.get("version") if isinstance(body, dict) else None
    return version if isinstance(version, str) else None


def _ask(
    client: httpx.Client,
    base: str,
    item: GoldenItem,
    *,
    cold: bool,
    sleep: Callable[[float], None],
    timer: Callable[[], float],
) -> QuestionResult:
    """One question with the retry policy; never raises on a demo failure."""

    def failed(status: str) -> QuestionResult:
        return QuestionResult(None, None, None, status, None, cold, None)

    retries_503 = 0
    retried_timeout = False
    while True:
        started = timer()
        try:
            resp = client.post(f"{base}/ask", json={"question": item.question}, timeout=TIMEOUT_S)
        except httpx.TimeoutException:
            # The demo scales to zero: only the first (cold) call gets a second try.
            if cold and not retried_timeout:
                retried_timeout = True
                continue
            return failed("timeout")
        except httpx.HTTPError:
            return failed("connection")
        latency = timer() - started

        if resp.status_code == 429:
            return failed("rate_limited")
        if resp.status_code == 503 and retries_503 < RETRIES_503:
            retries_503 += 1
            sleep(BACKOFF_503_S)
            continue
        if resp.status_code != 200:
            return failed(f"http_{resp.status_code}")
        try:
            body = resp.json()
            if not isinstance(body, dict):
                raise ValueError("response is not a JSON object")
            score = score_item(item, to_record(body, item.id, FieldMap(), latency))
        except (ValueError, TypeError, MappingError):
            return failed("bad_response")

        grounding = score.metrics.get("grounding")
        grounding = grounding if isinstance(grounding, bool) else None
        trace_id = body.get("trace_id")
        return QuestionResult(
            pass_=question_pass(item.expected, score.refused, grounding),
            refused=score.refused,
            grounding=grounding,
            status="ok",
            latency_s=latency,
            cold=cold,
            trace_id=trace_id if isinstance(trace_id, str) else None,
        )


def run_probe(
    base_url: str,
    golden_path: Path,
    *,
    run_id: str,
    kind: str,
    commit: str,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = _utcnow,
    timer: Callable[[], float] = time.perf_counter,
) -> ProbeRun:
    items = load_golden(golden_path)
    base = base_url.rstrip("/")
    own_client = client is None
    http = client if client is not None else httpx.Client(timeout=TIMEOUT_S)
    try:
        started_at = now().isoformat()
        run = ProbeRun(run_id, started_at, commit, kind, _demo_version(http, base))
        for i, item in enumerate(items):
            if i > 0:
                sleep(PACE_S)
            run.results[item.id] = _ask(http, base, item, cold=i == 0, sleep=sleep, timer=timer)
    finally:
        if own_client:
            http.close()

    results = list(run.results.values())
    run.n_pass = sum(r.pass_ is True for r in results)
    run.n_scored = sum(r.pass_ is not None for r in results)
    run.n_errors = sum(r.status != "ok" for r in results)
    warm = [
        r.latency_s for r in results if not r.cold and r.status == "ok" and r.latency_s is not None
    ]
    run.warm_p50_s = percentile(warm, 0.5)
    run.warm_p95_s = percentile(warm, 0.95)
    cold = next((r for r in results if r.cold and r.status == "ok"), None)
    run.cold_latency_s = cold.latency_s if cold else None
    return run
