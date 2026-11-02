"""Alert rules (spec section 7): uptime, quality and latency, evaluated on the probe store.

`evaluate` takes probe run dicts (as written by `ProbeRun.to_dict()`) in chronological order.
The recent window is the last `RECENT_RUNS` runs; the baseline is the up to `BASELINE_MAX_RUNS`
non-dry-run runs before it. Runs of kind `dry-run` never count toward a baseline.
"""

from __future__ import annotations

import copy
import random
import statistics
from dataclasses import dataclass, field
from typing import Literal

from rail_rag_monitor.stats import fisher_one_sided, percentile

MIN_BASELINE_RUNS = 7
RECENT_RUNS = 3
BASELINE_MAX_RUNS = 14
ALPHA = 0.01
DELTA_FLOOR = 2 / 24
LATENCY_FACTOR = 2.0
CLEAN_RUNS_TO_CLOSE = 3

UPTIME_RUNS = 2
LATENCY_RUNS = 2
DRY_RUN = "dry-run"
PLANTED_QUESTIONS = ("q01", "q06", "q09", "q10")


@dataclass
class Alert:
    rule: Literal["uptime", "quality", "latency"]
    summary: str
    details: dict = field(default_factory=dict)


def _results(run: dict) -> dict:
    return run.get("results") or {}


def _uptime(runs: list[dict]) -> Alert | None:
    last = runs[-UPTIME_RUNS:]
    if len(last) < UPTIME_RUNS:
        return None
    errors = [
        {
            "run_id": run["run_id"],
            "question_id": qid,
            "status": res.get("status"),
            "trace_id": res.get("trace_id"),
        }
        for run in last
        for qid, res in _results(run).items()
        if res.get("status") != "ok"
    ]
    if not all(any(r.get("status") != "ok" for r in _results(run).values()) for run in last):
        return None
    run_ids = [run["run_id"] for run in last]
    return Alert(
        "uptime",
        f"Probe questions ended in an error in {UPTIME_RUNS} consecutive runs "
        f"({', '.join(run_ids)}).",
        {"run_ids": run_ids, "errors": errors},
    )


def _fails_and_scored(run: dict) -> tuple[list[str], int]:
    passes = {qid: res.get("pass") for qid, res in _results(run).items()}
    scored = sum(p is not None for p in passes.values())
    return [qid for qid, p in passes.items() if p is False], scored


def _quality(recent: list[dict], baseline: list[dict], delta: float) -> Alert | None:
    if len(baseline) < MIN_BASELINE_RUNS:
        return None
    fail_recent = n_recent = fail_base = n_base = 0
    failed: list[dict] = []
    for run in recent:
        fails, scored = _fails_and_scored(run)
        fail_recent += len(fails)
        n_recent += scored
        failed += [
            {
                "run_id": run["run_id"],
                "question_id": qid,
                "trace_id": _results(run)[qid].get("trace_id"),
            }
            for qid in fails
        ]
    for run in baseline:
        fails, scored = _fails_and_scored(run)
        fail_base += len(fails)
        n_base += scored
    if n_recent == 0 or n_base == 0:
        return None

    f_recent = fail_recent / n_recent
    f_base = fail_base / n_base
    p_value = fisher_one_sided(fail_recent, n_recent, fail_base, n_base)
    if not (p_value < ALPHA and f_recent - f_base >= delta):
        return None
    return Alert(
        "quality",
        f"Fail rate rose to {f_recent:.1%} ({fail_recent}/{n_recent}) in the last "
        f"{len(recent)} runs from {f_base:.1%} ({fail_base}/{n_base}) in the "
        f"{len(baseline)} before (one-sided Fisher p={p_value:.2g}).",
        {
            "fail_recent": fail_recent,
            "n_recent": n_recent,
            "fail_base": fail_base,
            "n_base": n_base,
            "fail_rate_recent": f_recent,
            "fail_rate_base": f_base,
            "p_value": p_value,
            "alpha": ALPHA,
            "delta": delta,
            "recent_run_ids": [r["run_id"] for r in recent],
            "baseline_run_ids": [r["run_id"] for r in baseline],
            "failed": failed,
        },
    )


def _latency(recent: list[dict], baseline: list[dict]) -> Alert | None:
    last = recent[-LATENCY_RUNS:]
    values = [r["warm_p95_s"] for r in baseline if r.get("warm_p95_s") is not None]
    if len(last) < LATENCY_RUNS or len(values) < MIN_BASELINE_RUNS:
        return None
    median = statistics.median(values)
    threshold = LATENCY_FACTOR * median
    recent_p95 = [r.get("warm_p95_s") for r in last]
    if any(v is None or not v > threshold for v in recent_p95):
        return None
    return Alert(
        "latency",
        f"Warm p95 latency was {recent_p95[0]:.2f}s then {recent_p95[1]:.2f}s, above "
        f"{LATENCY_FACTOR:g}x the baseline median of {median:.2f}s.",
        {
            "recent_warm_p95_s": recent_p95,
            "baseline_median_s": median,
            "threshold_s": threshold,
            "factor": LATENCY_FACTOR,
            "n_base": len(values),
            "recent_run_ids": [r["run_id"] for r in last],
            "baseline_run_ids": [r["run_id"] for r in baseline if r.get("warm_p95_s") is not None],
        },
    )


def evaluate(runs: list[dict], delta: float) -> list[Alert]:
    """Run all rules on `runs` (chronological). `delta` is the quality noise margin."""
    alerts: list[Alert] = []
    uptime = _uptime(runs)
    if uptime:
        alerts.append(uptime)

    if len(runs) <= RECENT_RUNS:
        return alerts
    recent = runs[-RECENT_RUNS:]
    earlier = [r for r in runs[:-RECENT_RUNS] if r.get("kind") != DRY_RUN]
    baseline = earlier[-BASELINE_MAX_RUNS:]
    for alert in (_quality(recent, baseline, delta), _latency(recent, baseline)):
        if alert:
            alerts.append(alert)
    return alerts


def planted_runs(last_run: dict, n: int = 3) -> list[dict]:
    """In-memory regression: copies of `last_run` where q01, q06, q09 and q10 refuse and fail."""
    planted = []
    for i in range(1, n + 1):
        run = copy.deepcopy(last_run)
        run["run_id"] = f"{last_run['run_id']}-planted-{i}"
        run["kind"] = DRY_RUN
        for qid in PLANTED_QUESTIONS:
            res = _results(run).get(qid)
            if res is not None:
                res["refused"] = True
                res["pass"] = False
        planted.append(run)
    return planted


def simulate_delta(pass_rates: dict[str, float], *, seed: int = 0, draws: int = 10_000) -> float:
    """99th percentile of (recent fail rate - baseline fail rate) under the given pass rates.

    Each draw takes a recent window of RECENT_RUNS and an independent baseline window of
    BASELINE_MAX_RUNS runs, every question passing independently with its own rate. The result
    is floored at DELTA_FLOOR.
    """
    rng = random.Random(seed)
    fail_probs = [1.0 - p for p in pass_rates.values()]
    n_recent = RECENT_RUNS * len(fail_probs)
    n_base = BASELINE_MAX_RUNS * len(fail_probs)
    diffs = []
    for _ in range(draws):
        fail_recent = fail_base = 0
        for q in fail_probs:
            fail_recent += sum(rng.random() < q for _ in range(RECENT_RUNS))
            fail_base += sum(rng.random() < q for _ in range(BASELINE_MAX_RUNS))
        diffs.append(fail_recent / n_recent - fail_base / n_base)
    p99 = percentile(diffs, 0.99)
    return max(DELTA_FLOOR, p99 if p99 is not None else DELTA_FLOOR)
