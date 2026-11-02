"""Daily production-traffic summary built from Langfuse root and generation observations.

Privacy: question, answer and citation text is read only in memory to derive
`refused`, `refusal_reason` and a citation count, then dropped. A `TrafficDay`
holds counts, rates and latencies only.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta

from rail_rag_monitor.langfuse_api import LangfuseAPI
from rail_rag_monitor.stats import percentile

ENVIRONMENT = "azure-container-apps"
ROOT_NAME = "rag-ask"
GROUNDING_REASONS = frozenset({"insufficient_context", "no_citations"})


@dataclass
class TrafficDay:
    day: str
    n_requests: int
    n_errors: int
    refusal_rate: float | None
    refusals_by_reason: dict[str, int] = field(default_factory=dict)
    grounding_failure_share: float | None = None
    p50_s: float | None = None
    p95_s: float | None = None
    tokens_per_request: float | None = None
    cost_per_request: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_output(output: object) -> dict | None:
    """Return the output as a dict, or None when absent or unreadable."""
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except ValueError:
            return None
    return output if isinstance(output, dict) else None


def _is_error(root: dict) -> bool:
    return root.get("level") == "ERROR"


def _tokens_per_request(generations: list[dict], n_requests: int) -> float | None:
    """None (not 0) until the app reports usage on at least one generation."""
    if n_requests == 0 or not any(g.get("usageDetails") for g in generations):
        return None
    return sum(g.get("totalUsage") or 0 for g in generations) / n_requests


def _cost_per_request(generations: list[dict], n_requests: int) -> float | None:
    """None (not 0) until the app reports a cost on at least one generation."""
    costs = [g["totalCost"] for g in generations if g.get("totalCost") is not None]
    if n_requests == 0 or not costs:
        return None
    return sum(costs) / n_requests


def summarize_day(
    day: date,
    roots: Iterable[dict],
    generations: Iterable[dict],
    exclude_trace_ids: set[str],
) -> TrafficDay:
    kept = [r for r in roots if r.get("traceId") not in exclude_trace_ids]
    kept_generations = [g for g in generations if g.get("traceId") not in exclude_trace_ids]
    n_requests = len(kept)
    n_errors = sum(1 for r in kept if _is_error(r))

    answered_ok = [r for r in kept if not _is_error(r)]
    outputs = [o for o in (_parse_output(r.get("output")) for r in answered_ok) if o is not None]
    reasons: dict[str, int] = {}
    n_refused = 0
    for out in outputs:
        if out.get("refused"):
            n_refused += 1
            reason = out.get("refusal_reason") or "unknown"
            reasons[reason] = reasons.get(reason, 0) + 1
    refusal_rate = n_refused / len(outputs) if outputs else None
    grounding = sum(count for reason, count in reasons.items() if reason in GROUNDING_REASONS)
    grounding_share = grounding / n_refused if n_refused else None

    latencies = [float(r["latency"]) for r in answered_ok if r.get("latency") is not None]

    return TrafficDay(
        day=day.isoformat(),
        n_requests=n_requests,
        n_errors=n_errors,
        refusal_rate=refusal_rate,
        refusals_by_reason=reasons,
        grounding_failure_share=grounding_share,
        p50_s=percentile(latencies, 0.5),
        p95_s=percentile(latencies, 0.95),
        tokens_per_request=_tokens_per_request(kept_generations, n_requests),
        cost_per_request=_cost_per_request(kept_generations, n_requests),
    )


def collect_day(api: LangfuseAPI, day: date, exclude_trace_ids: set[str]) -> TrafficDay:
    window = {
        "environment": ENVIRONMENT,
        "fromStartTime": f"{day.isoformat()}T00:00:00Z",
        "toStartTime": f"{(day + timedelta(days=1)).isoformat()}T00:00:00Z",
    }
    roots = list(
        api.iter_observations(
            **window, isRootObservation="true", name=ROOT_NAME, fields="core,basic,io,metrics"
        )
    )
    generations = list(api.iter_observations(**window, type="GENERATION", fields="core,usage"))
    return summarize_day(day, roots, generations, exclude_trace_ids)
