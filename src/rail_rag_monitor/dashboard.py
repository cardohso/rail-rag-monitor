"""Static dashboard (spec section 8): one self-contained `index.html` with the data embedded.

Privacy: everything that reaches the page goes through an allow-list. Each stored row is rebuilt
from named fields only, with each value type-checked, so an unexpected key (answer text, a
question, a trace id, free text) is dropped rather than published. Per-question trace ids are
never embedded: they would be public links to private traces. The JSON is escaped for a
`<script>` element and the page script writes data into the DOM with `textContent` only.
"""

from __future__ import annotations

import html
import json
import math
import re
from importlib import resources
from pathlib import Path

from rail_rag_monitor.alerts import (
    ALPHA,
    BASELINE_MAX_RUNS,
    CLEAN_RUNS_TO_CLOSE,
    DELTA_FLOOR,
    LATENCY_FACTOR,
    MIN_BASELINE_RUNS,
    RECENT_RUNS,
)

MAX_RUNS = 90  # the page plots the latest runs; the rules only look at 17
MAX_TRAFFIC_DAYS = 90
MAX_TEXT = 64

REFUSAL_REASONS = (
    "low_retrieval_score",
    "no_relevant_passages",
    "insufficient_context",
    "no_citations",
    "other",
    "unknown",
)
ALERT_RULES = ("uptime", "quality", "latency")

_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_QUESTION_ID = re.compile(r"[A-Za-z0-9_.-]{1,16}")
_PLACEHOLDER = re.compile(r"__([A-Z_]+)__")

RUN_NUMBERS = ("n_pass", "n_scored", "n_errors")
RUN_LATENCIES = ("warm_p50_s", "warm_p95_s", "cold_latency_s")
TRAFFIC_COUNTS = ("n_requests", "n_errors")
TRAFFIC_NUMBERS = (
    "refusal_rate",
    "grounding_failure_share",
    "p50_s",
    "p95_s",
    "tokens_per_request",
    "cost_per_request",
)


def _number(value: object) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return value if math.isfinite(value) else None


def _count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _text(value: object) -> str | None:
    return value[:MAX_TEXT] if isinstance(value, str) else None


def _boolean(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _issue(value: object) -> int | None:
    count = _count(value)
    return count if count else None


def _clean_result(result: object) -> dict:
    result = result if isinstance(result, dict) else {}
    return {
        "pass": _boolean(result.get("pass")),
        "status": _text(result.get("status")),
        "latency_s": _number(result.get("latency_s")),
        "cold": _boolean(result.get("cold")) or False,
    }


def _clean_run(row: object) -> dict | None:
    if not isinstance(row, dict):
        return None
    run_id, started_at = _text(row.get("run_id")), _text(row.get("started_at"))
    if not run_id or not started_at:
        return None
    results = row.get("results")
    results = results if isinstance(results, dict) else {}
    clean = {
        "run_id": run_id,
        "started_at": started_at,
        "kind": _text(row.get("kind")),
        "demo_version": _text(row.get("demo_version")),
        "results": {
            qid: _clean_result(res)
            for qid, res in results.items()
            if isinstance(qid, str) and _QUESTION_ID.fullmatch(qid)
        },
    }
    for key in RUN_NUMBERS:
        clean[key] = _count(row.get(key))
    for key in RUN_LATENCIES:
        clean[key] = _number(row.get(key))
    return clean


def _clean_day(row: object) -> dict | None:
    if not isinstance(row, dict):
        return None
    day = _text(row.get("day"))
    if not day:
        return None
    reasons = row.get("refusals_by_reason")
    reasons = reasons if isinstance(reasons, dict) else {}
    clean: dict = {"day": day}
    for key in TRAFFIC_COUNTS:
        clean[key] = _count(row.get(key))
    for key in TRAFFIC_NUMBERS:
        clean[key] = _number(row.get(key))
    clean["refusals_by_reason"] = {
        reason: count
        for reason in REFUSAL_REASONS
        if (count := _count(reasons.get(reason))) is not None
    }
    return clean


def _clean_alerts(state: object) -> dict:
    state = state if isinstance(state, dict) else {}
    rules = state.get("rules")
    rules = rules if isinstance(rules, dict) else {}
    dry = state.get("dry_run_issues")
    dry = dry if isinstance(dry, dict) else {}
    open_alerts = []
    for rule in ALERT_RULES:
        entry = rules.get(rule)
        issue = _issue(entry.get("issue")) if isinstance(entry, dict) else None
        if issue:
            open_alerts.append(
                {"rule": rule, "issue": issue, "clean_runs": _count(entry.get("clean_runs")) or 0}
            )
    dry_run = [
        {"rule": rule, "issue": issue} for rule in ALERT_RULES if (issue := _issue(dry.get(rule)))
    ]
    return {"open": open_alerts, "dry_run": dry_run}


def _clean_noise(noise: object) -> dict | None:
    if not isinstance(noise, dict):
        return None
    delta = _number(noise.get("delta"))
    if delta is None:
        return None

    def per_question(values: object, convert) -> dict:
        values = values if isinstance(values, dict) else {}
        return {
            qid: v
            for qid, raw in values.items()
            if isinstance(qid, str)
            and _QUESTION_ID.fullmatch(qid)
            and (v := convert(raw)) is not None
        }

    return {
        "delta": delta,
        "n_runs": _count(noise.get("n_runs")),
        "pass_rates": per_question(noise.get("pass_rates"), _number),
        "flips": per_question(noise.get("flips"), _count),
        "computed_at": _text(noise.get("computed_at")),
    }


def _embed_json(payload: dict) -> str:
    """JSON that is safe inside `<script type="application/json">`."""
    text = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    for raw, escaped in (
        ("<", "\\u003c"),
        (">", "\\u003e"),
        ("&", "\\u0026"),
        (" ", "\\u2028"),
        (" ", "\\u2029"),
    ):
        text = text.replace(raw, escaped)
    return text


def _empty_note(message: str, present: bool) -> str:
    return "" if present else f'<p class="empty">{html.escape(message)}</p>'


def _usage_note(days: list[dict]) -> str:
    reported = sum(
        d["tokens_per_request"] is not None or d["cost_per_request"] is not None for d in days
    )
    if reported == 0:
        return "Tokens and cost per request: not reported by the app."
    return (
        f"Tokens and cost per request: reported on {reported} of {len(days)} days "
        "(see the table below); days without them show a dash."
    )


def build_site(
    probe_runs: list[dict],
    traffic_days: list[dict],
    alert_state: dict,
    noise: dict | None,
    out_dir: Path,
    *,
    repo: str,
) -> Path:
    """Write `out_dir/index.html` from the stored data and return its path."""
    if not _REPO.fullmatch(repo):
        raise ValueError(f"repo must look like 'owner/name', got {repo!r}")

    runs = [r for r in map(_clean_run, probe_runs) if r][-MAX_RUNS:]
    days = [d for d in map(_clean_day, traffic_days) if d][-MAX_TRAFFIC_DAYS:]
    clean_noise = _clean_noise(noise)
    delta = clean_noise["delta"] if clean_noise else DELTA_FLOOR
    delta_note = (
        f"measured from {clean_noise['n_runs']} noise runs"
        if clean_noise and clean_noise["n_runs"] is not None
        else "measured from probe noise"
        if clean_noise
        else "the 2/24 floor: no noise report yet"
    )

    payload = {
        "repo": repo,
        "runs": runs,
        "traffic": days,
        "alerts": _clean_alerts(alert_state),
        "noise": clean_noise,
        "params": {
            "min_baseline_runs": MIN_BASELINE_RUNS,
            "recent_runs": RECENT_RUNS,
            "baseline_max_runs": BASELINE_MAX_RUNS,
            "alpha": ALPHA,
            "latency_factor": LATENCY_FACTOR,
            "clean_runs_to_close": CLEAN_RUNS_TO_CLOSE,
            "delta": delta,
        },
    }

    values = {
        "REPO": html.escape(repo),
        "DELTA": f"{delta:.3f}",
        "DELTA_NOTE": html.escape(delta_note),
        "MIN_BASELINE": str(MIN_BASELINE_RUNS),
        "RECENT": str(RECENT_RUNS),
        "BASELINE_MAX": str(BASELINE_MAX_RUNS),
        "LATENCY_FACTOR": f"{LATENCY_FACTOR:g}",
        "CLEAN_TO_CLOSE": str(CLEAN_RUNS_TO_CLOSE),
        "RUNS_EMPTY": _empty_note("no runs yet", bool(runs)),
        "TRAFFIC_EMPTY": _empty_note("no traffic collected yet", bool(days)),
        "USAGE_NOTE": html.escape(_usage_note(days)),
        "DATA_JSON": _embed_json(payload),
    }
    template = (
        resources.files("rail_rag_monitor")
        .joinpath("templates/index.html")
        .read_text(encoding="utf-8")
    )
    # One pass, so a placeholder-looking string inside the data is never substituted.
    page = _PLACEHOLDER.sub(lambda m: values[m.group(1)], template)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "index.html"
    path.write_text(page, encoding="utf-8")
    return path
