"""Static dashboard (spec section 8): one self-contained `index.html` with the data embedded.

Privacy: everything that reaches the page goes through an allow-list. Each stored row is rebuilt
from named fields only, and every value is checked by type and, for strings, by value (a closed
set or a strict pattern), so an unexpected key or a free-text value (answer text, a question, a
trace id) is dropped rather than published. Per-question trace ids are
never embedded: they would be public links to private traces. The JSON is escaped for a
`<script>` element and the page script writes data into the DOM with `textContent` only.
"""

from __future__ import annotations

import html
import json
import math
import re
from datetime import date, datetime
from importlib import resources
from pathlib import Path

from rail_rag_monitor.alerts import (
    ALPHA,
    BASELINE_MAX_RUNS,
    CLEAN_RUNS_TO_CLOSE,
    DELTA_FLOOR,
    LATENCY_FACTOR,
    LATENCY_RUNS,
    MIN_BASELINE_RUNS,
    RECENT_RUNS,
    UPTIME_RUNS,
)

MAX_RUNS = 90  # the page plots the latest runs; the rules only look at 17
MAX_TRAFFIC_DAYS = 90

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
_RUN_ID = re.compile(r"[A-Za-z0-9._:-]{1,100}")
_DEMO_VERSION = re.compile(r"[A-Za-z0-9._+-]{1,40}")
_STATUS = re.compile(r"ok|http_[0-9]{3}|timeout|connection|rate_limited|bad_response")
_DAY = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_TIMESTAMP = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?(Z|\+00:00)"
)
RUN_KINDS = frozenset({"scheduled", "manual", "noise", "dry-run"})

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


def _matching(pattern: re.Pattern[str], value: object) -> str | None:
    """The string itself when it fullmatches `pattern`, else None: never a transformed value."""
    return value if isinstance(value, str) and pattern.fullmatch(value) else None


def _status(value: object) -> str | None:
    return _matching(_STATUS, value)


def _kind(value: object) -> str | None:
    return value if isinstance(value, str) and value in RUN_KINDS else None


def _timestamp(value: object) -> str | None:
    """An ISO-8601 UTC timestamp that is also a real instant."""
    if _matching(_TIMESTAMP, value) is None:
        return None
    try:
        datetime.fromisoformat(value)  # type: ignore[arg-type]
    except ValueError:
        return None
    return value  # type: ignore[return-value]


def _day(value: object) -> str | None:
    if _matching(_DAY, value) is None:
        return None
    try:
        date.fromisoformat(value)  # type: ignore[arg-type]
    except ValueError:
        return None
    return value  # type: ignore[return-value]


def _boolean(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _issue(value: object) -> int | None:
    count = _count(value)
    return count if count else None


def _clean_result(result: object) -> dict:
    result = result if isinstance(result, dict) else {}
    return {
        "pass": _boolean(result.get("pass")),
        "status": _status(result.get("status")),
        "latency_s": _number(result.get("latency_s")),
        "cold": _boolean(result.get("cold")) or False,
    }


def _clean_run(row: object) -> dict | None:
    if not isinstance(row, dict):
        return None
    run_id = _matching(_RUN_ID, row.get("run_id"))
    started_at = _timestamp(row.get("started_at"))
    if not run_id or not started_at:
        return None
    results = row.get("results")
    results = results if isinstance(results, dict) else {}
    clean = {
        "run_id": run_id,
        "started_at": started_at,
        "kind": _kind(row.get("kind")),
        "demo_version": _matching(_DEMO_VERSION, row.get("demo_version")),
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
    day = _day(row.get("day"))
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
        "computed_at": _timestamp(noise.get("computed_at")),
    }


def _embed_json(payload: dict) -> str:
    """JSON that is safe inside `<script type="application/json">`."""
    text = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    for raw, escaped in (
        ("<", "\\u003c"),
        (">", "\\u003e"),
        ("&", "\\u0026"),
        ("\u2028", "\\u2028"),
        ("\u2029", "\\u2029"),
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

    # Oldest first, then keep the latest: the stored order is not trusted to be chronological.
    # The sort is stable, so rows with the same instant keep their stored order.
    runs = sorted(
        (r for r in map(_clean_run, probe_runs) if r),
        key=lambda r: datetime.fromisoformat(r["started_at"]),
    )[-MAX_RUNS:]
    days = sorted((d for d in map(_clean_day, traffic_days) if d), key=lambda d: d["day"])[
        -MAX_TRAFFIC_DAYS:
    ]
    clean_noise = _clean_noise(noise)
    delta = clean_noise["delta"] if clean_noise else DELTA_FLOOR
    delta_note = (
        f"measured from {clean_noise['n_runs']} noise and scheduled runs"
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
        "ALPHA": f"{ALPHA:g}",
        "UPTIME_RUNS": str(UPTIME_RUNS),
        "LATENCY_RUNS": str(LATENCY_RUNS),
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
