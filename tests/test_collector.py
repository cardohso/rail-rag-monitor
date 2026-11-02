import json
from datetime import date
from pathlib import Path

import httpx
import pytest

from rail_rag_monitor.collector import TrafficDay, collect_day, summarize_day
from rail_rag_monitor.langfuse_api import LangfuseAPI
from rail_rag_monitor.store import read_jsonl, upsert_jsonl

FIXTURES = Path(__file__).parent / "fixtures"
DAY = date(2026, 10, 20)


def load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def roots():
    return load("roots_day.json")


def gens():
    return load("generations_day.json")


def test_summary_from_fixture():
    s = summarize_day(DAY, roots(), gens(), set())
    assert isinstance(s, TrafficDay)
    assert s.day == "2026-10-20"
    assert s.n_requests == 6
    assert s.n_errors == 1
    assert s.refusal_rate == 0.4
    assert s.refusals_by_reason == {"low_retrieval_score": 1, "no_citations": 1}
    assert s.grounding_failure_share == 0.5
    # latency over the 5 non-error roots: 1, 2, 3, 4, 5
    assert s.p50_s == 3.0
    assert s.p95_s == 4.8


def test_excludes_probe_traces():
    s = summarize_day(DAY, roots(), gens(), {"t1", "t6"})
    assert s.n_requests == 4
    assert s.n_errors == 0
    assert s.refusal_rate == 0.5
    generation = [
        {"traceId": "t1", "usageDetails": {"input": 1}, "totalUsage": 100, "totalCost": 1.0},
        {"traceId": "t2", "usageDetails": {"input": 1}, "totalUsage": 10, "totalCost": 0.5},
    ]
    s2 = summarize_day(DAY, roots(), generation, {"t1"})
    assert s2.n_requests == 5
    assert s2.tokens_per_request == 10 / 5
    assert s2.cost_per_request == 0.5 / 5


def test_output_as_json_string():
    as_strings = [
        {**r, "output": json.dumps(r["output"]) if r["output"] is not None else None}
        for r in roots()
    ]
    assert summarize_day(DAY, as_strings, gens(), set()) == summarize_day(
        DAY, roots(), gens(), set()
    )


def test_missing_output_counted_but_not_in_rates():
    data = roots()
    data.append({**data[0], "id": "r7", "traceId": "t7", "output": None})
    s = summarize_day(DAY, data, gens(), set())
    assert s.n_requests == 7
    assert s.n_errors == 1
    assert s.refusal_rate == 0.4  # still 2 of the 5 non-error roots with an output


def test_no_refusals_gives_none_shares():
    answered = [r for r in roots() if r["output"] and not r["output"]["refused"]]
    s = summarize_day(DAY, answered, [], set())
    assert s.refusal_rate == 0.0
    assert s.refusals_by_reason == {}
    assert s.grounding_failure_share is None
    assert s.tokens_per_request is None


def test_empty_day():
    s = summarize_day(DAY, [], [], set())
    assert s.n_requests == 0
    assert s.refusal_rate is None
    assert s.p50_s is None
    assert s.p95_s is None
    assert s.cost_per_request is None


def test_usage_missing_gives_none():
    s = summarize_day(DAY, roots(), gens(), set())
    assert s.tokens_per_request is None
    assert s.cost_per_request is None


def test_usage_present_is_summed():
    generation = [
        {"traceId": "t1", "usageDetails": {"input": 5}, "totalUsage": 30, "totalCost": 0.25},
        {"traceId": "t2", "usageDetails": {"input": 5}, "totalUsage": 30, "totalCost": 0.25},
    ]
    s = summarize_day(DAY, roots(), generation, set())
    assert s.tokens_per_request == 60 / 6
    assert s.cost_per_request == 0.5 / 6


def test_cost_none_when_usage_present_but_cost_null():
    generation = [
        {"traceId": "t1", "usageDetails": {"input": 5}, "totalUsage": 30, "totalCost": None}
    ]
    s = summarize_day(DAY, roots(), generation, set())
    assert s.tokens_per_request == 30 / 6
    assert s.cost_per_request is None


def test_unreadable_output_tolerated():
    data = roots()
    data.append({**data[0], "id": "r8", "traceId": "t8", "output": "not json at all"})
    s = summarize_day(DAY, data, gens(), set())
    assert s.n_requests == 7
    assert s.refusal_rate == 0.4


def test_no_text_in_row():
    s = summarize_day(DAY, roots(), gens(), set())
    dumped = json.dumps(s.to_dict())
    assert "FAKE-" not in dumped
    assert "ANSWER" not in dumped
    assert "QUESTION" not in dumped
    assert "PASSAGE" not in dumped
    assert set(s.to_dict()) == {
        "day",
        "n_requests",
        "n_errors",
        "refusal_rate",
        "refusals_by_reason",
        "grounding_failure_share",
        "p50_s",
        "p95_s",
        "tokens_per_request",
        "cost_per_request",
    }


def test_collect_day_window():
    seen = []

    def handler(request):
        seen.append(request.url.params)
        if request.url.params.get("type") == "GENERATION":
            return httpx.Response(200, json={"data": gens(), "meta": {}})
        return httpx.Response(200, json={"data": roots(), "meta": {}})

    api = LangfuseAPI("pk", "sk", "https://lf.test", transport=httpx.MockTransport(handler))
    s = collect_day(api, DAY, {"t1"})
    assert s.n_requests == 5

    root_req = next(p for p in seen if p.get("type") != "GENERATION")
    gen_req = next(p for p in seen if p.get("type") == "GENERATION")
    for p in (root_req, gen_req):
        assert p["environment"] == "azure-container-apps"
        assert p["fromStartTime"] == "2026-10-20T00:00:00Z"
        assert p["toStartTime"] == "2026-10-21T00:00:00Z"
    assert root_req["isRootObservation"] == "true"
    assert root_req["name"] == "rag-ask"
    assert root_req["fields"] == "core,basic,io,metrics"
    assert gen_req["fields"] == "core,usage"


def test_recollect_replaces_day(tmp_path):
    path = tmp_path / "metrics" / "traffic_daily.jsonl"
    first = summarize_day(DAY, roots(), gens(), set())
    second = summarize_day(DAY, roots(), gens(), {"t1"})
    upsert_jsonl(path, [first.to_dict()], key="day")
    upsert_jsonl(path, [second.to_dict()], key="day")
    rows = read_jsonl(path)
    assert len(rows) == 1
    assert rows[0]["n_requests"] == 5


def refused_root(i, reason, **extra):
    return {
        "id": f"x{i}",
        "traceId": f"tx{i}",
        "level": "DEFAULT",
        "latency": 1.0,
        "output": {"refused": True, "refusal_reason": reason, "answer": "FAKE-ANSWER"},
        **extra,
    }


def test_refusal_reason_is_limited_to_known_values():
    hostile = "SECRET-FREE-TEXT"
    data = [
        refused_root(1, "low_retrieval_score"),
        refused_root(2, "no_relevant_passages"),
        refused_root(3, "insufficient_context"),
        refused_root(4, "no_citations"),
        refused_root(5, hostile),
        refused_root(6, [hostile]),
        refused_root(7, {"k": hostile}),
        refused_root(8, 42),
        refused_root(9, None),
        refused_root(10, ""),
    ]
    s = summarize_day(DAY, data, [], set())
    assert s.refusals_by_reason == {
        "low_retrieval_score": 1,
        "no_relevant_passages": 1,
        "insufficient_context": 1,
        "no_citations": 1,
        "other": 4,
        "unknown": 2,
    }
    assert "SECRET" not in json.dumps(s.to_dict())
    # unknown reasons are not grounding failures: only the two grounding reasons count
    assert s.grounding_failure_share == 2 / 10


def test_generations_only_count_for_kept_rag_ask_roots():
    generation = [
        {"traceId": "t1", "usageDetails": {"i": 1}, "totalUsage": 10, "totalCost": 1.0},
        {"traceId": "t2", "usageDetails": {"i": 1}, "totalUsage": 20, "totalCost": 2.0},
        # a trace that has no rag-ask root in the day (another app, or a root outside the day)
        {"traceId": "other-trace", "usageDetails": {"i": 1}, "totalUsage": 900, "totalCost": 90.0},
        {"usageDetails": {"i": 1}, "totalUsage": 900, "totalCost": 90.0},
    ]
    s = summarize_day(DAY, roots(), generation, {"t1"})
    assert s.n_requests == 5
    assert s.tokens_per_request == 20 / 5
    assert s.cost_per_request == 2.0 / 5


def test_all_error_day_has_no_rates_or_latency():
    data = [r for r in roots() if r["level"] == "ERROR"]
    data.append({**data[0], "id": "r9", "traceId": "t9"})
    s = summarize_day(DAY, data, [], set())
    assert (s.n_requests, s.n_errors) == (2, 2)
    assert s.refusal_rate is None
    assert s.grounding_failure_share is None
    assert s.refusals_by_reason == {}
    assert s.p50_s is None and s.p95_s is None


def test_root_without_latency_is_left_out_of_percentiles():
    data = roots()
    data.append({**data[0], "id": "r7", "traceId": "t7", "latency": None})
    data.append({k: v for k, v in data[0].items() if k != "latency"} | {"traceId": "t8"})
    s = summarize_day(DAY, data, [], set())
    assert s.n_requests == 8
    assert s.p50_s == 3.0
    assert s.p95_s == 4.8


@pytest.mark.parametrize("bad", ["slow", [1], {"a": 1}, True, float("nan"), float("inf")])
def test_non_numeric_latency_is_skipped_not_a_crash(bad):
    data = roots()
    data.append({**data[0], "id": "r7", "traceId": "t7", "latency": bad})
    s = summarize_day(DAY, data, [], set())
    assert s.n_requests == 7
    assert s.p50_s == 3.0
    assert s.p95_s == 4.8


def test_numeric_string_latency_is_still_read():
    data = [{**roots()[0], "latency": "2.5"}]
    assert summarize_day(DAY, data, [], set()).p50_s == 2.5


def test_insufficient_context_counts_as_a_grounding_failure():
    data = [
        refused_root(1, "insufficient_context"),
        refused_root(2, "low_retrieval_score"),
        refused_root(3, "no_relevant_passages"),
        refused_root(4, "insufficient_context"),
    ]
    s = summarize_day(DAY, data, [], set())
    assert s.refusals_by_reason["insufficient_context"] == 2
    assert s.grounding_failure_share == 2 / 4
