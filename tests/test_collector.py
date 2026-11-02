import json
from datetime import date
from pathlib import Path

import httpx

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
