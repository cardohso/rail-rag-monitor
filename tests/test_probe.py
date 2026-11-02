import inspect
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from rail_rag_monitor.probe import (
    BACKOFF_503_S,
    DEADLINE_S,
    PROBE_GOLDEN,
    RETRIES_503,
    TIMEOUT_S,
    ProbeRun,
    QuestionResult,
    question_pass,
    run_probe,
)

ROOT = Path(__file__).parent.parent
GOLDEN = ROOT / "probe" / "golden.jsonl"
BASE = "https://demo.invalid"
IDS = ["q01", "q06", "q09", "q10", "q14", "q15", "q17", "q18"]
REFUSE = {"q17", "q18"}
FIXED_NOW = datetime(2026, 11, 2, 6, 0, tzinfo=UTC)


def trace_id(qid):
    """A trace id shaped like Langfuse's: lowercase hex."""
    return f"{IDS.index(qid) + 1:032x}"


ANSWER_TEXT = "ZZANSWER-distinctive-answer-text"
PASSAGE_TEXT = "ZZPASSAGE-distinctive-passage-text"
HEADING_TEXT = "ZZHEADING-distinctive-heading"


def load_items():
    return {
        json.loads(line)["id"]: json.loads(line)
        for line in GOLDEN.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


ITEMS = load_items()
QUESTION_TO_ID = {item["question"]: qid for qid, item in ITEMS.items()}


def good_body(qid):
    if qid in REFUSE:
        return {
            "answer": "",
            "refused": True,
            "refusal_reason": "low_retrieval_score",
            "citations": [],
            "trace_id": trace_id(qid),
        }
    docs = ITEMS[qid]["source_doc_ids"]
    return {
        "answer": ANSWER_TEXT,
        "refused": False,
        "refusal_reason": None,
        "citations": [
            {
                "n": i + 1,
                "passage_id": f"{doc}:p{i}",
                "doc_id": doc,
                "document": "Some doc",
                "section": "1",
                "heading": HEADING_TEXT,
                "passage": PASSAGE_TEXT,
                "url": "https://example.invalid/x",
                "score": 0.9,
            }
            for i, doc in enumerate(docs)
        ],
        "trace_id": trace_id(qid),
    }


class Demo:
    """Mock demo. `script(qid, attempt)` may return a body dict, an int status, or an exception."""

    def __init__(self, script=None, latency=1.0, health=None):
        self.script = script or (lambda qid, attempt: good_body(qid))
        self.latency = latency
        self.health = {"status": "ok", "version": "v-test-1"} if health is None else health
        self.clock = 0.0
        self.requests = []
        self.attempts = {}

    def timer(self):
        return self.clock

    def handler(self, request):
        self.requests.append((request.method, request.url.path))
        if request.method == "GET" and request.url.path == "/health":
            return httpx.Response(200, json=self.health)
        assert request.method == "POST" and request.url.path == "/ask"
        qid = QUESTION_TO_ID[json.loads(request.content)["question"]]
        attempt = self.attempts.get(qid, 0)
        self.attempts[qid] = attempt + 1
        out = self.script(qid, attempt)
        if isinstance(out, Exception):
            raise out
        self.clock += self.latency(qid, attempt) if callable(self.latency) else self.latency
        if isinstance(out, int):
            return httpx.Response(out, json={"detail": "x"})
        return httpx.Response(200, json=out)

    def client(self):
        return httpx.Client(transport=httpx.MockTransport(self.handler))

    def ask_count(self, qid):
        return self.attempts.get(qid, 0)


def probe(demo, sleeps=None, kind="manual", **kwargs):
    sleeps = [] if sleeps is None else sleeps
    kwargs.setdefault("sleep", sleeps.append)
    kwargs.setdefault("commit", "abc1234")
    return run_probe(
        BASE,
        GOLDEN,
        run_id="run-1",
        kind=kind,
        client=demo.client(),
        now=lambda: FIXED_NOW,
        timer=demo.timer,
        **kwargs,
    )


def test_golden_has_8_items():
    assert PROBE_GOLDEN == Path("probe/golden.jsonl")
    assert list(ITEMS) == IDS
    kinds = [item["expected"] for item in ITEMS.values()]
    assert kinds.count("answer") == 6
    assert kinds.count("refuse") == 2
    assert ITEMS["q14"]["tags"] == ["wrong-document"]
    assert ITEMS["q15"]["tags"] == ["wrong-document"]


def test_all_pass():
    run = probe(Demo())
    assert isinstance(run, ProbeRun)
    assert run.n_pass == 8
    assert run.n_scored == 8
    assert run.n_errors == 0
    assert run.run_id == "run-1"
    assert run.kind == "manual"
    assert run.commit == "abc1234"
    assert run.started_at == FIXED_NOW.isoformat()
    assert list(run.results) == IDS
    r = run.results["q01"]
    assert (r.pass_, r.refused, r.grounding, r.status) == (True, False, True, "ok")
    r = run.results["q17"]
    assert (r.pass_, r.refused, r.grounding, r.status) == (True, True, None, "ok")


def test_wrong_document_fails():
    def script(qid, attempt):
        body = good_body(qid)
        if qid == "q14":
            for c in body["citations"]:
                c["doc_id"] = "tsi-prm"
        return body

    run = probe(Demo(script))
    assert run.results["q14"].pass_ is False
    assert run.results["q14"].grounding is False
    assert run.n_pass == 7
    assert run.n_scored == 8


def test_refused_answerable_fails():
    def script(qid, attempt):
        if qid == "q01":
            return good_body("q17")
        return good_body(qid)

    run = probe(Demo(script))
    r = run.results["q01"]
    assert r.pass_ is False
    assert r.refused is True
    assert run.n_pass == 7


def test_answered_refusal_item_fails():
    def script(qid, attempt):
        return good_body("q01") if qid == "q17" else good_body(qid)

    run = probe(Demo(script))
    assert run.results["q17"].pass_ is False
    assert run.results["q17"].refused is False


def test_no_citations_is_a_grounding_fail():
    def script(qid, attempt):
        body = good_body(qid)
        if qid == "q06":
            body["citations"] = []
        return body

    run = probe(Demo(script))
    assert run.results["q06"].grounding is False
    assert run.results["q06"].pass_ is False


def test_pacing_and_health_first():
    demo = Demo()
    sleeps = []
    run = probe(demo, sleeps)
    assert demo.requests[0] == ("GET", "/health")
    assert [m for m, _ in demo.requests].count("GET") == 1
    assert len(sleeps) == 7  # between the 8 /ask calls, none before the first
    assert all(s >= 13 for s in sleeps)
    assert run.demo_version == "v-test-1"


def test_health_failure_does_not_stop_the_probe():
    demo = Demo(health={"status": "ok"})
    assert probe(demo).demo_version is None

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(500, text="boom")
        return httpx.Response(
            200, json=good_body(QUESTION_TO_ID[json.loads(request.content)["question"]])
        )

    run = run_probe(
        BASE,
        GOLDEN,
        run_id="r",
        kind="manual",
        commit="c",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda s: None,
        now=lambda: FIXED_NOW,
    )
    assert run.demo_version is None
    assert run.n_pass == 8


def test_429_not_retried():
    def script(qid, attempt):
        return 429 if qid == "q06" else good_body(qid)

    demo = Demo(script)
    run = probe(demo)
    r = run.results["q06"]
    assert r.status == "rate_limited"
    assert r.pass_ is None
    assert r.refused is None and r.grounding is None and r.latency_s is None
    assert demo.ask_count("q06") == 1
    # D1: the 429 also ends the run, so everything after q06 is recorded as rate_limited
    assert run.n_errors == 7
    assert run.n_scored == 1
    assert run.n_pass == 1


def test_503_retried_3_times():
    def script(qid, attempt):
        if qid == "q06" and attempt < 3:
            return 503
        return good_body(qid)

    demo = Demo(script)
    sleeps = []
    run = probe(demo, sleeps)
    assert run.results["q06"].status == "ok"
    assert run.results["q06"].pass_ is True
    assert demo.ask_count("q06") == 4
    assert sleeps.count(15) == 3
    assert run.n_errors == 0


def test_503_four_times_gives_up():
    def script(qid, attempt):
        return 503 if qid == "q06" else good_body(qid)

    demo = Demo(script)
    sleeps = []
    run = probe(demo, sleeps)
    r = run.results["q06"]
    assert r.status == "http_503"
    assert r.pass_ is None
    assert demo.ask_count("q06") == 4
    assert sleeps.count(15) == 3
    assert run.n_errors == 1


def test_other_http_error_not_retried():
    def script(qid, attempt):
        return 500 if qid == "q09" else good_body(qid)

    demo = Demo(script)
    run = probe(demo)
    assert run.results["q09"].status == "http_500"
    assert demo.ask_count("q09") == 1


def test_cold_timeout_retried_once():
    def script(qid, attempt):
        if qid == "q01" and attempt == 0:
            return httpx.ReadTimeout("slow")
        return good_body(qid)

    demo = Demo(script)
    run = probe(demo)
    assert demo.ask_count("q01") == 2
    assert run.results["q01"].status == "ok"
    assert run.results["q01"].cold is True
    assert all(not run.results[q].cold for q in IDS[1:])


def test_cold_timeout_twice_is_a_timeout():
    def script(qid, attempt):
        return httpx.ReadTimeout("slow") if qid == "q01" else good_body(qid)

    demo = Demo(script)
    run = probe(demo)
    assert demo.ask_count("q01") == 2
    assert run.results["q01"].status == "timeout"
    assert run.results["q01"].cold is True
    assert run.cold_latency_s is None


def test_later_timeout_not_retried():
    def script(qid, attempt):
        return httpx.ReadTimeout("slow") if qid == "q06" else good_body(qid)

    demo = Demo(script)
    run = probe(demo)
    assert demo.ask_count("q06") == 1
    assert run.results["q06"].status == "timeout"
    assert run.results["q06"].pass_ is None
    assert run.n_errors == 1


def test_connection_error_status():
    def script(qid, attempt):
        return httpx.ConnectError("down") if qid == "q09" else good_body(qid)

    demo = Demo(script)
    run = probe(demo)
    assert run.results["q09"].status == "connection"
    assert demo.ask_count("q09") == 1


def test_malformed_body_is_an_error_not_a_crash():
    def script(qid, attempt):
        return (
            {"answer": "x", "refused": False, "citations": "not-a-list"}
            if qid == "q09"
            else good_body(qid)
        )

    run = probe(Demo(script))
    assert run.results["q09"].status == "bad_response"
    assert run.results["q09"].pass_ is None
    assert run.n_errors == 1


def test_latency_split():
    lat = {
        "q01": 40.0,
        "q06": 1.0,
        "q09": 2.0,
        "q10": 3.0,
        "q14": 4.0,
        "q15": 5.0,
        "q17": 6.0,
        "q18": 7.0,
    }
    demo = Demo(latency=lambda qid, attempt: lat[qid])
    run = probe(demo)
    assert run.cold_latency_s == 40.0
    assert run.results["q01"].latency_s == 40.0
    # warm: 1..7 -> p50 4.0, p95 6.7
    assert run.warm_p50_s == 4.0
    assert run.warm_p95_s == pytest.approx(6.7)


def test_latency_split_excludes_errors():
    def script(qid, attempt):
        return 500 if qid == "q06" else good_body(qid)

    lat = {"q01": 40.0, "q09": 2.0, "q10": 3.0, "q14": 4.0, "q15": 5.0, "q17": 6.0, "q18": 7.0}
    demo = Demo(script, latency=lambda qid, attempt: lat.get(qid, 0.5))
    run = probe(demo)
    assert run.warm_p50_s == 4.5  # 2,3,4,5,6,7 (the error is not counted)


def test_trace_ids_kept():
    run = probe(Demo())
    assert [run.results[q].trace_id for q in IDS] == [trace_id(q) for q in IDS]


def test_no_answer_text_in_row():
    run = probe(Demo())
    text = json.dumps(run.to_dict(), ensure_ascii=False)
    for forbidden in (ANSWER_TEXT, PASSAGE_TEXT, HEADING_TEXT):
        assert forbidden not in text
    for item in ITEMS.values():
        assert item["question"] not in text
        assert item["reference_answer"] not in text


def test_to_dict_shape():
    run = probe(Demo())
    d = run.to_dict()
    assert set(d) == {
        "run_id",
        "started_at",
        "commit",
        "kind",
        "demo_version",
        "results",
        "n_pass",
        "n_scored",
        "n_errors",
        "warm_p50_s",
        "warm_p95_s",
        "cold_latency_s",
    }
    assert set(d["results"]["q01"]) == {
        "pass",
        "refused",
        "grounding",
        "status",
        "latency_s",
        "cold",
        "trace_id",
    }
    assert d["results"]["q01"]["cold"] is True
    assert json.loads(json.dumps(d)) == d


def test_question_result_to_dict_uses_pass_key():
    d = QuestionResult(True, False, True, "ok", 1.5, False, "t").to_dict()
    assert d["pass"] is True
    assert "pass_" not in d


@pytest.mark.parametrize(
    ("expected", "refused", "grounding", "want"),
    [
        ("answer", False, True, True),
        ("answer", False, False, False),
        ("answer", False, None, False),
        ("answer", True, None, False),
        ("answer", None, None, None),
        ("refuse", True, None, True),
        ("refuse", False, None, False),
        ("refuse", None, None, None),
    ],
)
def test_question_pass(expected, refused, grounding, want):
    assert question_pass(expected, refused, grounding) is want


# --- privacy: every stored string is from a closed set or pattern (P3) ----------------------------


def test_hostile_health_version_is_dropped():
    for hostile in ("<script>alert(1)</script>", "SECRET version", "v" * 41, ""):
        run = probe(Demo(health={"version": hostile}))
        assert run.demo_version is None
        assert "SECRET" not in json.dumps(run.to_dict())
    assert (
        probe(Demo(health={"version": "1.4.0+build.7-rc-1"})).demo_version == "1.4.0+build.7-rc-1"
    )


def test_hostile_trace_id_is_dropped():
    def script(qid, attempt):
        body = good_body(qid)
        body["trace_id"] = {"q01": "SECRET", "q06": "ABCDEF0123456789", "q09": "a" * 15}.get(
            qid, body["trace_id"]
        )
        return body

    run = probe(Demo(script))
    assert run.results["q01"].trace_id is None
    assert run.results["q06"].trace_id is None  # uppercase hex is not a Langfuse id
    assert run.results["q09"].trace_id is None  # too short
    assert run.results["q10"].trace_id == trace_id("q10")
    assert "SECRET" not in json.dumps(run.to_dict())


def test_commit_is_a_sha_or_local():
    assert probe(Demo(), commit="abc1234").commit == "abc1234"
    assert probe(Demo(), commit="a" * 40).commit == "a" * 40
    assert probe(Demo(), commit="local").commit == "local"
    for hostile in ("SECRET-COMMIT", "</script>", "abc123", "a" * 41, "ABC1234"):
        run = probe(Demo(), commit=hostile)
        assert run.commit == "local"
        assert "SECRET" not in json.dumps(run.to_dict())


def test_every_stored_string_matches_its_pattern():
    run = probe(Demo(lambda qid, attempt: 429 if qid == "q09" else good_body(qid)))
    d = run.to_dict()
    assert re.fullmatch(r"[0-9a-f]{7,40}|local", d["commit"])
    for res in d["results"].values():
        assert re.fullmatch(
            r"ok|http_[0-9]{3}|timeout|connection|rate_limited|bad_response", res["status"]
        )
        assert res["trace_id"] is None or re.fullmatch(r"[0-9a-f]{16,64}", res["trace_id"])


# --- demo safety (D1-D4) --------------------------------------------------------------------------


def test_first_429_stops_all_further_requests():
    def script(qid, attempt):
        return 429 if qid == "q09" else good_body(qid)

    demo = Demo(script)
    sleeps = []
    run = probe(demo, sleeps)
    posts = [r for r in demo.requests if r[0] == "POST"]
    assert len(posts) == 3  # q01, q06, q09 and nothing after the 429
    assert list(run.results) == IDS
    for qid in IDS[2:]:
        r = run.results[qid]
        assert r.status == "rate_limited"
        assert (r.pass_, r.refused, r.grounding, r.latency_s, r.trace_id) == (
            None,
            None,
            None,
            None,
            None,
        )
    assert [run.results[q].status for q in IDS[:2]] == ["ok", "ok"]
    assert run.n_errors == 6
    assert run.n_scored == 2
    # paced before q06 and q09 only: nothing is slept for the skipped questions
    assert sleeps == [13.0, 13.0]


def test_429_on_the_cold_question_skips_everything_else():
    demo = Demo(lambda qid, attempt: 429)
    sleeps = []
    run = probe(demo, sleeps)
    assert len([r for r in demo.requests if r[0] == "POST"]) == 1
    assert all(r.status == "rate_limited" for r in run.results.values())
    assert run.results["q01"].cold is True
    assert sleeps == []
    assert run.cold_latency_s is None


def test_deadline_skips_remaining_questions_as_timeouts():
    demo = Demo(latency=100.0)

    def sleep(seconds):
        demo.clock += seconds

    # clock after q: 100 per answer + 13 per pace. Before q06: 113; q09: 226; q10: 339 >= 300
    run = probe(demo, deadline_s=300.0, sleep=sleep)
    posts = [r for r in demo.requests if r[0] == "POST"]
    assert len(posts) == 3
    assert [run.results[q].status for q in IDS[:3]] == ["ok", "ok", "ok"]
    for qid in IDS[3:]:
        r = run.results[qid]
        assert r.status == "timeout"
        assert (r.pass_, r.refused, r.grounding, r.latency_s, r.trace_id) == (
            None,
            None,
            None,
            None,
            None,
        )
    assert run.n_errors == 5


def test_deadline_does_not_sleep_for_skipped_questions():
    demo = Demo(latency=400.0)
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        demo.clock += seconds

    run = probe(demo, deadline_s=300.0, sleep=sleep)
    assert [run.results[q].status for q in IDS] == ["ok"] + ["timeout"] * 7
    assert sleeps == [13.0]  # one pace before q06, after which the deadline had passed


def test_deadline_default_is_15_minutes_and_keyword_only():
    assert DEADLINE_S == 900
    param = inspect.signature(run_probe).parameters["deadline_s"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default == DEADLINE_S


def test_deadline_plus_worst_single_question_fits_the_step_limit():
    # The deadline is only checked before a question, so the last one can start at DEADLINE_S and
    # still run its full retry policy: 4 attempts of TIMEOUT_S plus RETRIES_503 backoffs (~525 s).
    worst_question_s = (RETRIES_503 + 1) * TIMEOUT_S + RETRIES_503 * BACKOFF_503_S
    step_limit_s = 25 * 60  # timeout-minutes of the workflow's probe step
    assert DEADLINE_S + worst_question_s < step_limit_s


def test_deadline_is_measured_from_the_start_of_the_run():
    demo = Demo(latency=1.0)
    demo.clock = 10_000.0  # a timer that does not start at zero
    run = probe(demo, deadline_s=60.0)
    assert run.n_errors == 0


def test_cold_latency_is_the_total_of_all_attempts():
    demo = None

    def script(qid, attempt):
        if qid == "q01" and attempt == 0:
            demo.clock += 120.0
            return httpx.ReadTimeout("slow")
        return good_body(qid)

    demo = Demo(script, latency=lambda qid, attempt: 3.0)
    run = probe(demo)
    assert run.results["q01"].status == "ok"
    assert run.results["q01"].latency_s == 3.0  # the final attempt only
    assert run.cold_latency_s == 123.0


def test_200_without_refused_key_is_a_bad_response():
    def script(qid, attempt):
        body = good_body(qid)
        if qid == "q09":
            del body["refused"]
        return body

    run = probe(Demo(script))
    r = run.results["q09"]
    assert r.status == "bad_response"
    assert r.pass_ is None and r.refused is None
    assert run.n_errors == 1
    assert run.n_scored == 7


# --- remaining coverage (D5) ----------------------------------------------------------------------


def test_pace_sleep_follows_a_timeout_question():
    demo = Demo(lambda qid, attempt: httpx.ReadTimeout("slow") if qid == "q06" else good_body(qid))
    sleeps = []
    probe(demo, sleeps)
    assert sleeps == [13.0] * 7


def test_pace_sleep_follows_an_exhausted_503_question():
    demo = Demo(lambda qid, attempt: 503 if qid == "q06" else good_body(qid))
    sleeps = []
    probe(demo, sleeps)
    assert sleeps.count(13.0) == 7
    assert sleeps.count(15.0) == 3
    assert len(sleeps) == 10


def test_single_warm_call_gives_that_value_for_both_percentiles():
    def script(qid, attempt):
        return good_body(qid) if qid in ("q01", "q09") else 500

    demo = Demo(script, latency=lambda qid, attempt: 7.5)
    run = probe(demo)
    assert run.warm_p50_s == 7.5
    assert run.warm_p95_s == 7.5


def test_no_warm_calls_gives_none():
    demo = Demo(lambda qid, attempt: good_body(qid) if qid == "q01" else 500)
    run = probe(demo)
    assert run.warm_p50_s is None
    assert run.warm_p95_s is None
    assert run.cold_latency_s == 1.0


def test_cold_non_timeout_error_leaves_cold_latency_none():
    for failure in (httpx.ConnectError("down"), 500, 503):
        demo = Demo(lambda qid, attempt, f=failure: f if qid == "q01" else good_body(qid))
        run = probe(demo)
        assert run.results["q01"].status != "ok"
        assert run.cold_latency_s is None
        assert run.warm_p50_s is not None
