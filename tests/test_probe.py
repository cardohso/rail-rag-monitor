import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from rail_rag_monitor.probe import (
    PROBE_GOLDEN,
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
            "trace_id": f"trace-{qid}",
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
        "trace_id": f"trace-{qid}",
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


def probe(demo, sleeps=None, kind="manual"):
    sleeps = [] if sleeps is None else sleeps
    return run_probe(
        BASE,
        GOLDEN,
        run_id="run-1",
        kind=kind,
        commit="abc1234",
        client=demo.client(),
        sleep=sleeps.append,
        now=lambda: FIXED_NOW,
        timer=demo.timer,
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
    assert run.n_errors == 1
    assert run.n_scored == 7
    assert run.n_pass == 7


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
        return 429 if qid == "q06" else good_body(qid)

    lat = {"q01": 40.0, "q09": 2.0, "q10": 3.0, "q14": 4.0, "q15": 5.0, "q17": 6.0, "q18": 7.0}
    demo = Demo(script, latency=lambda qid, attempt: lat.get(qid, 0.5))
    run = probe(demo)
    assert run.warm_p50_s == 4.5  # 2,3,4,5,6,7 (429 not counted)


def test_trace_ids_kept():
    run = probe(Demo())
    assert [run.results[q].trace_id for q in IDS] == [f"trace-{q}" for q in IDS]


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
