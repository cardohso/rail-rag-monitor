import json
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from rail_rag_monitor.dashboard import build_site

IDS = ["q01", "q06", "q09", "q10", "q14", "q15", "q17", "q18"]
REPO = "someone/rail-rag-monitor"
HOSTILE = "</script><script>alert(1)</script>"


def make_run(i, *, fails=(), kind="scheduled", **extra):
    results = {
        qid: {
            "pass": qid not in fails,
            "refused": qid in fails,
            "grounding": qid not in fails,
            "status": "ok",
            "latency_s": 2.5 + i / 100,
            "cold": qid == IDS[0],
            "trace_id": f"TRACE-{i}-{qid}",
        }
        for qid in IDS
    }
    return {
        "run_id": f"run-{i:03d}",
        "started_at": f"2026-11-{i:02d}T06:00:00+00:00",
        "commit": "COMMIT-SHA-123",
        "kind": kind,
        "demo_version": "1.4.0",
        "results": results,
        "n_pass": 8 - len(fails),
        "n_scored": 8,
        "n_errors": 0,
        "warm_p50_s": 2.5,
        "warm_p95_s": 4.0,
        "cold_latency_s": 31.0,
        **extra,
    }


def make_day(i, **extra):
    return {
        "day": f"2026-11-{i:02d}",
        "n_requests": 40 + i,
        "n_errors": 1,
        "refusal_rate": 0.1,
        "refusals_by_reason": {"low_retrieval_score": 3, "no_citations": 1},
        "grounding_failure_share": 0.25,
        "p50_s": 2.0,
        "p95_s": 5.0,
        "tokens_per_request": None,
        "cost_per_request": None,
        **extra,
    }


STATE = {
    "rules": {
        "quality": {"issue": 17, "clean_runs": 1},
        "uptime": {"issue": None, "clean_runs": 0},
    },
    "dry_run_issues": {"latency": 3},
}
NOISE = {
    "n_runs": 12,
    "pass_rates": {"q01": 0.9, "q06": 1.0},
    "delta": 0.1234,
    "flips": {"q01": 2, "q06": 0},
    "computed_at": "2026-11-05T06:00:00+00:00",
}
SECTIONS = ["Status", "Probe pass rate", "Question grid", "Latency", "Real traffic"]


def build(tmp_path, runs=None, days=None, state=None, noise=NOISE):
    runs = [make_run(i) for i in range(1, 11)] if runs is None else runs
    days = [make_day(i) for i in range(1, 6)] if days is None else days
    path = build_site(runs, days, STATE if state is None else state, noise, tmp_path, repo=REPO)
    return path, path.read_text(encoding="utf-8")


class _Scripts(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts: list[tuple[dict, str]] = []
        self._open: tuple[dict, list[str]] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self._open = (dict(attrs), [])

    def handle_data(self, data):
        if self._open is not None:
            self._open[1].append(data)

    def handle_endtag(self, tag):
        if tag == "script" and self._open is not None:
            self.scripts.append((self._open[0], "".join(self._open[1])))
            self._open = None


def scripts(html):
    parser = _Scripts()
    parser.feed(html)
    return parser.scripts


def blob(html):
    data = [body for attrs, body in scripts(html) if attrs.get("id") == "monitor-data"]
    assert len(data) == 1
    return json.loads(data[0])


def test_builds_from_fixture(tmp_path):
    path, html = build(tmp_path)
    assert path == tmp_path / "index.html"
    for heading in SECTIONS:
        assert re.search(rf"<h2[^>]*>\s*{heading}\s*(<|$)", html), heading
    assert "δ" in html
    assert "0.123" in html
    assert "p &lt; 0.01" in html
    assert "Fisher" in html
    assert "low volume: not alerted" in html
    assert "no runs yet" not in html
    assert "no traffic collected yet" not in html
    data = blob(html)
    assert [r["run_id"] for r in data["runs"]][:2] == ["run-001", "run-002"]
    assert data["alerts"]["open"] == [{"rule": "quality", "issue": 17, "clean_runs": 1}]
    assert data["alerts"]["dry_run"] == [{"rule": "latency", "issue": 3}]
    assert data["noise"]["delta"] == 0.1234
    assert data["repo"] == REPO


def test_pinned_cdn_scripts_have_integrity(tmp_path):
    _, html = build(tmp_path)
    cdn = [a for a, _ in scripts(html) if a.get("src")]
    assert len(cdn) == 3
    for attrs in cdn:
        assert re.fullmatch(
            r"https://cdn\.jsdelivr\.net/npm/[a-z-]+@\d+\.\d+\.\d+/build/[a-z.-]+\.js", attrs["src"]
        )
        assert attrs["integrity"].startswith("sha384-")
        assert attrs["crossorigin"] == "anonymous"


def test_allow_list_drops_unknown_fields(tmp_path):
    run = make_run(1, answer="SECRET-TEXT", question="SECRET-QUESTION")
    run["results"]["q01"]["answer"] = "SECRET-RESULT-TEXT"
    run["results"]["q01"]["citations"] = ["SECRET-CITATION"]
    day = make_day(1, question="SECRET-TRAFFIC-QUESTION", free_text="SECRET-FREE")
    day["refusals_by_reason"]["SECRET-REASON"] = 9
    state = {"rules": {"quality": {"issue": 4, "clean_runs": 0, "note": "SECRET-NOTE"}}}
    noise = {**NOISE, "answer": "SECRET-NOISE"}
    _, html = build(tmp_path, [run], [day], state, noise)
    assert "SECRET" not in html
    data = blob(html)
    assert set(data["runs"][0]) == {
        "run_id",
        "started_at",
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
    assert set(data["runs"][0]["results"]["q01"]) == {"pass", "status", "latency_s", "cold"}
    assert data["traffic"][0]["refusals_by_reason"] == {
        "low_retrieval_score": 3,
        "no_citations": 1,
    }


def test_trace_ids_and_commit_are_not_published(tmp_path):
    _, html = build(tmp_path)
    assert "TRACE-" not in html
    assert "COMMIT-SHA" not in html


def test_refusal_reasons_are_filtered_to_known_values(tmp_path):
    reasons = {
        "low_retrieval_score": 1,
        "no_relevant_passages": 2,
        "insufficient_context": 3,
        "no_citations": 4,
        "other": 5,
        "unknown": 6,
        "free text reason": 7,
        "<b>x</b>": 8,
    }
    _, html = build(tmp_path, days=[make_day(1, refusals_by_reason=reasons)])
    assert blob(html)["traffic"][0]["refusals_by_reason"] == {
        k: v for k, v in reasons.items() if k in list(reasons)[:6]
    }
    assert "free text reason" not in html


def test_wrong_types_are_dropped(tmp_path):
    run = make_run(1, n_pass="many", warm_p50_s=True, cold_latency_s=float("nan"))
    _, html = build(tmp_path, [run], [make_day(1, n_requests="lots", p95_s=[1])])
    data = blob(html)
    assert data["runs"][0]["n_pass"] is None
    assert data["runs"][0]["warm_p50_s"] is None
    assert data["runs"][0]["cold_latency_s"] is None
    assert data["traffic"][0]["n_requests"] is None
    assert data["traffic"][0]["p95_s"] is None
    assert "NaN" not in html


def test_rows_without_identity_are_dropped(tmp_path):
    runs = [make_run(1), {"started_at": "2026-11-02T00:00:00+00:00"}, "junk"]
    days = [make_day(1), {"n_requests": 3}, None]
    _, html = build(tmp_path, runs, days)
    data = blob(html)
    assert [r["run_id"] for r in data["runs"]] == ["run-001"]
    assert [d["day"] for d in data["traffic"]] == ["2026-11-01"]


def test_empty_store(tmp_path):
    _, html = build(tmp_path, runs=[], days=[], state={}, noise=None)
    assert "no runs yet" in html
    assert "no traffic collected yet" in html
    data = blob(html)
    assert data["runs"] == [] and data["traffic"] == []
    assert data["alerts"] == {"open": [], "dry_run": []}
    assert data["noise"] is None
    # without a noise report the floor (2/24) is shown and labelled as such
    assert "0.083" in html


def test_tokens_not_reported_copy(tmp_path):
    _, html = build(tmp_path)
    assert "not reported by the app" in html
    reported = [make_day(1, tokens_per_request=812.5, cost_per_request=0.0041)]
    _, html = build(tmp_path, days=reported)
    assert "not reported by the app" not in html
    assert blob(html)["traffic"][0]["tokens_per_request"] == 812.5


def test_hostile_value_cannot_break_out_of_the_script(tmp_path):
    runs = [make_run(i) for i in range(1, 4)]
    runs[0]["kind"] = HOSTILE
    runs[1]["demo_version"] = " <!--&--> "
    days = [make_day(1, day=HOSTILE)]
    _, html = build(tmp_path, runs, days)
    _, benign = build(tmp_path)
    # the same number of script elements as a benign page, and none contains the payload
    assert len(scripts(html)) == len(scripts(benign))
    for attrs, body in scripts(html):
        if attrs.get("id") != "monitor-data":
            assert "alert(1)" not in body
    assert html.count("</script>") == benign.count("</script>")
    assert " " not in html and " " not in html
    data = blob(html)
    # the data round-trips exactly, so nothing was lost by escaping
    assert data["runs"][0]["kind"] == HOSTILE
    assert data["runs"][1]["demo_version"] == " <!--&--> "
    assert data["traffic"][0]["day"] == HOSTILE


def test_template_never_uses_innerhtml():
    template = Path("src/rail_rag_monitor/templates/index.html").read_text(encoding="utf-8")
    assert "innerHTML" not in template
    assert "outerHTML" not in template
    assert "insertAdjacentHTML" not in template
    assert "document.write" not in template


def test_repo_must_look_like_owner_name(tmp_path):
    with pytest.raises(ValueError):
        build_site([], [], {}, None, tmp_path, repo='x"><script>alert(1)</script>')


def test_out_dir_is_created(tmp_path):
    out = tmp_path / "nested" / "site"
    path = build_site([], [], {}, None, out, repo=REPO)
    assert path.is_file()


def test_template_is_packaged():
    import importlib.resources as r

    assert r.files("rail_rag_monitor").joinpath("templates/index.html").is_file()
