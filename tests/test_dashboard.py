import json
import re
from datetime import UTC, date, datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path

import pytest

from rail_rag_monitor.dashboard import MAX_RUNS, MAX_TRAFFIC_DAYS, _embed_json, build_site

TEMPLATE = Path(__file__).parent.parent / "src" / "rail_rag_monitor" / "templates" / "index.html"
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


def test_embed_json_cannot_break_out_of_the_script():
    hostile = {"a": HOSTILE, "b": "\u2028<!--&-->\u2029", "c": ["</SCRIPT>", "x\u2028y"]}
    text = _embed_json(hostile)
    for raw in ("<", ">", "&", "\u2028", "\u2029"):
        assert raw not in text
    assert json.loads(text) == hostile  # nothing is lost by escaping


def test_page_has_no_raw_line_separators_and_one_script_set(tmp_path):
    _, html = build(tmp_path)
    assert "\u2028" not in html and "\u2029" not in html
    assert html.count("</script>") == len(scripts(html))


# (a space keeps it outside the run_id / demo_version character sets, which "SECRET-TEXT" is in)
HOSTILE_VALUES = ["SECRET TEXT", "SECRET</script><script>alert(1)</script>", "", "x" * 300]


def assert_nothing_published(tmp_path, runs=None, days=None, noise=NOISE):
    _, benign = build(tmp_path)
    _, html = build(tmp_path, runs, days, noise=noise)
    assert "SECRET" not in html
    assert "alert(1)" not in html
    assert "x" * 100 not in html
    assert html.count("</script>") == benign.count("</script>")
    assert len(scripts(html)) == len(scripts(benign))
    return html


@pytest.mark.parametrize("field", ["kind", "demo_version"])
@pytest.mark.parametrize("hostile", HOSTILE_VALUES)
def test_run_text_fields_are_constrained_by_value(tmp_path, field, hostile):
    run = make_run(1, **{field: hostile})
    html = assert_nothing_published(tmp_path, [run, make_run(2)])
    data = blob(html)
    assert [r["run_id"] for r in data["runs"]] == ["run-001", "run-002"]
    assert data["runs"][0][field] is None


@pytest.mark.parametrize("hostile", HOSTILE_VALUES)
def test_result_status_is_constrained_by_value(tmp_path, hostile):
    run = make_run(1)
    run["results"]["q06"]["status"] = hostile
    html = assert_nothing_published(tmp_path, [run])
    assert blob(html)["runs"][0]["results"]["q06"]["status"] is None


@pytest.mark.parametrize("field", ["run_id", "started_at"])
@pytest.mark.parametrize("hostile", HOSTILE_VALUES)
def test_run_identity_fields_are_constrained_by_value(tmp_path, field, hostile):
    runs = [make_run(1, **{field: hostile}), make_run(2)]
    html = assert_nothing_published(tmp_path, runs)
    assert [r["run_id"] for r in blob(html)["runs"]] == ["run-002"]


@pytest.mark.parametrize("hostile", HOSTILE_VALUES)
def test_day_is_constrained_by_value(tmp_path, hostile):
    html = assert_nothing_published(tmp_path, days=[make_day(1, day=hostile), make_day(2)])
    assert [d["day"] for d in blob(html)["traffic"]] == ["2026-11-02"]


@pytest.mark.parametrize("hostile", HOSTILE_VALUES)
def test_noise_computed_at_is_constrained_by_value(tmp_path, hostile):
    html = assert_nothing_published(tmp_path, noise={**NOISE, "computed_at": hostile})
    assert blob(html)["noise"]["computed_at"] is None


@pytest.mark.parametrize("kind", ["scheduled", "manual", "noise", "dry-run"])
def test_known_kinds_pass(tmp_path, kind):
    _, html = build(tmp_path, [make_run(1, kind=kind)])
    assert blob(html)["runs"][0]["kind"] == kind


@pytest.mark.parametrize("kind", ["Scheduled", "manual ", "dry_run", "other"])
def test_unknown_kinds_are_dropped(tmp_path, kind):
    _, html = build(tmp_path, [make_run(1, kind=kind)])
    assert blob(html)["runs"][0]["kind"] is None


@pytest.mark.parametrize(
    "status",
    ["ok", "http_200", "http_503", "timeout", "connection", "rate_limited", "bad_response"],
)
def test_known_statuses_pass(tmp_path, status):
    run = make_run(1)
    run["results"]["q01"]["status"] = status
    _, html = build(tmp_path, [run])
    assert blob(html)["runs"][0]["results"]["q01"]["status"] == status


@pytest.mark.parametrize("status", ["http_50", "http_5000", "http_abc", "OK", "ok ", "error"])
def test_unknown_statuses_are_dropped(tmp_path, status):
    run = make_run(1)
    run["results"]["q01"]["status"] = status
    _, html = build(tmp_path, [run])
    assert blob(html)["runs"][0]["results"]["q01"]["status"] is None


@pytest.mark.parametrize(
    "stamp",
    [
        "2026-11-02T06:00:00Z",
        "2026-11-02T06:00:00+00:00",
        "2026-11-02T06:00:00.123456+00:00",
        "2026-11-02T06:00:00.5Z",
    ],
)
def test_iso_timestamps_pass(tmp_path, stamp):
    _, html = build(
        tmp_path, [make_run(1, started_at=stamp)], noise={**NOISE, "computed_at": stamp}
    )
    data = blob(html)
    assert data["runs"][0]["started_at"] == stamp
    assert data["noise"]["computed_at"] == stamp


@pytest.mark.parametrize(
    "stamp",
    [
        "2026-11-02",
        "2026-11-02 06:00:00Z",
        "2026-11-02T06:00:00",
        "2026-11-02T06:00:00+01:00",
        "2026-13-02T06:00:00Z",
        "2026-11-02T25:00:00Z",
        "2026-11-02T06:00:00Z\n",
    ],
)
def test_non_iso_timestamps_are_dropped(tmp_path, stamp):
    _, html = build(
        tmp_path, [make_run(1, started_at=stamp)], noise={**NOISE, "computed_at": stamp}
    )
    data = blob(html)
    assert data["runs"] == []
    assert data["noise"]["computed_at"] is None


def test_run_id_and_version_patterns(tmp_path):
    ok = make_run(1, run_id="gh-123456789:1.2_a-b", demo_version="1.4.0+build.7-rc1")
    bad_id = make_run(2, run_id="has space")
    long_id = make_run(3, run_id="a" * 101)
    bad_version = make_run(4, demo_version="v" * 41)
    _, html = build(tmp_path, [ok, bad_id, long_id, bad_version])
    runs = blob(html)["runs"]
    assert [r["run_id"] for r in runs] == ["gh-123456789:1.2_a-b", "run-004"]
    assert runs[0]["demo_version"] == "1.4.0+build.7-rc1"
    assert runs[1]["demo_version"] is None


@pytest.mark.parametrize("day", ["2026-1-02", "26-11-02", "2026-11-02T00:00:00Z", "2026-13-40"])
def test_day_must_be_a_date(tmp_path, day):
    _, html = build(tmp_path, days=[make_day(1, day=day)])
    assert blob(html)["traffic"] == []


def test_runs_are_sorted_by_time_before_truncation(tmp_path):
    base = datetime(2026, 8, 1, 6, tzinfo=UTC)
    count = MAX_RUNS + 5
    runs = [
        make_run(1, run_id=f"r{i:03d}", started_at=(base + timedelta(hours=i)).isoformat())
        for i in range(count)
    ]
    runs.reverse()  # stored newest-first (or merged out of order)
    _, html = build(tmp_path, runs)
    kept = [r["run_id"] for r in blob(html)["runs"]]
    assert kept == [f"r{i:03d}" for i in range(5, count)]  # the latest 90, oldest first


def test_runs_sort_by_instant_not_by_text(tmp_path):
    runs = [
        make_run(1, run_id="b", started_at="2026-11-02T06:00:00Z"),
        make_run(2, run_id="a", started_at="2026-11-02T05:59:59.5+00:00"),
        make_run(3, run_id="c", started_at="2026-11-02T06:00:00.25Z"),
    ]
    _, html = build(tmp_path, runs)
    assert [r["run_id"] for r in blob(html)["runs"]] == ["a", "b", "c"]


def test_traffic_days_are_sorted_before_truncation(tmp_path):
    base = date(2026, 5, 1)
    count = MAX_TRAFFIC_DAYS + 4
    days = [make_day(1, day=(base + timedelta(days=i)).isoformat()) for i in range(count)]
    days.reverse()
    _, html = build(tmp_path, days=days)
    kept = [d["day"] for d in blob(html)["traffic"]]
    assert kept == [(base + timedelta(days=i)).isoformat() for i in range(4, count)]


def test_template_never_uses_innerhtml():
    template = TEMPLATE.read_text(encoding="utf-8")
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


def test_template_keeps_charts_full_width_on_phones():
    template = TEMPLATE.read_text(encoding="utf-8")
    # no text label in a reserved right margin: the plot keeps the full card width
    assert "width + 8" not in template
    # end-of-line labels are only drawn when the card is wide enough
    assert "isNarrow(" in template
    # y-axis titles short enough for a 190px plot, and bounded size legends
    assert "(% of refusals)" not in template
    assert "values: legendVals" in template


def test_status_tile_never_says_null_of_n():
    template = TEMPLATE.read_text(encoding="utf-8")
    assert '"no result"' in template
    # n_pass is checked before it is printed
    assert re.search(r"last\.n_pass\s*(===|!==)\s*null", template)


def test_legend_values_of_no_rows_is_empty_and_the_legend_is_omitted():
    template = TEMPLATE.read_text(encoding="utf-8")
    body = template[template.index("function legendValues") :]
    body = body[: body.index("function sizedDots")]
    assert re.search(r"if \(!\w+\.length\) return \[\];", body)
    assert re.search(r"legend:\s*legendVals\.length\s*\?", template)


def test_end_labels_show_on_the_desktop_two_column_cards():
    template = TEMPLATE.read_text(encoding="utf-8")
    assert re.search(r"NARROW_PX\s*=\s*440;", template)


def css_block(template, selector_start):
    start = template.index(selector_start)
    end = template.index("}", start)
    return template[start:end]


def test_theme_can_be_overridden_by_attribute():
    template = TEMPLATE.read_text(encoding="utf-8")
    media = template[template.index("@media (prefers-color-scheme: dark)") :]
    guarded = css_block(media, ':root:not([data-theme="light"]) {')
    forced = css_block(template, ':root[data-theme="dark"] {')
    assert css_block(template, ':root[data-theme="light"] {')
    # the same dark tokens in both, so the override cannot drift from the media query
    tokens = lambda block: sorted(re.findall(r"(--[a-z0-9-]+):\s*([^;]+);", block))  # noqa: E731
    assert tokens(guarded) == tokens(forced)
    assert len(tokens(guarded)) >= 10
    # every token the dark blocks override is defined by the light :root
    light = css_block(template, ":root {")
    defined = {name for name, _ in tokens(light)}
    assert {name for name, _ in tokens(guarded)} <= defined
    # and each token is declared once in the light block
    names = [name for name, _ in tokens(light)]
    assert len(names) == len(set(names))


def test_rule_copy_is_true_to_the_code(tmp_path):
    from rail_rag_monitor import alerts

    _, html = build(tmp_path)
    assert "None of them alert" not in html
    assert "no baseline" in html  # uptime needs none
    assert f"{alerts.UPTIME_RUNS} consecutive runs" in html
    assert f"{alerts.MIN_BASELINE_RUNS} baseline runs" in html
    assert f"{alerts.LATENCY_FACTOR:g}&times;" in html
    assert "daily probe only" in html and "never alerted" in html
    assert "rules that watch both" not in html
    assert "at least 3 failed questions of 24" in html
    assert "never binds" in html
    assert "no trace id" in html


def test_delta_note_counts_noise_and_scheduled_runs(tmp_path):
    _, html = build(tmp_path)
    assert "noise and scheduled runs" in html
    assert "from 12 noise runs" not in html
