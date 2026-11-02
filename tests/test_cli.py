import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from rail_rag_monitor import cli
from rail_rag_monitor.alerts import DELTA_FLOOR
from rail_rag_monitor.store import (
    ALERT_STATE,
    NOISE,
    PROBE_RUNS,
    TRAFFIC_DAILY,
    read_json,
    read_jsonl,
    upsert_jsonl,
)

ROOT = Path(__file__).parent.parent
GOLDEN = ROOT / "probe" / "golden.jsonl"
IDS = ["q01", "q06", "q09", "q10", "q14", "q15", "q17", "q18"]
REFUSE = {"q17", "q18"}
ITEMS = {
    json.loads(line)["id"]: json.loads(line)
    for line in GOLDEN.read_text(encoding="utf-8").splitlines()
    if line.strip()
}
QUESTION_TO_ID = {item["question"]: qid for qid, item in ITEMS.items()}

NOW = datetime(2026, 11, 2, 6, 17, 30, tzinfo=UTC)
ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,100}")

LF_PUBLIC = "pk-lf-PUBLIC-SENTINEL"
LF_SECRET = "sk-lf-SECRET-SENTINEL"
GH_TOKEN = "ghs_TOKEN_SENTINEL"
SECRETS = (LF_PUBLIC, LF_SECRET, GH_TOKEN)
GH_ENV = {"GITHUB_REPOSITORY": "owner/repo", "GITHUB_TOKEN": GH_TOKEN}


def good_body(qid):
    if qid in REFUSE:
        return {
            "answer": "",
            "refused": True,
            "refusal_reason": "low_retrieval_score",
            "citations": [],
            "trace_id": f"{IDS.index(qid) + 1:032x}",
        }
    docs = ITEMS[qid]["source_doc_ids"]
    return {
        "answer": "an answer",
        "refused": False,
        "refusal_reason": None,
        "citations": [
            {
                "n": i + 1,
                "passage_id": f"{doc}:p{i}",
                "doc_id": doc,
                "document": "Some doc",
                "section": "1",
                "heading": "H",
                "passage": "P",
                "url": "https://example.invalid/x",
                "score": 0.9,
            }
            for i, doc in enumerate(docs)
        ],
        "trace_id": f"{IDS.index(qid) + 1:032x}",
    }


class Demo:
    def __init__(self, events=None):
        self.events = [] if events is None else events

    def handler(self, request):
        self.events.append(("req", request.url.path))
        if request.url.path == "/health":
            return httpx.Response(200, json={"version": "v-test"})
        qid = QUESTION_TO_ID[json.loads(request.content)["question"]]
        return httpx.Response(200, json=good_body(qid))

    @property
    def transport(self):
        return httpx.MockTransport(self.handler)


class GitHub:
    """Mock GitHub issues API. `fail_create_at` makes that (1-based) create return 500."""

    def __init__(self, fail_create_at=None):
        self.requests = []
        self.creates = 0
        self.fail_create_at = fail_create_at

    def handler(self, request):
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.method, request.url.path, body))
        if request.method == "POST" and request.url.path.endswith("/issues"):
            self.creates += 1
            if self.creates == self.fail_create_at:
                return httpx.Response(500, json={"message": "boom"})
            return httpx.Response(201, json={"number": 100 + self.creates})
        return httpx.Response(200, json={"state": "open"})

    @property
    def transport(self):
        return httpx.MockTransport(self.handler)

    def creates_made(self):
        return [r for r in self.requests if r[0] == "POST" and r[1].endswith("/issues")]


def stored_run(i, *, started_at=None, fails=(), errors=(), kind="scheduled"):
    results = {}
    for qid in IDS:
        failing = qid in fails
        errored = qid in errors
        results[qid] = {
            "pass": None if errored else not failing,
            "refused": None if errored else qid in REFUSE,
            "grounding": None if errored else not failing,
            "status": "timeout" if errored else "ok",
            "latency_s": None if errored else 1.0,
            "cold": qid == "q01",
            "trace_id": f"stored-{i}-{qid}",
        }
    return {
        "run_id": f"run-{i:02d}",
        "started_at": started_at
        or (datetime(2026, 10, 1, tzinfo=UTC) + timedelta(days=i)).isoformat(),
        "commit": "abc1234",
        "kind": kind,
        "demo_version": "v1",
        "results": results,
        "n_pass": 8,
        "n_scored": 8,
        "n_errors": 0,
        "warm_p50_s": 1.0,
        "warm_p95_s": 1.0,
        "cold_latency_s": 1.0,
    }


def clean_runs(n):
    return [stored_run(i) for i in range(n)]


def regression_runs():
    """14 clean runs, then 3 chronologically later runs with 4 fails each plus an error."""
    bad = [
        stored_run(14 + j, fails=("q01", "q06", "q09", "q10"), errors=("q14",)) for j in range(3)
    ]
    return clean_runs(14) + bad


def write_runs(data, runs):
    upsert_jsonl(data / PROBE_RUNS, runs, "run_id")


def run_cli(capsys, args, env=None, **hooks):
    code = cli.main(args, env or {}, **hooks)
    captured = capsys.readouterr()
    for secret in SECRETS:
        assert secret not in captured.out
        assert secret not in captured.err
    return code, captured.out, captured.err


def hooks(demo=None, github=None, langfuse=None, sleeps=None):
    transports = {}
    if demo:
        transports["demo"] = demo.transport
    if github:
        transports["github"] = github.transport
    if langfuse:
        transports["langfuse"] = langfuse
    sleeps = [] if sleeps is None else sleeps
    return {"_transports": transports, "_sleep": sleeps.append, "_now": lambda: NOW}


def test_collect_without_keys_skips(tmp_path, capsys):
    data = tmp_path / "data"
    code, out, _ = run_cli(capsys, ["collect", "--data", str(data)], {}, **hooks())
    assert code == 0
    assert "Langfuse keys not set: skipping collection" in out
    assert not data.exists()


def test_probe_then_alert_end_to_end(tmp_path, capsys):
    data = tmp_path / "data"
    write_runs(data, clean_runs(13))
    env = {"DEMO_URL": "https://demo.invalid", "GITHUB_SHA": "deadbeef"}
    code, _, _ = run_cli(
        capsys, ["probe", "--kind", "scheduled", "--data", str(data)], env, **hooks(Demo())
    )
    assert code == 0
    runs = read_jsonl(data / PROBE_RUNS)
    assert len(runs) == 14
    assert runs[-1]["commit"] == "deadbeef"
    assert runs[-1]["kind"] == "scheduled"
    assert runs[-1]["n_pass"] == 8
    before = (data / PROBE_RUNS).read_bytes()

    github = GitHub()
    code, out, _ = run_cli(
        capsys,
        ["alert", "--plant-regression", "--data", str(data)],
        GH_ENV,
        **hooks(github=github),
    )
    assert code == 0
    assert "quality" in out
    creates = github.creates_made()
    assert len(creates) == 1
    assert creates[0][2]["title"] == "[dry-run] [monitor] quality alert"
    assert (data / PROBE_RUNS).read_bytes() == before
    state = read_json(data / ALERT_STATE, {})
    assert state["dry_run_issues"] == {"quality": 101}
    assert all(not e["issue"] for e in state["rules"].values())


def test_probe_rerun_same_id_replaces(tmp_path, capsys):
    data = tmp_path / "data"
    env = {
        "DEMO_URL": "https://demo.invalid",
        "GITHUB_RUN_ID": "9001",
        "GITHUB_RUN_ATTEMPT": "2",
    }
    for _ in range(2):
        code, _, _ = run_cli(
            capsys, ["probe", "--kind", "manual", "--data", str(data)], env, **hooks(Demo())
        )
        assert code == 0
    runs = read_jsonl(data / PROBE_RUNS)
    assert [r["run_id"] for r in runs] == ["9001-2-1"]


def test_probe_requires_demo_url(tmp_path, capsys):
    code, _, err = run_cli(
        capsys, ["probe", "--kind", "manual", "--data", str(tmp_path / "d")], {}, **hooks()
    )
    assert code != 0
    assert "DEMO_URL" in err
    assert not (tmp_path / "d").exists()


def test_probe_repeat_waits_between_runs(tmp_path, capsys):
    data = tmp_path / "data"
    events = []
    demo = Demo(events)

    def sleep(seconds):
        events.append(("sleep", seconds))

    h = hooks(demo)
    h["_sleep"] = sleep
    code, _, _ = run_cli(
        capsys,
        ["probe", "--kind", "noise", "--repeat", "3", "--data", str(data)],
        {"DEMO_URL": "https://demo.invalid"},
        **h,
    )
    assert code == 0
    starts = [i for i, e in enumerate(events) if e == ("req", "/health")]
    assert len(starts) == 3
    for index in starts[1:]:
        assert events[index - 1][0] == "sleep"
        assert events[index - 1][1] >= 13
    runs = read_jsonl(data / PROBE_RUNS)
    ids = [r["run_id"] for r in runs]
    assert len(set(ids)) == 3
    assert all(ID_RE.fullmatch(i) for i in ids)
    assert {r["kind"] for r in runs} == {"noise"}
    assert ids[0].startswith("20261102T061730Z-")


def test_probe_ci_run_ids_follow_run_and_attempt(tmp_path, capsys):
    env = {
        "DEMO_URL": "https://demo.invalid",
        "GITHUB_RUN_ID": "77",
        "GITHUB_RUN_ATTEMPT": "1",
    }
    code, _, _ = run_cli(
        capsys,
        ["probe", "--kind", "noise", "--repeat", "2", "--data", str(tmp_path)],
        env,
        **hooks(Demo()),
    )
    assert code == 0
    assert [r["run_id"] for r in read_jsonl(tmp_path / PROBE_RUNS)] == ["77-1-1", "77-1-2"]


@pytest.mark.parametrize(
    "extra", [["--kind", "manual", "--repeat", "2"], ["--kind", "noise", "--repeat", "6"]]
)
def test_probe_repeat_limits(tmp_path, capsys, extra):
    code, _, _ = run_cli(
        capsys,
        ["probe", *extra, "--data", str(tmp_path)],
        {"DEMO_URL": "https://demo.invalid"},
        **hooks(Demo()),
    )
    assert code != 0
    assert not (tmp_path / PROBE_RUNS).exists()


def test_alert_saves_state_when_sync_fails(tmp_path, capsys):
    data = tmp_path / "data"
    write_runs(data, regression_runs())
    github = GitHub(fail_create_at=2)  # uptime is created, quality fails
    code, out, err = run_cli(
        capsys, ["alert", "--data", str(tmp_path / "data")], GH_ENV, **hooks(github=github)
    )
    assert code != 0
    assert "quality" in out + err
    state = read_json(data / ALERT_STATE, None)
    assert state is not None
    assert state["rules"]["uptime"] == {"issue": 101, "clean_runs": 0}
    assert not state["rules"].get("quality", {}).get("issue")

    # A retry comments on the saved uptime issue instead of opening a duplicate.
    retry = GitHub()
    code, _, _ = run_cli(capsys, ["alert", "--data", str(data)], GH_ENV, **hooks(github=retry))
    assert code == 0
    titles = [r[2]["title"] for r in retry.creates_made()]
    assert titles == ["[monitor] quality alert"]
    assert read_json(data / ALERT_STATE, {})["rules"]["uptime"]["issue"] == 101


def test_alert_sorts_runs_by_started_at(tmp_path, capsys):
    data = tmp_path / "data"
    # The file is newest-first: unsorted, its last three rows are the oldest clean runs.
    write_runs(data, list(reversed(regression_runs())))
    github = GitHub()
    code, _, _ = run_cli(capsys, ["alert", "--data", str(data)], GH_ENV, **hooks(github=github))
    assert code == 0
    titles = {r[2]["title"] for r in github.creates_made()}
    assert titles == {"[monitor] uptime alert", "[monitor] quality alert"}


def test_alert_uses_stored_noise_delta(tmp_path, capsys):
    data = tmp_path / "data"
    write_runs(data, regression_runs())
    (data / NOISE).parent.mkdir(parents=True, exist_ok=True)
    (data / NOISE).write_text(json.dumps({"delta": 0.9}), encoding="utf-8")
    github = GitHub()
    code, _, _ = run_cli(capsys, ["alert", "--data", str(data)], GH_ENV, **hooks(github=github))
    assert code == 0
    # 12/21 vs 0/112 is below a 0.9 margin: only the uptime rule fires.
    assert {r[2]["title"] for r in github.creates_made()} == {"[monitor] uptime alert"}
    assert DELTA_FLOOR < 0.9


def test_alert_without_github_env_skips_sync(tmp_path, capsys):
    data = tmp_path / "data"
    write_runs(data, regression_runs())
    code, out, _ = run_cli(capsys, ["alert", "--data", str(data)], {}, **hooks())
    assert code == 0
    assert "uptime" in out and "quality" in out
    assert "skipping" in out
    assert not (data / ALERT_STATE).exists()


def test_alert_plant_needs_a_stored_run(tmp_path, capsys):
    code, _, err = run_cli(
        capsys, ["alert", "--plant-regression", "--data", str(tmp_path)], GH_ENV, **hooks()
    )
    assert code != 0
    assert "no stored probe runs" in err


def test_alert_passes_run_url_and_project_id(tmp_path, capsys):
    data = tmp_path / "data"
    write_runs(data, regression_runs())

    def langfuse(request):
        assert request.url.path == "/api/public/projects"
        return httpx.Response(200, json={"data": [{"id": "proj-xyz"}]})

    github = GitHub()
    env = {
        **GH_ENV,
        "LANGFUSE_PUBLIC_KEY": LF_PUBLIC,
        "LANGFUSE_SECRET_KEY": LF_SECRET,
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_RUN_ID": "555",
    }
    code, _, _ = run_cli(
        capsys,
        ["alert", "--data", str(data)],
        env,
        **hooks(github=github, langfuse=httpx.MockTransport(langfuse)),
    )
    assert code == 0
    body = github.creates_made()[0][2]["body"]
    assert "https://github.com/owner/repo/actions/runs/555" in body
    assert "proj-xyz" in body


def test_alert_survives_project_id_failure(tmp_path, capsys):
    data = tmp_path / "data"
    write_runs(data, regression_runs())
    github = GitHub()
    env = {**GH_ENV, "LANGFUSE_PUBLIC_KEY": LF_PUBLIC, "LANGFUSE_SECRET_KEY": LF_SECRET}
    broken = httpx.MockTransport(lambda request: httpx.Response(401, json={}))
    code, _, _ = run_cli(
        capsys, ["alert", "--data", str(data)], env, **hooks(github=github, langfuse=broken)
    )
    assert code == 0
    assert len(github.creates_made()) == 2


def test_collect_excludes_stored_probe_trace_ids(tmp_path, capsys):
    data = tmp_path / "data"
    write_runs(data, [stored_run(0), stored_run(1)])

    def roots_for(request):
        params = request.url.params
        if params.get("isRootObservation") == "true":
            traces = ["stored-0-q01", "stored-1-q18", "real-1", "real-2"]
            rows = [
                {"traceId": t, "level": "DEFAULT", "output": {"refused": False}, "latency": 1.0}
                for t in traces
            ]
            return httpx.Response(200, json={"data": rows, "meta": {}})
        return httpx.Response(200, json={"data": [], "meta": {}})

    env = {"LANGFUSE_PUBLIC_KEY": LF_PUBLIC, "LANGFUSE_SECRET_KEY": LF_SECRET}
    args = ["collect", "--day", "2026-10-20", "--data", str(data)]
    for _ in range(2):
        code, _, _ = run_cli(capsys, args, env, **hooks(langfuse=httpx.MockTransport(roots_for)))
        assert code == 0
    rows = read_jsonl(data / TRAFFIC_DAILY)
    assert [r["day"] for r in rows] == ["2026-10-20"]
    assert rows[0]["n_requests"] == 2


def test_collect_defaults_to_yesterday_utc(tmp_path, capsys):
    seen = []

    def handler(request):
        seen.append(request.url.params.get("fromStartTime"))
        return httpx.Response(200, json={"data": [], "meta": {}})

    env = {"LANGFUSE_PUBLIC_KEY": LF_PUBLIC, "LANGFUSE_SECRET_KEY": LF_SECRET}
    code, _, _ = run_cli(
        capsys,
        ["collect", "--data", str(tmp_path)],
        env,
        **hooks(langfuse=httpx.MockTransport(handler)),
    )
    assert code == 0
    assert seen[0] == "2026-11-01T00:00:00Z"
    assert read_jsonl(tmp_path / TRAFFIC_DAILY)[0]["day"] == "2026-11-01"


def test_collect_failure_returns_nonzero_without_secrets(tmp_path, capsys):
    env = {"LANGFUSE_PUBLIC_KEY": LF_PUBLIC, "LANGFUSE_SECRET_KEY": LF_SECRET}
    broken = httpx.MockTransport(lambda request: httpx.Response(401, json={}))
    code, _, err = run_cli(
        capsys, ["collect", "--data", str(tmp_path)], env, **hooks(langfuse=broken)
    )
    assert code != 0
    assert "HTTP 401" in err
    assert not (tmp_path / TRAFFIC_DAILY).exists()


def test_noise_report_sorts_and_writes(tmp_path, capsys):
    data = tmp_path / "data"
    write_runs(data, list(reversed(clean_runs(4))))
    code, _, _ = run_cli(capsys, ["noise-report", "--data", str(data)], {}, **hooks())
    assert code == 0
    noise = read_json(data / NOISE, None)
    assert noise["n_runs"] == 4
    assert noise["pass_rates"]["q01"] == 1.0
    assert noise["computed_at"] == NOW.isoformat()


def test_build_site_writes_index(tmp_path, capsys):
    data = tmp_path / "data"
    write_runs(data, clean_runs(3))
    out_dir = tmp_path / "site"
    code, _, _ = run_cli(
        capsys,
        ["build-site", "--out", str(out_dir), "--data", str(data)],
        {"GITHUB_REPOSITORY": "owner/repo"},
        **hooks(),
    )
    assert code == 0
    page = (out_dir / "index.html").read_text(encoding="utf-8")
    assert "owner/repo" in page


def test_build_site_with_empty_data_dir(tmp_path, capsys):
    code, _, _ = run_cli(
        capsys,
        ["build-site", "--out", str(tmp_path / "site"), "--data", str(tmp_path / "none")],
        {},
        **hooks(),
    )
    assert code == 0
    assert (tmp_path / "site" / "index.html").exists()


def test_unknown_command_is_an_error(capsys):
    code, _, _ = run_cli(capsys, ["nope"], {}, **hooks())
    assert code != 0
