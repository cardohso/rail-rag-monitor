import json

import httpx
import pytest

from rail_rag_monitor.alerts import Alert
from rail_rag_monitor.github_issues import (
    GitHubError,
    GitHubIssues,
    SyncError,
    render_body,
    sync,
    trace_url,
)

REPO = "owner/rail-demo"
TOKEN = "ghp_SECRETTOKEN123"
RUN_URL = "https://github.com/owner/rail-demo/actions/runs/42"


class FakeGitHub:
    """Records requests; `open_issues` is the set of issue numbers GitHub reports as open."""

    def __init__(self, open_issues=(), next_number=100):
        self.requests: list[httpx.Request] = []
        self.open_issues = set(open_issues)
        self.next_number = next_number

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if request.method == "POST" and path == f"/repos/{REPO}/issues":
            number = self.next_number
            self.next_number += 1
            self.open_issues.add(number)
            return httpx.Response(201, json={"number": number})
        if request.method == "POST" and path.endswith("/comments"):
            return httpx.Response(201, json={"id": 1})
        if request.method == "PATCH":
            return httpx.Response(200, json={"state": "closed"})
        if request.method == "GET":
            number = int(path.rsplit("/", 1)[1])
            state = "open" if number in self.open_issues else "closed"
            return httpx.Response(200, json={"number": number, "state": state})
        return httpx.Response(500)

    def gh(self) -> GitHubIssues:
        return GitHubIssues(REPO, TOKEN, transport=httpx.MockTransport(self))

    def calls(self) -> list[tuple[str, str]]:
        return [(r.method, r.url.path) for r in self.requests]

    def json_body(self, index: int) -> dict:
        return json.loads(self.requests[index].content)


def quality_alert(**extra) -> Alert:
    details = {
        "fail_recent": 9,
        "n_recent": 24,
        "fail_base": 3,
        "n_base": 112,
        "fail_rate_recent": 9 / 24,
        "fail_rate_base": 3 / 112,
        "p_value": 0.000123,
        "alpha": 0.01,
        "delta": 2 / 24,
        "recent_run_ids": ["run-015", "run-016", "run-017"],
        "baseline_run_ids": [f"run-{i:03d}" for i in range(1, 15)],
        "failed": [
            {"run_id": "run-016", "question_id": "q01", "trace_id": "tr-aaa"},
            {"run_id": "run-017", "question_id": "q06", "trace_id": "tr-bbb"},
        ],
        **extra,
    }
    return Alert("quality", "Fail rate rose.", details)


def uptime_alert() -> Alert:
    return Alert(
        "uptime",
        "errors",
        {
            "run_ids": ["run-001", "run-002"],
            "errors": [
                {"run_id": "run-001", "question_id": "q01", "status": "timeout", "trace_id": "t1"},
                {"run_id": "run-002", "question_id": "q02", "status": "http_500", "trace_id": None},
            ],
        },
    )


def latency_alert() -> Alert:
    return Alert(
        "latency",
        "slow",
        {
            "recent_warm_p95_s": [9.5, 10.25],
            "baseline_median_s": 3.0,
            "threshold_s": 6.0,
            "factor": 2.0,
            "n_base": 14,
            "recent_run_ids": ["run-016", "run-017"],
            "baseline_run_ids": ["run-001", "run-002"],
        },
    )


def run_sync(alerts, state, fake, *, dry_run=False, project_id="proj1"):
    return sync(alerts, state, fake.gh(), run_url=RUN_URL, project_id=project_id, dry_run=dry_run)


def test_opens_issue_when_none():
    fake = FakeGitHub()
    state = run_sync([quality_alert()], {"rules": {}}, fake)

    assert fake.calls() == [("POST", f"/repos/{REPO}/issues")]
    body = fake.json_body(0)
    assert body["title"] == "[monitor] quality alert"
    assert body["labels"] == ["monitor-alert", "rule:quality"]
    assert state["rules"]["quality"] == {"issue": 100, "clean_runs": 0}


def test_empty_state_is_accepted():
    fake = FakeGitHub()
    state = run_sync([latency_alert()], {}, fake)
    assert state["rules"]["latency"] == {"issue": 100, "clean_runs": 0}


def test_does_not_mutate_input_state():
    fake = FakeGitHub()
    original = {"rules": {"quality": {"issue": None, "clean_runs": 2}}}
    run_sync([quality_alert()], original, fake)
    assert original == {"rules": {"quality": {"issue": None, "clean_runs": 2}}}


def test_comments_when_open():
    fake = FakeGitHub(open_issues={7})
    state = run_sync([quality_alert()], {"rules": {"quality": {"issue": 7, "clean_runs": 2}}}, fake)

    assert fake.calls() == [
        ("GET", f"/repos/{REPO}/issues/7"),
        ("POST", f"/repos/{REPO}/issues/7/comments"),
    ]
    assert state["rules"]["quality"] == {"issue": 7, "clean_runs": 0}


def test_reopens_new_when_closed_by_hand():
    fake = FakeGitHub(open_issues=set())  # owner closed #7
    state = run_sync([quality_alert()], {"rules": {"quality": {"issue": 7, "clean_runs": 1}}}, fake)

    assert fake.calls() == [
        ("GET", f"/repos/{REPO}/issues/7"),
        ("POST", f"/repos/{REPO}/issues"),
    ]
    assert state["rules"]["quality"] == {"issue": 100, "clean_runs": 0}


def test_closes_after_3_clean_runs():
    fake = FakeGitHub(open_issues={7})
    state = {"rules": {"quality": {"issue": 7, "clean_runs": 0}}}

    state = run_sync([], state, fake)
    state = run_sync([], state, fake)
    assert fake.requests == []
    assert state["rules"]["quality"] == {"issue": 7, "clean_runs": 2}

    state = run_sync([], state, fake)
    assert fake.calls() == [
        ("GET", f"/repos/{REPO}/issues/7"),
        ("POST", f"/repos/{REPO}/issues/7/comments"),
        ("PATCH", f"/repos/{REPO}/issues/7"),
    ]
    assert "recovered" in fake.json_body(1)["body"]
    assert fake.json_body(2) == {"state": "closed"}
    assert state["rules"]["quality"] == {"issue": None, "clean_runs": 0}


def test_firing_resets_clean_run_count():
    fake = FakeGitHub(open_issues={7})
    state = {"rules": {"quality": {"issue": 7, "clean_runs": 2}}}
    state = run_sync([quality_alert()], state, fake)
    state = run_sync([], state, fake)
    assert state["rules"]["quality"] == {"issue": 7, "clean_runs": 1}
    assert not any(method == "PATCH" for method, _ in fake.calls())


def test_clean_run_without_issue_stays_zero():
    fake = FakeGitHub()
    state = run_sync([], {"rules": {"quality": {"issue": None, "clean_runs": 0}}}, fake)
    assert fake.requests == []
    assert state["rules"]["quality"] == {"issue": None, "clean_runs": 0}


def test_recovery_skips_comment_when_closed_by_hand():
    fake = FakeGitHub(open_issues=set())
    state = {"rules": {"quality": {"issue": 7, "clean_runs": 2}}}
    state = run_sync([], state, fake)
    assert fake.calls() == [("GET", f"/repos/{REPO}/issues/7")]
    assert state["rules"]["quality"] == {"issue": None, "clean_runs": 0}


def test_rules_are_independent():
    fake = FakeGitHub(open_issues={7})
    state = {
        "rules": {
            "quality": {"issue": 7, "clean_runs": 0},
            "latency": {"issue": None, "clean_runs": 0},
        }
    }
    state = run_sync([latency_alert()], state, fake)
    assert state["rules"]["latency"] == {"issue": 100, "clean_runs": 0}
    assert state["rules"]["quality"] == {"issue": 7, "clean_runs": 1}


def test_body_has_trace_links():
    body = render_body(quality_alert(), RUN_URL, "proj1")
    assert "https://cloud.langfuse.com/project/proj1/traces/tr-aaa" in body
    assert "https://cloud.langfuse.com/project/proj1/traces/tr-bbb" in body
    assert RUN_URL in body
    assert "q01" in body and "q06" in body
    assert "run-016" in body
    assert "9/24" in body
    assert "37.5%" in body
    assert trace_url("proj1", "tr-aaa") == "https://cloud.langfuse.com/project/proj1/traces/tr-aaa"


def test_body_without_project_id():
    body = render_body(quality_alert(), RUN_URL, None)
    assert "cloud.langfuse.com" not in body
    assert "tr-aaa" in body and "tr-bbb" in body


def test_uptime_and_latency_bodies():
    up = render_body(uptime_alert(), RUN_URL, "proj1")
    assert "timeout" in up and "http_500" in up and "q02" in up
    assert "traces/t1" in up
    lat = render_body(latency_alert(), RUN_URL, "proj1")
    assert "10.25" in lat and "3.00" in lat and "6.00" in lat


def test_body_never_contains_free_text_from_details():
    alert = quality_alert(
        answer="SECRET-TEXT",
        question="SECRET-QUESTION",
        citations=["SECRET-CITATION"],
    )
    alert.details["failed"][0]["answer"] = "SECRET-TEXT"
    alert.details["failed"][1]["question"] = "SECRET-QUESTION"
    alert.details["failed"][1]["question_id"] = "SECRET-TEXT with spaces"
    alert.details["failed"][1]["trace_id"] = "SECRET-TEXT [x](http://evil)"
    alert.summary = "SECRET-TEXT"
    body = render_body(alert, RUN_URL, "proj1")
    assert "SECRET" not in body
    assert "evil" not in body

    up = uptime_alert()
    up.details["errors"][0]["status"] = "SECRET-TEXT leaked"
    up.details["errors"][0]["body"] = "SECRET-TEXT"
    assert "SECRET" not in render_body(up, RUN_URL, "proj1")


def test_body_ignores_unsafe_run_url_and_project_id():
    body = render_body(quality_alert(), "javascript:alert(1) [x]", "p/../x")
    assert "javascript" not in body
    assert "cloud.langfuse.com" not in body


def test_dry_run_labels_and_state_isolated():
    fake = FakeGitHub(open_issues={7})
    state = {
        "rules": {"quality": {"issue": 7, "clean_runs": 2}},
    }
    new = run_sync([quality_alert()], state, fake, dry_run=True)

    assert fake.calls() == [("POST", f"/repos/{REPO}/issues")]
    body = fake.json_body(0)
    assert body["title"] == "[dry-run] [monitor] quality alert"
    assert body["labels"] == ["monitor-alert", "rule:quality", "dry-run"]
    assert new["rules"] == {"quality": {"issue": 7, "clean_runs": 2}}
    assert new["dry_run_issues"] == {"quality": 100}
    assert state == {"rules": {"quality": {"issue": 7, "clean_runs": 2}}}


def test_dry_run_comments_on_its_own_open_issue():
    fake = FakeGitHub(open_issues={50})
    state = {"rules": {}, "dry_run_issues": {"quality": 50}}
    new = run_sync([quality_alert()], state, fake, dry_run=True)
    assert fake.calls() == [
        ("GET", f"/repos/{REPO}/issues/50"),
        ("POST", f"/repos/{REPO}/issues/50/comments"),
    ]
    assert new["dry_run_issues"] == {"quality": 50}


def test_dry_run_with_no_alerts_changes_nothing():
    fake = FakeGitHub(open_issues={7})
    state = {"rules": {"quality": {"issue": 7, "clean_runs": 2}}}
    new = run_sync([], state, fake, dry_run=True)
    assert fake.requests == []
    assert new["rules"] == state["rules"]


def test_real_run_never_touches_dry_run_issues():
    fake = FakeGitHub()
    state = {"rules": {}, "dry_run_issues": {"quality": 50}}
    new = run_sync([quality_alert()], state, fake)
    assert new["dry_run_issues"] == {"quality": 50}
    assert new["rules"]["quality"]["issue"] == 100


def test_token_only_in_authorization_header():
    fake = FakeGitHub()
    run_sync([quality_alert()], {"rules": {}}, fake)
    request = fake.requests[0]
    assert request.headers["Authorization"] == f"Bearer {TOKEN}"
    assert TOKEN not in str(request.url)
    assert TOKEN not in request.content.decode()
    assert TOKEN not in repr(fake.gh())


def test_http_error_does_not_leak_token():
    def handler(request):
        return httpx.Response(403, json={"message": f"echo {TOKEN}"})

    gh = GitHubIssues(REPO, TOKEN, transport=httpx.MockTransport(handler))
    with pytest.raises(GitHubError) as info:
        gh.create("t", "b", ["monitor-alert"])
    assert TOKEN not in str(info.value)
    assert TOKEN not in repr(info.value)
    assert "403" in str(info.value)


def test_transport_error_does_not_leak_token():
    def handler(request):
        raise httpx.ConnectError(f"boom {request.headers['Authorization']}")

    gh = GitHubIssues(REPO, TOKEN, transport=httpx.MockTransport(handler))
    with pytest.raises(GitHubError) as info:
        gh.comment(1, "b")
    assert TOKEN not in str(info.value)
    assert TOKEN not in repr(info.value)
    assert info.value.__cause__ is None


def test_is_open_handles_missing_issue_and_errors():
    def handler(request):
        return httpx.Response(404 if request.url.path.endswith("/9") else 500)

    gh = GitHubIssues(REPO, TOKEN, transport=httpx.MockTransport(handler))
    assert gh.is_open(9) is False
    with pytest.raises(GitHubError):
        gh.is_open(10)


@pytest.mark.parametrize(
    "repo",
    ["", "owner", "owner/", "/name", "a/b/c", "a b/c", "a/b c", "../x", "a/..", "a/b?x=1", "a/b\n"],
)
def test_rejects_bad_repo(repo):
    with pytest.raises(ValueError, match="owner/name"):
        GitHubIssues(repo, TOKEN, transport=httpx.MockTransport(lambda r: httpx.Response(200)))


def test_accepts_dotted_repo():
    GitHubIssues("my-org/rail_rag.monitor", TOKEN, transport=httpx.MockTransport(FakeGitHub()))


@pytest.mark.parametrize("token", ["", "a\nb", "a b"])
def test_rejects_bad_token_without_echoing_it(token):
    with pytest.raises(ValueError) as info:
        GitHubIssues(REPO, token, transport=httpx.MockTransport(FakeGitHub()))
    assert token not in str(info.value) or token == ""


class FailingFake(FakeGitHub):
    """Like FakeGitHub, but creating an issue whose title mentions a rule in `fail_rules` is 403."""

    def __init__(self, fail_rules, **kwargs):
        super().__init__(**kwargs)
        self.fail_rules = fail_rules

    def __call__(self, request):
        if request.method == "POST" and request.url.path == f"/repos/{REPO}/issues":
            title = json.loads(request.content)["title"]
            if any(f"[monitor] {rule} alert" in title for rule in self.fail_rules):
                self.requests.append(request)
                return httpx.Response(403, json={"message": f"echo {TOKEN}"})
        return super().__call__(request)


def test_mid_sync_failure_keeps_earlier_issue_numbers():
    fake = FailingFake({"quality"})
    with pytest.raises(SyncError) as info:
        run_sync([uptime_alert(), quality_alert()], {"rules": {}}, fake)

    state = info.value.state
    assert state["rules"]["uptime"] == {"issue": 100, "clean_runs": 0}
    assert state["rules"]["quality"] == {"issue": None, "clean_runs": 0}
    assert isinstance(info.value.__cause__, GitHubError)

    # A rerun with the saved state comments on the uptime issue instead of duplicating it.
    fake2 = FakeGitHub(open_issues={100}, next_number=200)
    new = run_sync([uptime_alert(), quality_alert()], state, fake2)
    assert ("POST", f"/repos/{REPO}/issues/100/comments") in fake2.calls()
    assert fake2.calls().count(("POST", f"/repos/{REPO}/issues")) == 1
    assert new["rules"]["uptime"]["issue"] == 100
    assert new["rules"]["quality"]["issue"] == 200


def test_failing_first_rule_does_not_stop_next_rule():
    fake = FailingFake({"uptime"})
    with pytest.raises(SyncError) as info:
        run_sync([uptime_alert(), latency_alert()], {"rules": {}}, fake)
    state = info.value.state
    assert state["rules"]["uptime"]["issue"] is None
    assert state["rules"]["latency"] == {"issue": 100, "clean_runs": 0}
    assert "uptime" in str(info.value) and "latency" not in str(info.value)


def test_dry_run_mid_sync_failure_keeps_partial_dry_run_issues():
    fake = FailingFake({"quality"})
    state = {"rules": {"quality": {"issue": 7, "clean_runs": 2}}}
    with pytest.raises(SyncError) as info:
        run_sync([uptime_alert(), quality_alert(), latency_alert()], state, fake, dry_run=True)
    new = info.value.state
    assert new["dry_run_issues"] == {"uptime": 100, "latency": 101}
    assert new["rules"] == state["rules"]


def test_sync_error_never_contains_token():
    fake = FailingFake({"quality", "uptime"})
    with pytest.raises(SyncError) as info:
        run_sync([uptime_alert(), quality_alert()], {"rules": {}}, fake)
    for text in (str(info.value), repr(info.value), str(info.value.__cause__)):
        assert TOKEN not in text
    assert "uptime" in str(info.value) and "quality" in str(info.value)
    assert TOKEN not in repr(info.value.__cause__)


# --- hardening (H-B) --------------------------------------------------------------------------


def test_requests_carry_the_github_headers():
    fake = FakeGitHub()
    run_sync([quality_alert()], {"rules": {}}, fake)
    headers = fake.requests[0].headers
    assert headers["Accept"] == "application/vnd.github+json"
    assert headers["X-GitHub-Api-Version"] == "2022-11-28"
    assert headers["User-Agent"] == "rail-rag-monitor"


def test_trace_links_appear_in_the_firing_comment():
    fake = FakeGitHub(open_issues={7})
    run_sync([quality_alert()], {"rules": {"quality": {"issue": 7, "clean_runs": 0}}}, fake)
    assert fake.calls()[1] == ("POST", f"/repos/{REPO}/issues/7/comments")
    comment = fake.json_body(1)["body"]
    assert comment.startswith("Alert fired again.")
    assert "https://cloud.langfuse.com/project/proj1/traces/tr-aaa" in comment
    assert "https://cloud.langfuse.com/project/proj1/traces/tr-bbb" in comment
    assert RUN_URL in comment


def test_is_open_treats_410_as_closed():
    gh = GitHubIssues(REPO, TOKEN, transport=httpx.MockTransport(lambda r: httpx.Response(410)))
    assert gh.is_open(9) is False


def test_github_error_carries_the_status_code():
    gh = GitHubIssues(REPO, TOKEN, transport=httpx.MockTransport(lambda r: httpx.Response(403)))
    with pytest.raises(GitHubError) as info:
        gh.comment(1, "b")
    assert info.value.status_code == 403
    assert GitHubError("no response").status_code is None

    def boom(request):
        raise httpx.ConnectError("x")

    gh = GitHubIssues(REPO, TOKEN, transport=httpx.MockTransport(boom))
    with pytest.raises(GitHubError) as info:
        gh.comment(1, "b")
    assert info.value.status_code is None


def test_is_open_does_not_match_status_text_in_the_message():
    # A 500 whose message happens to contain "HTTP 404" is still an error, not "closed".
    def handler(request):
        return httpx.Response(500)

    gh = GitHubIssues(REPO, TOKEN, transport=httpx.MockTransport(handler))
    with pytest.raises(GitHubError) as info:
        gh.is_open(404)
    assert info.value.status_code == 500


@pytest.mark.parametrize("status", [301, 302, 307, 308])
def test_redirects_are_errors(status):
    # A renamed repository answers with a redirect; that must not look like success.
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            status,
            headers={"Location": "https://api.github.com/repositories/1/issues"},
            json={"number": 5},
        )

    gh = GitHubIssues(REPO, TOKEN, transport=httpx.MockTransport(handler))
    with pytest.raises(GitHubError) as info:
        gh.create("t", "b", ["monitor-alert"])
    assert info.value.status_code == status
    assert str(status) in str(info.value)
    with pytest.raises(GitHubError):
        gh.close(5)
    with pytest.raises(GitHubError):
        gh.is_open(5)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json={}),
        httpx.Response(200, json={"number": "abc"}),
        httpx.Response(200, json={"number": None}),
        httpx.Response(200, json={"number": True}),
        httpx.Response(200, json=[1, 2]),
        httpx.Response(200, content=b"<html>SECRET-BODY</html>"),
        httpx.Response(200, content=b""),
    ],
)
def test_create_wraps_a_malformed_2xx_body(response):
    gh = GitHubIssues(REPO, TOKEN, transport=httpx.MockTransport(lambda r: response))
    with pytest.raises(GitHubError) as info:
        gh.create("t", "b", ["monitor-alert"])
    assert "SECRET-BODY" not in str(info.value)
    assert TOKEN not in str(info.value)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"not json"),
        httpx.Response(200, content=b""),
        httpx.Response(200, json=[{"state": "open"}]),
        httpx.Response(200, json="open"),
    ],
)
def test_is_open_wraps_a_malformed_2xx_body(response):
    gh = GitHubIssues(REPO, TOKEN, transport=httpx.MockTransport(lambda r: response))
    with pytest.raises(GitHubError):
        gh.is_open(7)


def test_unknown_rule_name_is_rejected_before_any_request():
    fake = FakeGitHub()
    bad = Alert("bogus rule\nSECRET", "s", {})
    with pytest.raises(ValueError, match="unknown alert rule") as info:
        run_sync([quality_alert(), bad], {"rules": {}}, fake)
    assert fake.requests == []
    assert "SECRET" not in str(info.value)
    with pytest.raises(ValueError, match="unknown alert rule"):
        render_body(bad, RUN_URL, "proj1")


def test_dry_run_never_patches_while_real_issues_are_open():
    fake = FakeGitHub(open_issues={7, 8, 9, 50})
    state = {
        "rules": {
            "uptime": {"issue": 7, "clean_runs": 2},
            "quality": {"issue": 8, "clean_runs": 2},
            "latency": {"issue": 9, "clean_runs": 2},
        },
        "dry_run_issues": {"quality": 50},
    }
    new = run_sync([quality_alert()], state, fake, dry_run=True)
    assert not any(method == "PATCH" for method, _ in fake.calls())
    assert fake.calls() == [
        ("GET", f"/repos/{REPO}/issues/50"),
        ("POST", f"/repos/{REPO}/issues/50/comments"),
    ]
    assert new["rules"] == state["rules"]


class MalformedCreateFake(FakeGitHub):
    """Like FakeGitHub, but creating a `fail_rules` issue returns 200 with a body lacking the id."""

    def __init__(self, fail_rules, **kwargs):
        super().__init__(**kwargs)
        self.fail_rules = fail_rules

    def __call__(self, request):
        if request.method == "POST" and request.url.path == f"/repos/{REPO}/issues":
            title = json.loads(request.content)["title"]
            if any(f"[monitor] {rule} alert" in title for rule in self.fail_rules):
                self.requests.append(request)
                return httpx.Response(200, json={})
        return super().__call__(request)


def test_malformed_create_body_still_returns_sync_error_with_partial_state():
    fake = MalformedCreateFake({"quality"})
    with pytest.raises(SyncError) as info:
        run_sync(
            [uptime_alert(), quality_alert(), latency_alert()],
            {"rules": {}},
            fake,
        )
    state = info.value.state
    assert state["rules"]["uptime"] == {"issue": 100, "clean_runs": 0}
    assert state["rules"]["quality"] == {"issue": None, "clean_runs": 0}
    assert state["rules"]["latency"] == {"issue": 101, "clean_runs": 0}
    assert isinstance(info.value.__cause__, GitHubError)
    assert "quality" in str(info.value)


def test_malformed_is_open_body_does_not_lose_partial_state():
    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, content=b"not json")
        return httpx.Response(201, json={"number": 100})

    gh = GitHubIssues(REPO, TOKEN, transport=httpx.MockTransport(handler))
    state = {"rules": {"quality": {"issue": 7, "clean_runs": 1}}}
    with pytest.raises(SyncError) as info:
        sync(
            [uptime_alert(), quality_alert()],
            state,
            gh,
            run_url=RUN_URL,
            project_id="proj1",
            dry_run=False,
        )
    assert info.value.state["rules"]["uptime"]["issue"] == 100
    assert info.value.state["rules"]["quality"]["issue"] == 7


def test_failed_fire_resets_the_clean_run_count_first():
    # Issue 7 was reported open with 2 clean runs banked. The firing run's GET fails: the count
    # must already be 0, so one later clean run cannot close the issue.
    failing = {"on": True}

    def handler(request):
        if request.method == "GET" and failing["on"]:
            return httpx.Response(500)
        return FakeGitHub(open_issues={7})(request)

    gh = GitHubIssues(REPO, TOKEN, transport=httpx.MockTransport(handler))
    state = {"rules": {"quality": {"issue": 7, "clean_runs": 2}}}
    with pytest.raises(SyncError) as info:
        sync([quality_alert()], state, gh, run_url=RUN_URL, project_id="p", dry_run=False)
    saved = info.value.state
    assert saved["rules"]["quality"] == {"issue": 7, "clean_runs": 0}
    assert state == {"rules": {"quality": {"issue": 7, "clean_runs": 2}}}

    failing["on"] = False
    after = sync([], saved, gh, run_url=RUN_URL, project_id="p", dry_run=False)
    assert after["rules"]["quality"] == {"issue": 7, "clean_runs": 1}
