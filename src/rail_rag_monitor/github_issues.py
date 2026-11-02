"""GitHub issue notifier (spec section 7, "Issue handling" and "Dry run").

One open issue per rule. A new firing comments on it; if the owner closed it by hand the next
firing opens a new one; after `CLEAN_RUNS_TO_CLOSE` consecutive clean runs the monitor comments
posts a closing comment and closes it.

State shape: `{"rules": {rule: {"issue": int | None, "clean_runs": int}}, "dry_run_issues":
{rule: int}}`. Real runs only touch `rules`; dry runs only touch `dry_run_issues`, so a planted
regression can never open, comment on or close a real alert issue.

Privacy: issue text is built only from numbers, rule names, run ids, question ids (q01, ...),
statuses and trace ids, each read explicitly from `Alert.details`. Anything else in `details`
(and `Alert.summary`) is ignored, and every id-like value must match a strict pattern or it is
replaced by a placeholder. The token only ever travels in the `Authorization` header and never
appears in an exception message or repr.
"""

from __future__ import annotations

import copy
import re

import httpx

from rail_rag_monitor.alerts import CLEAN_RUNS_TO_CLOSE, Alert

API_URL = "https://api.github.com"
LANGFUSE_URL = "https://cloud.langfuse.com"
RULES = ("uptime", "quality", "latency")
LABEL = "monitor-alert"
DRY_RUN_LABEL = "dry-run"
DRY_RUN_PREFIX = "[dry-run] "
MAX_LIST_ROWS = 50

_REPO_PART = r"[A-Za-z0-9_.-]+"
_REPO_RE = re.compile(rf"{_REPO_PART}/{_REPO_PART}")
_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,100}")
_RUN_URL_RE = re.compile(r"https://[^\s()<>\[\]]{1,500}")
_PLACEHOLDER = "(invalid)"
_MISSING = "n/a"


class GitHubError(RuntimeError):
    """A GitHub API call failed. The message never contains the token or a response body.

    `status_code` is the HTTP status of the failing response, or None when there was none
    (transport error, malformed body).
    """

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class SyncError(RuntimeError):
    """One or more rules failed to sync. `.state` is the up-to-date state, including every issue
    number created or closed before and after the failure; save it before exiting non-zero.
    `__cause__` is the first underlying `GitHubError`. The message has only rule names and
    exception class names, never the token or a response body."""

    def __init__(self, message: str, state: dict):
        super().__init__(message)
        self.state = state


def langfuse_base_url(host: str | None = None) -> str:
    """The Langfuse base URL: `host` (https only, no trailing slash) or the cloud default."""
    base = (host or "").strip().rstrip("/") or LANGFUSE_URL
    if not base.startswith("https://"):
        raise ValueError("the Langfuse host must start with https://")
    return base


def trace_url(project_id: str, trace_id: str, host: str | None = None) -> str:
    return f"{langfuse_base_url(host)}/project/{project_id}/traces/{trace_id}"


class GitHubIssues:
    def __init__(self, repo: str, token: str, *, transport: httpx.BaseTransport | None = None):
        if not _REPO_RE.fullmatch(repo) or any(part in (".", "..") for part in repo.split("/")):
            raise ValueError("repo must look like owner/name (letters, digits, '-', '_', '.')")
        if not token or not token.isascii() or not token.isprintable() or " " in token:
            raise ValueError("token must be a non-empty string without spaces or control chars")
        self.repo = repo
        self._client = httpx.Client(
            base_url=API_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "rail-rag-monitor",
            },
            transport=transport,
            timeout=30.0,
        )

    def __repr__(self) -> str:
        return f"GitHubIssues(repo={self.repo!r})"

    def _request(self, method: str, path: str, *, json: dict | None = None) -> httpx.Response:
        url = f"/repos/{self.repo}{path}"
        try:
            response = self._client.request(method, url, json=json)
        except httpx.HTTPError as exc:
            raise GitHubError(f"GitHub {method} {url} failed: {type(exc).__name__}") from None
        if response.status_code >= 300:  # redirects (a renamed repo) are not success either
            raise GitHubError(
                f"GitHub {method} {url} failed: HTTP {response.status_code}",
                response.status_code,
            )
        return response

    @staticmethod
    def _json_object(response: httpx.Response, what: str) -> dict:
        """The JSON object in a 2xx response. The body is never put in an error."""
        try:
            body = response.json()
        except ValueError:
            body = None
        if not isinstance(body, dict):
            raise GitHubError(f"GitHub {what} returned an unexpected response")
        return body

    def create(self, title: str, body: str, labels: list[str]) -> int:
        payload = {"title": title, "body": body, "labels": labels}
        response = self._request("POST", "/issues", json=payload)
        number = self._json_object(response, "issue creation").get("number")
        if isinstance(number, bool) or not isinstance(number, int):
            raise GitHubError("GitHub issue creation returned an unexpected response")
        return number

    def comment(self, number: int, body: str) -> None:
        self._request("POST", f"/issues/{int(number)}/comments", json={"body": body})

    def close(self, number: int) -> None:
        self._request("PATCH", f"/issues/{int(number)}", json={"state": "closed"})

    def is_open(self, number: int) -> bool:
        try:
            response = self._request("GET", f"/issues/{int(number)}")
        except GitHubError as exc:
            if exc.status_code in (404, 410):
                return False
            raise
        return self._json_object(response, "issue lookup").get("state") == "open"


# --- body rendering -------------------------------------------------------------------------


def _ident(value: object) -> str | None:
    """A safe id-like string, the placeholder for anything else, None when absent."""
    if value is None:
        return None
    text = str(value) if isinstance(value, str | int) and not isinstance(value, bool) else ""
    return text if _ID_RE.fullmatch(text) else _PLACEHOLDER


def _num(value: object, fmt: str = "g") -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return _MISSING
    return format(value, fmt)


def _ratio(num: object, den: object) -> str:
    if (
        isinstance(num, int)
        and isinstance(den, int)
        and not isinstance(num, bool)
        and not isinstance(den, bool)
    ):
        return f"{num}/{den}"
    return _MISSING


def _id_list(values: object) -> str:
    if not isinstance(values, list):
        return _MISSING
    shown = [_ident(v) or _MISSING for v in values[:MAX_LIST_ROWS]]
    more = f" (+{len(values) - MAX_LIST_ROWS} more)" if len(values) > MAX_LIST_ROWS else ""
    return (", ".join(shown) or _MISSING) + more


def _trace(trace_id: object, project_id: str | None, host: str | None = None) -> str:
    tid = _ident(trace_id)
    if tid is None:
        return _MISSING
    pid = _ident(project_id)
    if tid == _PLACEHOLDER or pid is None or pid == _PLACEHOLDER:
        return f"`{tid}`" if tid != _PLACEHOLDER else tid
    return f"[{tid}]({trace_url(pid, tid, host)})"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def _rows(items: object) -> tuple[list[dict], int]:
    if not isinstance(items, list):
        return [], 0
    dicts = [i for i in items if isinstance(i, dict)]
    return dicts[:MAX_LIST_ROWS], max(0, len(dicts) - MAX_LIST_ROWS)


def _trace_table(
    items: object, project_id: str | None, with_status: bool, host: str | None = None
) -> list[str]:
    rows, extra = _rows(items)
    if not rows:
        return []
    headers = ["Run", "Question"] + (["Status"] if with_status else []) + ["Trace"]
    table = []
    for item in rows:
        row = [_ident(item.get("run_id")) or _MISSING, _ident(item.get("question_id")) or _MISSING]
        if with_status:
            row.append(_ident(item.get("status")) or _MISSING)
        row.append(_trace(item.get("trace_id"), project_id, host))
        table.append(row)
    lines = _table(headers, table)
    if extra:
        lines.append(f"\n(+{extra} more not shown)")
    return lines


def _run_link(run_url: str) -> str | None:
    return f"[workflow run]({run_url})" if _RUN_URL_RE.fullmatch(run_url or "") else None


def render_body(
    alert: Alert, run_url: str, project_id: str | None, langfuse_host: str | None = None
) -> str:
    """Markdown for an issue or comment. Only whitelisted `details` fields are rendered."""
    d = alert.details if isinstance(alert.details, dict) else {}
    lines: list[str] = []
    link = _run_link(run_url)
    if link:
        lines += [f"Run: {link}", ""]

    if alert.rule == "uptime":
        lines += ["**Uptime**: probe questions ended in an error in consecutive runs.", ""]
        lines += _table(["Metric", "Value"], [["Runs", _id_list(d.get("run_ids"))]])
        errors = _trace_table(d.get("errors"), project_id, with_status=True, host=langfuse_host)
        if errors:
            lines += ["", "Failed questions:", ""] + errors
    elif alert.rule == "quality":
        recent = _ratio(d.get("fail_recent"), d.get("n_recent"))
        base = _ratio(d.get("fail_base"), d.get("n_base"))
        lines += ["**Quality**: the probe fail rate rose above the baseline.", ""]
        lines += _table(
            ["Metric", "Value"],
            [
                [
                    "Fail rate, recent window",
                    f"{_num(d.get('fail_rate_recent'), '.1%')} ({recent})",
                ],
                ["Fail rate, baseline", f"{_num(d.get('fail_rate_base'), '.1%')} ({base})"],
                ["One-sided Fisher p", _num(d.get("p_value"), ".3g")],
                ["Alpha", _num(d.get("alpha"), "g")],
                ["Minimum rise (delta)", _num(d.get("delta"), ".1%")],
                ["Recent runs", _id_list(d.get("recent_run_ids"))],
                ["Baseline runs", _id_list(d.get("baseline_run_ids"))],
            ],
        )
        failed = _trace_table(d.get("failed"), project_id, with_status=False, host=langfuse_host)
        if failed:
            lines += ["", "Failed questions:", ""] + failed
    elif alert.rule == "latency":
        p95 = d.get("recent_warm_p95_s")
        p95_text = ", ".join(_num(v, ".2f") for v in p95) if isinstance(p95, list) else _MISSING
        lines += ["**Latency**: warm p95 latency stayed above the threshold.", ""]
        lines += _table(
            ["Metric", "Value"],
            [
                ["Recent warm p95 (s)", p95_text],
                ["Baseline median (s)", _num(d.get("baseline_median_s"), ".2f")],
                ["Threshold (s)", _num(d.get("threshold_s"), ".2f")],
                ["Factor", _num(d.get("factor"), "g")],
                ["Baseline runs with latency", _num(d.get("n_base"), "d")],
                ["Recent runs", _id_list(d.get("recent_run_ids"))],
                ["Baseline runs", _id_list(d.get("baseline_run_ids"))],
            ],
        )
    else:
        raise ValueError(f"unknown alert rule: {_ident(alert.rule)}")

    if project_id is not None:
        lines += ["", "Langfuse trace links need the owner's login."]
    return "\n".join(lines) + "\n"


# --- sync -----------------------------------------------------------------------------------


def _title(rule: str, dry_run: bool) -> str:
    return f"{DRY_RUN_PREFIX if dry_run else ''}[monitor] {rule} alert"


def _labels(rule: str, dry_run: bool) -> list[str]:
    return [LABEL, f"rule:{rule}"] + ([DRY_RUN_LABEL] if dry_run else [])


def _fire(
    alert: Alert,
    issue: int | None,
    gh: GitHubIssues,
    *,
    run_url: str,
    project_id: str | None,
    dry_run: bool,
    langfuse_host: str | None = None,
) -> int:
    """Comment on the open issue, or open a new one. Returns the issue number now in use."""
    body = render_body(alert, run_url, project_id, langfuse_host)
    if issue is not None and gh.is_open(issue):
        gh.comment(issue, "Alert fired again.\n\n" + body)
        return issue
    return gh.create(_title(alert.rule, dry_run), body, _labels(alert.rule, dry_run))


def _sync_rule(
    rule: str,
    alert: Alert | None,
    new: dict,
    gh: GitHubIssues,
    *,
    run_url: str,
    project_id: str | None,
    dry_run: bool,
    langfuse_host: str | None = None,
) -> None:
    """Apply one rule to `new` in place.

    A firing run resets `clean_runs` before it calls GitHub, so a failed call cannot leave banked
    clean runs behind (one later clean run would otherwise close an issue that is still firing).
    Any other API failure leaves that rule's entry as it was.
    """
    if dry_run:
        # Dry runs open or comment on their own issues and never close anything.
        if alert is not None:
            dry = new.setdefault("dry_run_issues", {})
            dry[rule] = _fire(
                alert,
                dry.get(rule),
                gh,
                run_url=run_url,
                project_id=project_id,
                dry_run=True,
                langfuse_host=langfuse_host,
            )
        return

    entry = new["rules"].setdefault(rule, {"issue": None, "clean_runs": 0})
    if alert is not None:
        entry["clean_runs"] = 0
        entry["issue"] = _fire(
            alert,
            entry.get("issue"),
            gh,
            run_url=run_url,
            project_id=project_id,
            dry_run=False,
            langfuse_host=langfuse_host,
        )
        return
    issue = entry.get("issue")
    if issue is None:
        entry["clean_runs"] = 0
        return
    clean_runs = entry.get("clean_runs", 0) + 1
    if clean_runs >= CLEAN_RUNS_TO_CLOSE:
        if gh.is_open(issue):
            gh.comment(
                issue,
                f"This rule has not fired for {CLEAN_RUNS_TO_CLOSE} evaluations in a row. That "
                "means either the problem is gone or the rolling baseline has absorbed the "
                "change, so check the dashboard before assuming it is fixed. Closing this issue.",
            )
            gh.close(issue)
        entry["issue"] = None
        clean_runs = 0
    entry["clean_runs"] = clean_runs


def sync(
    alerts: list[Alert],
    state: dict,
    gh: GitHubIssues,
    *,
    run_url: str,
    project_id: str | None,
    dry_run: bool,
    langfuse_host: str | None = None,
) -> dict:
    """Apply this run's alerts to GitHub and return the new state (the input is not modified).

    Every rule is processed even if an earlier one fails. If any rule's API call fails, a
    `SyncError` is raised after the loop; its `.state` holds the up-to-date state, including the
    issue numbers created before and after the failure. The caller must save `exc.state` before
    exiting non-zero, otherwise a retry would open duplicate issues.
    """
    new = copy.deepcopy(state)
    new.setdefault("rules", {})
    by_rule: dict[str, Alert] = {}
    for alert in alerts:
        if alert.rule not in RULES:
            raise ValueError(f"unknown alert rule: {_ident(alert.rule)}")
        by_rule.setdefault(alert.rule, alert)

    failures: list[tuple[str, GitHubError]] = []
    for rule in RULES:
        try:
            _sync_rule(
                rule,
                by_rule.get(rule),
                new,
                gh,
                run_url=run_url,
                project_id=project_id,
                dry_run=dry_run,
                langfuse_host=langfuse_host,
            )
        except GitHubError as exc:
            failures.append((rule, exc))
    if failures:
        names = ", ".join(f"{rule} ({type(exc).__name__})" for rule, exc in failures)
        raise SyncError(f"GitHub sync failed for: {names}", new) from failures[0][1]
    return new
