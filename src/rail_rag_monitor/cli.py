"""Command line: probe, collect, alert, noise-report and build-site.

Secrets (Langfuse keys, GitHub token) are read from the environment and handed to the client
modules; this module never prints them. All paths are under `--data DIR`.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx

from rail_rag_monitor.alerts import DELTA_FLOOR, evaluate, noise_report, planted_runs
from rail_rag_monitor.collector import collect_day
from rail_rag_monitor.dashboard import build_site
from rail_rag_monitor.github_issues import GitHubIssues, SyncError, sync
from rail_rag_monitor.langfuse_api import LangfuseAPI, LangfuseError
from rail_rag_monitor.probe import PACE_S, PROBE_GOLDEN, run_probe
from rail_rag_monitor.store import (
    ALERT_STATE,
    NOISE,
    PROBE_RUNS,
    TRAFFIC_DAILY,
    read_json,
    read_jsonl,
    upsert_jsonl,
    write_json,
)

MAX_REPEAT = 5
PLACEHOLDER_REPO = "example-owner/rail-rag-monitor"  # shown only when no repo is known
RUN_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,100}")


class CliError(Exception):
    """A user-facing failure: the message is printed to stderr and the exit code is 1."""


@dataclass
class _Deps:
    """Everything a command needs from the outside world, so tests can replace it."""

    env: Mapping[str, str]
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], datetime] = lambda: datetime.now(UTC)
    transports: Mapping[str, httpx.BaseTransport] = field(default_factory=dict)

    def langfuse(self) -> LangfuseAPI | None:
        if LangfuseAPI.from_env(self.env) is None:  # keys missing
            return None
        return LangfuseAPI(
            self.env["LANGFUSE_PUBLIC_KEY"],
            self.env["LANGFUSE_SECRET_KEY"],
            self.env.get("LANGFUSE_HOST") or "https://cloud.langfuse.com",
            transport=self.transports.get("langfuse"),
            sleep=self.sleep,
        )

    def github(self) -> GitHubIssues | None:
        repo, token = self.env.get("GITHUB_REPOSITORY"), self.env.get("GITHUB_TOKEN")
        if not repo or not token:
            return None
        try:
            return GitHubIssues(repo, token, transport=self.transports.get("github"))
        except ValueError as exc:
            raise CliError(f"cannot use GITHUB_REPOSITORY / GITHUB_TOKEN: {exc}") from None


def _sorted_runs(data: Path) -> list[dict]:
    """Stored probe runs, oldest first (the rules and the noise report expect that order)."""
    return sorted(read_jsonl(data / PROBE_RUNS), key=lambda run: run.get("started_at") or "")


# --- probe ----------------------------------------------------------------------------------


def _run_ids(deps: _Deps, count: int) -> list[str]:
    run_id, attempt = deps.env.get("GITHUB_RUN_ID"), deps.env.get("GITHUB_RUN_ATTEMPT") or "1"
    base = (
        f"{run_id}-{attempt}" if run_id else deps.now().astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    ids = [f"{base}-{i}" for i in range(1, count + 1)]
    if not all(RUN_ID_RE.fullmatch(i) for i in ids):
        raise CliError("GITHUB_RUN_ID / GITHUB_RUN_ATTEMPT do not form a valid run id")
    return ids


def _cmd_probe(args: argparse.Namespace, deps: _Deps) -> int:
    base_url = deps.env.get("DEMO_URL")
    if not base_url:
        raise CliError("DEMO_URL is not set: it is the demo base URL the probe asks")
    data = Path(args.data)
    commit = deps.env.get("GITHUB_SHA") or "local"
    for i, run_id in enumerate(_run_ids(deps, args.repeat)):
        if i:
            deps.sleep(PACE_S)  # the demo allows 5 /ask calls a minute
        with httpx.Client(transport=deps.transports.get("demo"), timeout=120.0) as client:
            run = run_probe(
                base_url,
                PROBE_GOLDEN,
                run_id=run_id,
                kind=args.kind,
                commit=commit,
                client=client,
                sleep=deps.sleep,
                now=deps.now,
            )
        upsert_jsonl(data / PROBE_RUNS, [run.to_dict()], "run_id")
        print(f"probe {run.run_id}: {run.n_pass}/{run.n_scored} passed, {run.n_errors} errors")
    return 0


# --- collect --------------------------------------------------------------------------------


def _cmd_collect(args: argparse.Namespace, deps: _Deps) -> int:
    api = deps.langfuse()
    if api is None:
        print("Langfuse keys not set: skipping collection")
        return 0
    data = Path(args.data)
    day = args.day or (deps.now().astimezone(UTC).date() - timedelta(days=1))
    probe_traces = {
        res["trace_id"]
        for run in read_jsonl(data / PROBE_RUNS)
        for res in (run.get("results") or {}).values()
        if res.get("trace_id")
    }
    traffic = collect_day(api, day, probe_traces)
    upsert_jsonl(data / TRAFFIC_DAILY, [traffic.to_dict()], "day")
    print(f"collected {traffic.day}: {traffic.n_requests} requests, {traffic.n_errors} errors")
    return 0


# --- alert ----------------------------------------------------------------------------------


def _delta(data: Path) -> float:
    delta = (read_json(data / NOISE, None) or {}).get("delta")
    if isinstance(delta, bool) or not isinstance(delta, int | float):
        return DELTA_FLOOR
    return float(delta)


def _run_url(env: Mapping[str, str]) -> str:
    server, repo, run_id = (
        env.get(k) for k in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID")
    )
    return (
        f"{server.rstrip('/')}/{repo}/actions/runs/{run_id}" if server and repo and run_id else ""
    )


def _project_id(deps: _Deps) -> str | None:
    try:
        api = deps.langfuse()
        return api.project_id() if api else None
    except Exception:  # links are optional: alerting must go on
        return None


def _cmd_alert(args: argparse.Namespace, deps: _Deps) -> int:
    data = Path(args.data)
    runs = _sorted_runs(data)
    if args.plant_regression:
        if not runs:
            raise CliError("cannot plant a regression: no stored probe runs")
        runs = runs + planted_runs(runs[-1])  # in memory only, never stored
    alerts = evaluate(runs, _delta(data))
    prefix = "dry run: " if args.plant_regression else ""
    rules = ", ".join(a.rule for a in alerts)
    print(f"{prefix}{len(alerts)} alert(s): {rules or 'none'}")

    gh = deps.github()
    if gh is None:
        print("GITHUB_REPOSITORY or GITHUB_TOKEN not set: skipping issue sync")
        return 0
    state = read_json(data / ALERT_STATE, {})
    try:
        new_state = sync(
            alerts,
            state,
            gh,
            run_url=_run_url(deps.env),
            project_id=_project_id(deps) if alerts else None,
            dry_run=args.plant_regression,
        )
    except SyncError as exc:
        write_json(data / ALERT_STATE, exc.state)  # keep the issue numbers: no duplicates on retry
        raise CliError(str(exc)) from None
    write_json(data / ALERT_STATE, new_state)
    return 0


# --- noise-report and build-site ------------------------------------------------------------


def _cmd_noise_report(args: argparse.Namespace, deps: _Deps) -> int:
    data = Path(args.data)
    report = noise_report(_sorted_runs(data), now=deps.now)
    write_json(data / NOISE, report)
    print(f"noise report: {report['n_runs']} runs, delta {report['delta']:.3f}")
    return 0


def _cmd_build_site(args: argparse.Namespace, deps: _Deps) -> int:
    data = Path(args.data)
    repo = args.repo or deps.env.get("GITHUB_REPOSITORY")
    if not repo:
        print(f"no repository given: using the placeholder {PLACEHOLDER_REPO}")
        repo = PLACEHOLDER_REPO
    try:
        path = build_site(
            _sorted_runs(data),
            read_jsonl(data / TRAFFIC_DAILY),
            read_json(data / ALERT_STATE, {}),
            read_json(data / NOISE, None),
            Path(args.out),
            repo=repo,
        )
    except ValueError as exc:
        raise CliError(str(exc)) from None
    print(f"wrote {path}")
    return 0


# --- parser and entry point -----------------------------------------------------------------


def _day(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError("expected a day as YYYY-MM-DD") from None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rail-rag-monitor")
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name: str, handler: Callable, help_text: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--data", default="data", help="data directory (default: data)")
        sub.set_defaults(handler=handler)
        return sub

    probe = command("probe", _cmd_probe, "ask the demo the golden questions")
    probe.add_argument("--kind", required=True, choices=["scheduled", "manual", "noise"])
    probe.add_argument("--repeat", type=int, default=1, help=f"noise runs only, 1 to {MAX_REPEAT}")

    collect = command("collect", _cmd_collect, "summarise one UTC day of production traffic")
    collect.add_argument("--day", type=_day, help="YYYY-MM-DD (default: yesterday UTC)")

    alert = command("alert", _cmd_alert, "evaluate the alert rules and sync GitHub issues")
    alert.add_argument("--plant-regression", action="store_true", help="dry run on planted data")

    command("noise-report", _cmd_noise_report, "write the noise report from stored runs")

    site = command("build-site", _cmd_build_site, "build the static dashboard")
    site.add_argument("--out", required=True, help="output directory")
    site.add_argument("--repo", help="owner/name (default: $GITHUB_REPOSITORY)")
    return parser


def main(
    argv: list[str] | None = None,
    env: Mapping[str, str] = os.environ,
    *,
    _transports: Mapping[str, httpx.BaseTransport] | None = None,
    _sleep: Callable[[float], None] = time.sleep,
    _now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2
    if args.command == "probe" and not (
        args.repeat == 1 or (args.kind == "noise" and 1 <= args.repeat <= MAX_REPEAT)
    ):
        print(
            f"error: --repeat is for --kind noise only and must be 1 to {MAX_REPEAT}",
            file=sys.stderr,
        )
        return 2
    deps = _Deps(env, _sleep, _now, dict(_transports or {}))
    try:
        return args.handler(args, deps)
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
    except LangfuseError as exc:
        print(f"error: {exc}", file=sys.stderr)
    except (OSError, ValueError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
    return 1
