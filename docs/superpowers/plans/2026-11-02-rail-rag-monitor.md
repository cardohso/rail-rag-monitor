# rail-rag-monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A daily GitHub Actions job that probes the live rail-regulation-rag demo, summarises its Langfuse traces, alerts on real regressions through GitHub issues, and publishes an aggregates-only dashboard to GitHub Pages.

**Architecture:** Small pure functions (collector summary, probe scoring, alert rules, site builder) with thin I/O edges (Langfuse v2 API, demo HTTP, GitHub REST, JSONL files on a `data` branch). Every edge takes an injectable `httpx` transport or clock so the tests run offline.

**Tech Stack:** Python 3.12, uv, httpx, goldpass v0.2.0 (library, in-process scoring), pytest, ruff, Vega-Lite (CDN, pinned), GitHub Actions, GitHub Pages.

**Spec:** `docs/superpowers/specs/2026-11-02-rail-rag-monitor-design.md`

## Global Constraints

- Free tiers only; no paid service, no database server.
- Never commit or print Langfuse keys. Local: `.env` (gitignored). CI: secrets `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, added by the owner.
- No question, answer or citation text from real traces is written to disk, logs, issues or the site. Probe rows hold scores and trace ids only.
- Langfuse reads use `GET /api/public/v2/observations` only (plus `GET /api/public/projects` for the project id). Host default `https://cloud.langfuse.com`.
- Production filter: `environment=azure-container-apps`, root `name=rag-ask`.
- Demo base URL: `https://rail-rag.calmcliff-0073b8b6.switzerlandnorth.azurecontainerapps.io`. Probe: 13 s between `/ask` calls, 120 s timeout, ≤ 3 retries on 503 with 15 s backoff, no retry on 429.
- Alert constants: `MIN_BASELINE_RUNS = 7`, `RECENT_RUNS = 3`, `BASELINE_MAX_RUNS = 14`, `ALPHA = 0.01`, `DELTA_FLOOR = 2/24`, `LATENCY_FACTOR = 2.0`, `CLEAN_RUNS_TO_CLOSE = 3`.
- Gates (stop and ask the owner): before creating the GitHub repo, before the first push, before making it public, before any live-demo traffic beyond the approved probe, before publishing the README.
- Model per task as in the owner's table: Opus for Task 7 (noise and δ), Sonnet for the rest, Haiku for lint and CI fixes.
- Licence MIT, copyright "2026 João Pedro Cardoso".

## Review Focus

1. A root observation with no `output` (crashed before `root.update`, or still flushing) must be counted in `n_requests` but left out of refusal and grounding denominators, never read as "answered". Test in Task 4.
2. The demo scales to zero: the first `/ask` of a run can time out while the container starts. The cold call is retried once on timeout before it counts as an error, so a slow start alone does not fire R1. Test in Task 5.
3. If the owner closes an alert issue by hand, the next firing must open a new issue, not comment on the closed one. Test in Task 8.
4. Missing Langfuse secrets (not yet added, or a fork) must skip the collector with a warning while the probe, alerts and site still run. Test in Task 10.
5. Re-running the workflow for the same day must replace that day's traffic row, not duplicate it; a re-attempt of the same run id replaces its probe row. Tests in Tasks 2 and 4.

---

### Task 1: Repo skeleton and CI (owner's step 2, Sonnet)

⏸ Gate: confirm the local folder (`~/Documents/rail-rag-monitor`) and repo name before starting. Do not create the GitHub repo.

**Files:**
- Create: `pyproject.toml`, `.python-version` (`3.12`), `.gitignore`, `.env.example`, `LICENSE`, `README.md` (title + one-line summary + "work in progress"), `src/rail_rag_monitor/__init__.py` (`__version__ = "0.1.0"`), `src/rail_rag_monitor/__main__.py` (calls `cli.main`), `.github/workflows/ci.yml`, `tests/test_smoke.py`
- Move in: `docs/superpowers/specs/…design.md`, `docs/superpowers/plans/…monitor.md`

**Interfaces:**
- Produces: package `rail_rag_monitor`; `uv run pytest`, `uv run ruff check`, `uv run ruff format --check` all work.

- [ ] **Step 1:** `git init -b main`; create the files. `pyproject.toml`: hatchling build, `requires-python = ">=3.12"`, deps `httpx>=0.28`, `goldpass @ git+https://github.com/cardohso/goldpass@v0.2.0`; dev group `pytest`, `ruff`; ruff `line-length = 100`, `select = ["E","F","I","UP","B"]`. `.gitignore` includes `.env`, `.venv/`, `site/`, `data/`, `__pycache__/`. `.env.example` lists `LANGFUSE_PUBLIC_KEY=`, `LANGFUSE_SECRET_KEY=`, `LANGFUSE_HOST=https://cloud.langfuse.com`, `DEMO_URL=` with the demo base URL, `GITHUB_TOKEN=`.
- [ ] **Step 2:** `tests/test_smoke.py::test_version` asserts `rail_rag_monitor.__version__ == "0.1.0"`.
- [ ] **Step 3:** `ci.yml`: on push and pull_request; `astral-sh/setup-uv`, `uv sync --locked`, ruff check, ruff format --check, pytest. Pin actions to their current major versions.
- [ ] **Step 4:** Run `uv sync && uv run ruff check && uv run ruff format --check && uv run pytest`. Expected: 1 passed, no lint errors.
- [ ] **Step 5:** Commit `chore: skeleton, CI, spec and plan`.

### Task 2: JSONL store

**Files:** Create `src/rail_rag_monitor/store.py`, `tests/test_store.py`

**Interfaces:**
- Produces: `read_jsonl(path: Path) -> list[dict]`; `upsert_jsonl(path: Path, rows: Iterable[dict], key: str) -> None`; `read_json(path: Path, default: Any) -> Any`; `write_json(path: Path, obj: Any) -> None`. Constants `PROBE_RUNS = Path("metrics/probe_runs.jsonl")`, `TRAFFIC_DAILY = Path("metrics/traffic_daily.jsonl")`, `ALERT_STATE = Path("alerts/state.json")`, `NOISE = Path("alerts/noise.json")` (relative to the data dir).

- [ ] **Step 1: Tests**
  - `test_read_missing_returns_empty`: `read_jsonl(tmp/"x.jsonl") == []`.
  - `test_upsert_appends_new_keys`: upsert `[{"k":1}]` then `[{"k":2}]` → `[{"k":1},{"k":2}]`.
  - `test_upsert_replaces_same_key_in_place`: upsert `[{"k":1,"v":"a"},{"k":2}]` then `[{"k":1,"v":"b"}]` → `[{"k":1,"v":"b"},{"k":2}]`.
  - `test_writes_one_line_per_row_utf8`: row with `"é"` round-trips; file ends with `\n`; creates parent dirs.
  - `test_read_json_default` and `test_write_json_roundtrip`.
- [ ] **Step 2:** Run `uv run pytest tests/test_store.py -v`. Expected: FAIL (module missing).
- [ ] **Step 3:** Implement. Write to a temp file in the same dir and `os.replace` it.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit `feat: JSONL store`.

### Task 3: Langfuse v2 client and shared stats

**Files:** Create `src/rail_rag_monitor/langfuse_api.py`, `src/rail_rag_monitor/stats.py`, `tests/test_langfuse_api.py`, `tests/test_stats.py`

**Interfaces:**
- Produces:
  - `class LangfuseError(RuntimeError)`.
  - `class LangfuseAPI(public_key: str, secret_key: str, host: str = "https://cloud.langfuse.com", *, transport: httpx.BaseTransport | None = None, sleep: Callable[[float], None] = time.sleep)` with `iter_observations(**params: str | int) -> Iterator[dict]` and `project_id() -> str`.
  - `LangfuseAPI.from_env(env: Mapping[str, str]) -> LangfuseAPI | None` (None when either key is empty).
  - `percentile(values: Sequence[float], q: float) -> float | None` (linear interpolation, numpy's default; None for empty).
  - `fisher_one_sided(fail_recent: int, n_recent: int, fail_base: int, n_base: int) -> float` (P of ≥ `fail_recent` fails in the recent group under the hypergeometric null, via `math.comb`).

- [ ] **Step 1: Tests** (`httpx.MockTransport`; `sleep` records calls)
  - `test_basic_auth_header`: request has `Authorization: Basic base64("pk:sk")`.
  - `test_follows_cursor`: page 1 `meta.cursor="c1"`, page 2 no cursor → yields both pages' rows in order; second request has `cursor=c1`; every request has `limit=1000`.
  - `test_retries_429_then_succeeds`: 429 with `Retry-After: 2`, then 200 → rows returned, `sleep` called with `2`.
  - `test_raises_after_5_failures`: five 503s → `LangfuseError`; no key value in the message.
  - `test_from_env_none_without_keys`.
  - `test_project_id`: `GET /api/public/projects` → `{"data":[{"id":"p1"}]}` → `"p1"`.
  - `test_percentile`: `[1,2,3,4]` q=0.5 → 2.5, q=0.95 → 3.85; `[]` → None.
  - `test_fisher`: `(0,24,0,112)` → 1.0; `(4,24,0,112)` → `comb(24,4)/comb(136,4)` within 1e-12; `(1,24,5,112)` > 0.5.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. Retry 429 and 5xx up to 5 attempts, waiting `Retry-After` if present else `2 * 2**attempt` s.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit `feat: Langfuse v2 observations client and stats helpers`.

### Task 4: Collector (owner's step 3, Sonnet)

**Files:** Create `src/rail_rag_monitor/collector.py`, `tests/fixtures/roots_day.json`, `tests/fixtures/generations_day.json`, `tests/test_collector.py`

**Interfaces:**
- Consumes: `LangfuseAPI.iter_observations`, `percentile`.
- Produces:
  - `@dataclass TrafficDay` with fields exactly as spec §6 (`day: str` ISO date, `n_requests`, `n_errors`, `refusal_rate`, `refusals_by_reason: dict[str,int]`, `grounding_failure_share`, `p50_s`, `p95_s`, `tokens_per_request`, `cost_per_request`; rates and latencies `float | None`), with `to_dict()`.
  - `summarize_day(day: date, roots: Iterable[dict], generations: Iterable[dict], exclude_trace_ids: set[str]) -> TrafficDay` (pure).
  - `collect_day(api: LangfuseAPI, day: date, exclude_trace_ids: set[str]) -> TrafficDay`.

Definitions: an error is `level == "ERROR"`. `refusal_rate` = refused ÷ roots that are not errors and have an output. `grounding_failure_share` = refusals with reason `insufficient_context` or `no_citations` ÷ all refusals (None if no refusals). Latency percentiles over non-error roots. `tokens_per_request` = sum of generation `totalUsage` ÷ `n_requests`, None when every generation has empty `usageDetails`; same for `totalCost`. `output` may be a dict or a JSON string.

- [ ] **Step 1: Tests**
  - `test_summary_from_fixture`: fixture of 6 roots (3 answered, 1 `low_retrieval_score`, 1 `no_citations`, 1 ERROR) → `n_requests=6, n_errors=1, refusal_rate=0.4, refusals_by_reason={"low_retrieval_score":1,"no_citations":1}, grounding_failure_share=0.5`.
  - `test_excludes_probe_traces`: a root whose `traceId` is in `exclude_trace_ids` is not counted, and nor are its generations.
  - `test_output_as_json_string`: same result when `output` is a JSON string.
  - `test_missing_output_counted_but_not_in_rates` (Review Focus 1).
  - `test_usage_missing_gives_none`: generations with `usageDetails={}` and `totalCost=None` → both None, not 0.
  - `test_no_text_in_row`: `json.dumps(summary.to_dict())` contains no answer or question string from the fixture.
  - `test_collect_day_window` (MockTransport): requests carry `environment=azure-container-apps`, `isRootObservation=true`, `name=rag-ask`, `fromStartTime=<day>T00:00:00Z`, `toStartTime=<day+1>T00:00:00Z`; root requests ask `fields=core,basic,io,metrics`, generation requests `type=GENERATION&fields=core,usage`.
  - `test_recollect_replaces_day` (Review Focus 5): `upsert_jsonl` twice with the same `day` → one row.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit `feat: daily traffic collector`.

### Task 5: Probe (owner's step 4, Sonnet)

⏸ The probe's size and schedule are approved (spec D2). This task runs offline only; the first live call happens in Task 10.

**Files:** Create `probe/golden.jsonl` (q01, q06, q09, q10, q14, q15, q17, q18 copied verbatim from goldpass `examples/rail/golden.jsonl`), `probe/NOTICE.md` (source and MIT attribution), `src/rail_rag_monitor/probe.py`, `tests/test_probe.py`

**Interfaces:**
- Consumes: `goldpass.golden.load_golden`, `goldpass.mapping.to_record`, `goldpass.mapping.FieldMap`, `goldpass.metrics.score_item`, `percentile`.
- Produces:
  - `@dataclass QuestionResult(pass_: bool | None, refused: bool | None, grounding: bool | None, status: str, latency_s: float | None, cold: bool, trace_id: str | None)`; `to_dict()` writes the key `pass` (not `pass_`).
  - `@dataclass ProbeRun` with the spec §6 fields; `results: dict[str, QuestionResult]`; `to_dict()`.
  - `question_pass(expected: str, refused: bool | None, grounding: bool | None) -> bool | None`.
  - `run_probe(base_url: str, golden_path: Path, *, run_id: str, kind: str, commit: str, client: httpx.Client | None = None, sleep: Callable[[float], None] = time.sleep, now: Callable[[], datetime] = …) -> ProbeRun`.
  - `PROBE_GOLDEN = Path("probe/golden.jsonl")`.

`status` values: `"ok"`, `"http_<code>"`, `"timeout"`, `"connection"`, `"rate_limited"`. Only `"ok"` responses are scored, by `score_item(item, to_record(body, item.id, FieldMap(), latency))`. `question_pass`: `answer` → `refused is False and grounding is True`; `refuse` → `refused is True`; anything not scored → None. `n_scored` counts non-None passes; `n_errors` counts non-`ok` statuses.

- [ ] **Step 1: Tests** (MockTransport demo; `sleep` recorder; fake clock)
  - `test_golden_has_8_items`: ids as listed; 6 `answer`, 2 `refuse`.
  - `test_all_pass`: mocked responses that answer the 6 with citations to their `source_doc_ids` and refuse the 2 → `n_pass == 8`, `n_scored == 8`, `n_errors == 0`.
  - `test_wrong_document_fails`: q14 cites `tsi-prm` → `results["q14"].pass_ is False`, `grounding is False`.
  - `test_refused_answerable_fails`: q01 refused → pass False.
  - `test_pacing`: `sleep` total between `/ask` calls ≥ 13 s each; one `GET /health` first and its `version` stored as `demo_version`.
  - `test_429_not_retried`: one 429 → that question `status == "rate_limited"`, `pass_ is None`, exactly one request for it.
  - `test_503_retried_3_times`: 503, 503, 503, 200 → `ok`; 503 ×4 → `http_503`; backoff sleeps of 15 s.
  - `test_cold_timeout_retried_once` (Review Focus 2): first `/ask` raises `httpx.ReadTimeout` then succeeds → `ok`, `cold is True`; a timeout on a later question is not retried → `timeout`.
  - `test_latency_split`: `cold_latency_s` is the first call's; `warm_p50_s`/`warm_p95_s` use the other 7 `ok` calls.
  - `test_trace_ids_kept` and `test_no_answer_text_in_row`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit `feat: synthetic probe scored with goldpass`.

### Task 6: Alert rules (owner's step 6, part 1, Sonnet)

**Files:** Create `src/rail_rag_monitor/alerts.py`, `tests/test_alerts.py`

**Interfaces:**
- Consumes: probe run dicts (as written by `ProbeRun.to_dict()`), `fisher_one_sided`, `percentile`.
- Produces:
  - `@dataclass Alert(rule: Literal["uptime","quality","latency"], summary: str, details: dict)`.
  - `evaluate(runs: list[dict], delta: float) -> list[Alert]` (runs in chronological order).
  - `planted_runs(last_run: dict, n: int = 3) -> list[dict]` (copies with q01, q06, q09, q10 set to `refused=True, pass=False`, `kind="dry-run"`).
  - `simulate_delta(pass_rates: dict[str, float], *, seed: int = 0, draws: int = 10_000) -> float` (99th percentile of `f_r − f_b`, floored at `DELTA_FLOOR`).
  - The constants from Global Constraints.

Rules exactly as spec §7. Quality: recent = last 3 runs, baseline = up to 14 runs before them, and it needs ≥ 7 baseline runs; counts question-runs with `pass` not None. Latency: compares each of the last 2 runs' `warm_p95_s` with 2× the median `warm_p95_s` of up to 14 runs before them (≥ 7 needed). `details` carries the numbers, window run ids, failed question ids and their trace ids.

- [ ] **Step 1: Tests**
  - `test_planted_regression_fires`: 14 clean runs + 3 runs with 4 answerable fails each, `delta=2/24` → exactly one `quality` alert; `details["fail_recent"] == 12`; `details["p_value"] < 0.01`.
  - `test_noise_does_not_fire`: seeded generator draws 10 000 windows of 17 runs from per-question pass rates `{q01, q06, q09, q10, q14, q15: 0.95; q17: 1.0; q18: 0.85}`; with `delta = simulate_delta(rates)`, quality fires in ≤ 1% of windows.
  - `test_small_drop_below_delta_silent`: one extra fail in the window → no alert.
  - `test_needs_7_baseline_runs`: 6 baseline + 3 bad → no quality or latency alert.
  - `test_uptime_needs_two_consecutive`: one run with an error → none; two in a row → one `uptime`; error, clean, error → none.
  - `test_latency_two_consecutive_and_ignores_cold`: cold latency 60 s with warm p95 normal → none; two runs at 2.1× baseline median warm p95 → one `latency`.
  - `test_simulate_delta_floor`: all rates 1.0 → `2/24`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. `simulate_delta` draws the baseline and recent windows independently from the same rates with `random.Random(seed)`.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit `feat: alert rules with planted-regression and noise tests`.

### Task 7: Noise report and δ (owner's step 5, **Opus**)

⏸ Needs real data: the 3 back-to-back noise runs plus daily runs (from Task 10). Code and tests can land first; the δ decision waits for at least 7 runs.

**Files:** Modify `src/rail_rag_monitor/alerts.py` (add `noise_report`), `tests/test_alerts.py`

**Interfaces:**
- Produces: `noise_report(runs: list[dict]) -> dict` returning `{"n_runs", "pass_rates": {qid: float}, "delta", "flips": {qid: int}, "computed_at"}`, using runs of kind `noise` and `scheduled` only; written to `NOISE` by the CLI.

- [ ] **Step 1: Test** `test_noise_report`: 4 runs where q18 passes 3 of 4 → `pass_rates["q18"] == 0.75`, `flips["q18"] == 2` (pass→fail→pass changes counted), `delta == simulate_delta(pass_rates)`; `dry-run` and `manual` runs ignored.
- [ ] **Step 2:** Run. Expected: FAIL. **Step 3:** Implement. **Step 4:** Run. Expected: PASS. **Step 5:** Commit `feat: noise report`.
- [ ] **Step 6 (after ≥ 7 real runs, Opus):** run `uv run python -m rail_rag_monitor noise-report`, review the per-question rates with the owner, confirm or adjust δ and the rule, and record the reasoning for the README. ⏸ Owner review.

### Task 8: GitHub issue notifier (owner's step 6, part 2, Sonnet)

**Files:** Create `src/rail_rag_monitor/github_issues.py`, `tests/test_github_issues.py`

**Interfaces:**
- Consumes: `Alert`, `read_json`/`write_json` state shape `{"rules": {rule: {"issue": int | None, "clean_runs": int}}}`.
- Produces:
  - `class GitHubIssues(repo: str, token: str, *, transport: httpx.BaseTransport | None = None)` with `create(title, body, labels) -> int`, `comment(number, body)`, `close(number)`, `is_open(number) -> bool`.
  - `trace_url(project_id: str, trace_id: str) -> str` → `https://cloud.langfuse.com/project/{project_id}/traces/{trace_id}`.
  - `sync(alerts: list[Alert], state: dict, gh: GitHubIssues, *, run_url: str, project_id: str | None, dry_run: bool) -> dict` (returns the new state).
  - `render_body(alert: Alert, run_url: str, project_id: str | None) -> str` (Markdown table of numbers, failed questions with trace links; plain trace ids when `project_id` is None).

Titles: `[monitor] <rule> alert` (prefix `[dry-run] ` when `dry_run`). Labels: `monitor-alert`, `rule:<rule>`, plus `dry-run`. Dry runs never touch the non-dry-run state entries.

- [ ] **Step 1: Tests** (MockTransport records requests)
  - `test_opens_issue_when_none`: one quality alert, empty state → one `POST /repos/{repo}/issues`, state `issue` set, `clean_runs == 0`.
  - `test_comments_when_open`: state has open issue 7 → `POST …/issues/7/comments`, no new issue.
  - `test_reopens_new_when_closed_by_hand` (Review Focus 3): `is_open(7)` False → new issue created, state updated.
  - `test_closes_after_3_clean_runs`: three `sync([] …)` calls → comment "recovered" then `PATCH state=closed` on the third; `issue` reset to None.
  - `test_body_has_trace_links` and `test_body_without_project_id`.
  - `test_dry_run_labels_and_state_isolated`.
- [ ] **Step 2:** Run. Expected: FAIL. **Step 3:** Implement. **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit `feat: GitHub issue alerts`.

### Task 9: Static dashboard (owner's step 7, part 1, Sonnet)

**Files:** Create `src/rail_rag_monitor/dashboard.py`, `src/rail_rag_monitor/templates/index.html`, `tests/test_dashboard.py`

**Interfaces:**
- Consumes: probe run dicts, traffic day dicts, alert state, noise dict.
- Produces: `build_site(probe_runs: list[dict], traffic_days: list[dict], alert_state: dict, noise: dict | None, out_dir: Path, *, repo: str) -> Path` (writes `out_dir/index.html`, returns it).

The template holds the page shell and Vega-Lite specs; `build_site` embeds one JSON blob built from an **allow-list** of fields (spec §6 aggregate fields, per-question `pass`/`status`/`latency_s`, run ids, dates). Sections and copy as spec §8. Vega, Vega-Lite and vega-embed loaded from jsDelivr at exact versions checked to exist when implementing. Light and dark colour schemes. Empty data shows "no runs yet", not a broken chart.

- [ ] **Step 1: Tests**
  - `test_builds_from_fixture`: output exists, contains every section heading and the `δ` value.
  - `test_allow_list_drops_unknown_fields`: a run row with an extra `"answer": "SECRET-TEXT"` and a traffic row with `"question"` → neither string appears in the HTML.
  - `test_empty_store`: builds with "no runs yet".
  - `test_tokens_not_reported_copy`: traffic rows with `tokens_per_request: None` → "not reported by the app".
- [ ] **Step 2:** Run. Expected: FAIL. **Step 3:** Implement. **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Open the built page with the built-in browser at phone and desktop widths, both themes; fix layout issues.
- [ ] **Step 6:** Commit `feat: static dashboard`.

### Task 10: CLI and monitor workflow (owner's step 7, part 2, Sonnet)

⏸ Gates in this task: create the GitHub repo (private or public, owner decides), run gitleaks, first push, owner adds secrets, first live workflow run (the approved 3 noise runs), enable Pages.

**Files:** Create `src/rail_rag_monitor/cli.py`, `.github/workflows/monitor.yml`, `tests/test_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `main(argv: list[str] | None = None, env: Mapping[str, str] = os.environ) -> int` with subcommands, all taking `--data DIR` (default `data`):
  - `probe --kind {scheduled,manual,noise} [--repeat N]` → upserts `PROBE_RUNS` by `run_id` (`<GITHUB_RUN_ID>-<GITHUB_RUN_ATTEMPT>-<i>`, or a UTC timestamp locally).
  - `collect [--day YYYY-MM-DD]` (default yesterday UTC) → upserts `TRAFFIC_DAILY` by `day`, excluding all stored probe trace ids. Without keys: prints `Langfuse keys not set: skipping collection` and returns 0.
  - `alert [--plant-regression]` → `evaluate` on stored runs (plus `planted_runs` in memory when planting), then `sync` with `dry_run` set accordingly; δ from `NOISE`, or `DELTA_FLOOR` if absent.
  - `noise-report`, `build-site --out site`.

- [ ] **Step 1: Tests**
  - `test_collect_without_keys_skips` (Review Focus 4): empty env → exit 0, message printed, no file written.
  - `test_probe_then_alert_end_to_end`: MockTransport demo + GitHub; 14 stored clean runs; `alert --plant-regression` → one `[dry-run]` issue request; `PROBE_RUNS` unchanged on disk.
  - `test_probe_rerun_same_id_replaces` (Review Focus 5).
- [ ] **Step 2:** Run. Expected: FAIL. **Step 3:** Implement. **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** `monitor.yml`: `schedule: cron "17 6 * * *"`; `workflow_dispatch` inputs `plant_regression` (bool, false), `noise_runs` (number, 0). Job: checkout `main`; checkout `data` branch into `data/` (create an orphan `data` branch on first run); `uv sync --locked`; probe (`--kind noise --repeat <noise_runs>` when `noise_runs > 0`, `--kind scheduled` on cron, `--kind manual` otherwise; skipped when `plant_regression`); collect; alert; build-site; commit `data/` as `github-actions[bot]` if changed and push to `data`; `actions/upload-pages-artifact` + `actions/deploy-pages`. Permissions and concurrency as spec §9. Env: Langfuse secrets, `GITHUB_TOKEN`, `DEMO_URL`.
- [ ] **Step 6:** `actionlint` (if installed) and full `uv run pytest`. Expected: clean, all pass.
- [ ] **Step 7:** Commit `feat: CLI and daily monitor workflow`.
- [ ] **Step 8:** ⏸ Owner: create repo, `gitleaks detect` (working tree and history) clean, push, add secrets, enable Pages (source: GitHub Actions). Then dispatch once with `noise_runs=3`; check the run log, the `data` branch rows and the Pages URL.
- [ ] **Step 9:** Check whether workflow commits to `data` keep the schedule alive past 60 days of inactivity (GitHub docs); record the answer for the README.

### Task 11: README and launch (owner's step 8, Sonnet)

⏸ Show the owner before publishing or making the repo public.

**Files:** Modify `README.md`; create `docs/images/dashboard.png`, `docs/images/alert-issue.png`

- [ ] **Step 1:** README sections: 30-second summary; CI badge; Mermaid diagram (from spec §4); what is measured and why (spec §3 facts, including why "has a citation" is not used); the alert rule with the measured δ and its data from `alerts/noise.json`; planted-regression demo (test name + link to the dry-run issue); limitations (spec §11 plus the Task 10 Step 9 answer); run locally (`cp .env.example .env`, `uv sync`, CLI commands); licence. Numbers only from the data branch or test output, never typed by hand.
- [ ] **Step 2:** Screenshots of the live dashboard and the dry-run issue with the built-in browser.
- [ ] **Step 3:** `uv run ruff check && uv run pytest`; `gitleaks detect`. Expected: clean.
- [ ] **Step 4:** Commit `docs: README, screenshots, limitations`; ⏸ owner review before push.
- [ ] **Step 5 (done check):** ≥ 3 consecutive green scheduled runs; dashboard live with real data; dry-run issue exists; CI green.
