import random
import time
from datetime import UTC, datetime, timedelta

from rail_rag_monitor.alerts import (
    ALPHA,
    DELTA_FLOOR,
    Alert,
    evaluate,
    noise_report,
    planted_runs,
    simulate_delta,
)

IDS = ["q01", "q06", "q09", "q10", "q14", "q15", "q17", "q18"]
DELTA = 2 / 24


def make_run(i, *, fails=(), errors=(), warm=1.0, cold=3.0, kind="scheduled"):
    results = {}
    for qid in IDS:
        if qid in errors:
            res = {"pass": None, "refused": None, "grounding": None, "status": "timeout"}
            latency = None
        else:
            res = {
                "pass": qid not in fails,
                "refused": qid in fails,
                "grounding": qid not in fails,
                "status": "ok",
            }
            latency = warm
        results[qid] = {
            **res,
            "latency_s": latency,
            "cold": qid == IDS[0],
            "trace_id": f"trace-{i}-{qid}",
        }
    return {
        "run_id": f"run-{i:03d}",
        "started_at": f"2026-11-{i:02d}T06:00:00+00:00",
        "commit": "abc",
        "kind": kind,
        "demo_version": None,
        "results": results,
        "n_pass": sum(r["pass"] is True for r in results.values()),
        "n_scored": sum(r["pass"] is not None for r in results.values()),
        "n_errors": len(errors),
        "warm_p50_s": warm,
        "warm_p95_s": warm,
        "cold_latency_s": cold,
    }


def kinds(alerts):
    return [a.rule for a in alerts]


BAD = ("q01", "q06", "q09", "q10")


def test_planted_regression_fires():
    runs = [make_run(i) for i in range(14)] + [make_run(14 + i, fails=BAD) for i in range(3)]
    alerts = evaluate(runs, DELTA)
    assert kinds(alerts) == ["quality"]
    alert = alerts[0]
    assert isinstance(alert, Alert)
    d = alert.details
    assert d["fail_recent"] == 12
    assert d["n_recent"] == 24
    assert d["fail_base"] == 0
    assert d["n_base"] == 14 * 8
    assert d["p_value"] < 0.01
    assert d["recent_run_ids"] == ["run-014", "run-015", "run-016"]
    assert d["baseline_run_ids"] == [f"run-{i:03d}" for i in range(14)]
    assert {f["question_id"] for f in d["failed"]} == set(BAD)
    assert len(d["failed"]) == 12
    assert {"run_id": "run-014", "question_id": "q01", "trace_id": "trace-14-q01"} in d["failed"]


def test_small_drop_below_delta_silent():
    runs = [make_run(i) for i in range(14)]
    runs += [make_run(14, fails=("q01",)), make_run(15), make_run(16)]
    assert evaluate(runs, DELTA) == []


def test_delta_gate_is_independent_of_p_value():
    runs = [make_run(i) for i in range(14)] + [make_run(14 + i, fails=BAD) for i in range(3)]
    assert kinds(evaluate(runs, 0.6)) == []  # f_r - f_b = 0.5 < 0.6, though p << 0.01


def test_p_value_gate_blocks_noisy_baseline():
    # Baseline already fails a lot: the same 12 recent fails are no longer significant.
    base = [make_run(i, fails=BAD) for i in range(14)]
    runs = base + [make_run(14 + i, fails=BAD) for i in range(3)]
    assert evaluate(runs, 0.0) == []


def test_unscored_questions_are_not_fails():
    errors = ("q01", "q06", "q09", "q10")
    runs = [make_run(i) for i in range(14)] + [make_run(14 + i, errors=errors) for i in range(3)]
    alerts = evaluate(runs, DELTA)
    assert kinds(alerts) == ["uptime"]  # no quality alert: errors are not scored


def test_needs_7_baseline_runs():
    runs = [make_run(i) for i in range(6)] + [
        make_run(6 + i, fails=BAD, warm=5.0) for i in range(3)
    ]
    assert evaluate(runs, DELTA) == []
    runs = [make_run(i) for i in range(7)] + [
        make_run(7 + i, fails=BAD, warm=5.0) for i in range(3)
    ]
    assert sorted(kinds(evaluate(runs, DELTA))) == ["latency", "quality"]


def test_dry_run_runs_never_count_toward_baseline():
    stored_dry = [make_run(i, kind="dry-run") for i in range(10)]
    real = [make_run(10 + i) for i in range(6)]
    recent = planted_runs(make_run(99))
    assert evaluate(stored_dry + real + recent, DELTA) == []  # only 6 real baseline runs
    real.append(make_run(16))
    alerts = evaluate(stored_dry + real + recent, DELTA)
    assert kinds(alerts) == ["quality"]
    assert alerts[0].details["baseline_run_ids"] == [f"run-{i:03d}" for i in range(10, 17)]


def test_baseline_is_capped_at_14_runs():
    old_bad = [make_run(i, fails=BAD) for i in range(6)]
    clean = [make_run(6 + i) for i in range(14)]
    recent = [make_run(20 + i, fails=BAD) for i in range(3)]
    alerts = evaluate(old_bad + clean + recent, DELTA)
    assert kinds(alerts) == ["quality"]
    assert alerts[0].details["fail_base"] == 0
    assert len(alerts[0].details["baseline_run_ids"]) == 14


def test_uptime_needs_two_consecutive():
    base = [make_run(i) for i in range(10)]
    assert evaluate(base + [make_run(10, errors=("q06",))], DELTA) == []
    two = evaluate(base + [make_run(10, errors=("q06",)), make_run(11, errors=("q09",))], DELTA)
    assert kinds(two) == ["uptime"]
    d = two[0].details
    assert d["run_ids"] == ["run-010", "run-011"]
    assert {
        "run_id": "run-010",
        "question_id": "q06",
        "status": "timeout",
        "trace_id": "trace-10-q06",
    } in d["errors"]
    assert (
        evaluate(
            base + [make_run(10, errors=("q06",)), make_run(11), make_run(12, errors=("q06",))],
            DELTA,
        )
        == []
    )


def test_uptime_needs_no_baseline():
    runs = [make_run(0, errors=("q01",)), make_run(1, errors=("q01",))]
    assert kinds(evaluate(runs, DELTA)) == ["uptime"]
    assert evaluate(runs[:1], DELTA) == []
    assert evaluate([], DELTA) == []


def test_latency_two_consecutive_and_ignores_cold():
    base = [make_run(i, warm=1.0) for i in range(10)]
    cold_only = base + [make_run(10 + i, warm=1.0, cold=60.0) for i in range(3)]
    assert evaluate(cold_only, DELTA) == []
    one_slow = base + [make_run(10, warm=1.0), make_run(11, warm=1.0), make_run(12, warm=2.1)]
    assert evaluate(one_slow, DELTA) == []
    slow = base + [make_run(10, warm=1.0), make_run(11, warm=2.1), make_run(12, warm=2.1)]
    alerts = evaluate(slow, DELTA)
    assert kinds(alerts) == ["latency"]
    d = alerts[0].details
    assert d["baseline_median_s"] == 1.0
    assert d["threshold_s"] == 2.0
    assert d["recent_warm_p95_s"] == [2.1, 2.1]
    assert d["recent_run_ids"] == ["run-011", "run-012"]


def test_latency_exactly_2x_is_silent_and_none_is_skipped():
    base = [make_run(i, warm=1.0) for i in range(10)]
    exact = base + [make_run(10), make_run(11, warm=2.0), make_run(12, warm=2.0)]
    assert evaluate(exact, DELTA) == []
    # Baseline runs without a warm p95 are skipped; 6 values left is too few.
    sparse = [make_run(i, warm=None) for i in range(4)] + [make_run(4 + i) for i in range(6)]
    sparse += [make_run(10), make_run(11, warm=9.0), make_run(12, warm=9.0)]
    assert evaluate(sparse, DELTA) == []


def test_planted_runs_shape():
    last = make_run(5)
    planted = planted_runs(last)
    assert len(planted) == 3
    assert all(p["kind"] == "dry-run" for p in planted)
    assert len({p["run_id"] for p in planted}) == 3
    assert last["kind"] == "scheduled"  # input untouched
    assert last["results"]["q01"]["pass"] is True
    for p in planted:
        for qid in BAD:
            assert p["results"][qid]["refused"] is True
            assert p["results"][qid]["pass"] is False
        assert p["results"]["q14"] == last["results"]["q14"]
    assert len(planted_runs(last, n=2)) == 2


def test_planted_runs_are_clean_apart_from_the_plant():
    last = make_run(5, errors=("q01", "q14"), warm=9.0)
    for p in planted_runs(last):
        assert all(r["status"] == "ok" for r in p["results"].values())
        assert p["n_errors"] == 0
        assert p["warm_p95_s"] is None
        for qid in BAD:
            r = p["results"][qid]
            assert (r["status"], r["refused"], r["grounding"], r["pass"]) == (
                "ok",
                True,
                False,
                False,
            )
        assert p["results"]["q14"]["pass"] is True
        assert p["n_pass"] == 4
        assert p["n_scored"] == 8


def test_dry_run_after_errored_or_slow_real_run_yields_only_quality():
    base = [make_run(i) for i in range(12)]
    last = make_run(12, errors=("q14", "q01"), warm=9.0)
    alerts = evaluate(base + [last] + planted_runs(last), DELTA)
    assert kinds(alerts) == ["quality"]
    assert {f["question_id"] for f in alerts[0].details["failed"]} == set(BAD)
    assert alerts[0].details["fail_recent"] == 12  # the errored q01 is a planted fail, not an error


def test_simulate_delta_floor():
    assert simulate_delta(dict.fromkeys(IDS, 1.0)) == DELTA_FLOOR == 2 / 24


def test_simulate_delta_is_deterministic_and_grows_with_noise():
    rates = dict.fromkeys(IDS, 0.95)
    assert simulate_delta(rates, draws=2000) == simulate_delta(rates, draws=2000)
    noisy = dict.fromkeys(IDS, 0.5)
    assert simulate_delta(noisy, draws=2000) > simulate_delta(rates, draws=2000)


NOISE_RATES = {
    "q01": 0.95,
    "q06": 0.95,
    "q09": 0.95,
    "q10": 0.95,
    "q14": 0.95,
    "q15": 0.95,
    "q17": 1.0,
    "q18": 0.85,
}


def false_alarm_rate(n_baseline, *, seed=12345, windows=10_000):
    """Share of all-noise windows (`n_baseline` runs, then 3 recent) that raise a quality alert."""
    delta = simulate_delta(NOISE_RATES)
    rng = random.Random(seed)  # separate generator from simulate_delta's own
    fired = 0
    for _ in range(windows):
        runs = []
        for i in range(n_baseline + 3):
            run = make_run(i)
            for qid, rate in NOISE_RATES.items():
                if rng.random() >= rate:
                    run["results"][qid]["pass"] = False
            runs.append(run)
        if any(a.rule == "quality" for a in evaluate(runs, delta)):
            fired += 1
    return delta, fired / windows


def test_noise_does_not_fire():
    started = time.perf_counter()
    delta, rate = false_alarm_rate(14)
    assert delta >= DELTA_FLOOR
    print(f"delta={delta:.4f} false alarms={rate:.2%} in {time.perf_counter() - started:.1f}s")
    assert rate <= 0.01


def test_noise_does_not_fire_with_the_minimum_7_run_baseline():
    # delta assumes a full 14-run baseline; with only 7 the baseline is noisier. Measured 0.30%.
    delta, rate = false_alarm_rate(7)
    print(f"7-run baseline: delta={delta:.4f} false alarms={rate:.2%}")
    assert rate <= 0.01


def test_borderline_three_fails_fire_at_the_floor_but_not_at_the_noise_delta():
    # 3 fails in 24 recent results over a clean 14-run baseline: rate 0.125, Fisher p = 0.0049.
    runs = [make_run(i) for i in range(14)]
    runs += [
        make_run(14, fails=("q01",)),
        make_run(15, fails=("q06",)),
        make_run(16, fails=("q09",)),
    ]
    alerts = evaluate(runs, DELTA)
    assert kinds(alerts) == ["quality"]
    assert alerts[0].details["fail_recent"] == 3
    assert alerts[0].details["p_value"] < ALPHA
    noise_delta = simulate_delta(NOISE_RATES)
    assert abs(noise_delta - 0.1369) < 5e-5
    assert 3 / 24 < noise_delta
    assert evaluate(runs, noise_delta) == []


def test_alpha_constant():
    assert ALPHA == 0.01


FIXED_NOW = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)


def noise_run(i, q18, *, kind="noise"):
    run = make_run(i, kind=kind)
    run["results"]["q18"]["pass"] = q18
    return run


def test_noise_report():
    runs = [
        noise_run(0, True),
        noise_run(1, False),
        noise_run(2, True, kind="scheduled"),
        noise_run(3, True),
        noise_run(4, False, kind="dry-run"),  # ignored
        noise_run(5, False, kind="manual"),  # ignored
    ]
    report = noise_report(runs, now=lambda: FIXED_NOW)
    assert set(report) == {"n_runs", "pass_rates", "delta", "flips", "computed_at"}
    assert report["n_runs"] == 4
    assert report["pass_rates"]["q18"] == 0.75
    assert report["pass_rates"]["q01"] == 1.0
    assert set(report["pass_rates"]) == set(IDS)
    assert report["flips"]["q18"] == 2  # pass -> fail -> pass
    assert report["flips"]["q01"] == 0
    assert report["delta"] == simulate_delta(report["pass_rates"])
    assert report["computed_at"] == FIXED_NOW.isoformat()


def test_noise_report_unscored_results_are_not_fails_or_flips():
    # q18 is unscored in run 1: it is dropped from the rate and does not break the flip chain.
    runs = [noise_run(0, True), noise_run(1, None), noise_run(2, True), noise_run(3, False)]
    report = noise_report(runs, now=lambda: FIXED_NOW)
    assert report["n_runs"] == 4
    assert report["pass_rates"]["q18"] == 2 / 3
    assert report["flips"]["q18"] == 1
    # A question that is never scored has no rate and no flips entry.
    never = [noise_run(0, None), noise_run(1, None)]
    report = noise_report(never, now=lambda: FIXED_NOW)
    assert "q18" not in report["pass_rates"]
    assert "q18" not in report["flips"]
    assert report["pass_rates"]["q01"] == 1.0


def test_noise_report_without_qualifying_runs():
    for runs in ([], [noise_run(0, True, kind="dry-run"), noise_run(1, True, kind="manual")]):
        report = noise_report(runs, now=lambda: FIXED_NOW)
        assert report == {
            "n_runs": 0,
            "pass_rates": {},
            "delta": DELTA_FLOOR,
            "flips": {},
            "computed_at": FIXED_NOW.isoformat(),
        }


def test_noise_report_default_clock_is_utc_iso():
    stamp = noise_report([])["computed_at"]
    assert datetime.fromisoformat(stamp).utcoffset() == timedelta(0)
