import random
import time

from rail_rag_monitor.alerts import (
    ALPHA,
    DELTA_FLOOR,
    Alert,
    evaluate,
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
    last = make_run(5, errors=("q14",))
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


def test_noise_does_not_fire():
    started = time.perf_counter()
    delta = simulate_delta(NOISE_RATES)
    assert delta >= DELTA_FLOOR
    rng = random.Random(12345)  # separate generator from simulate_delta's own
    windows = 10_000
    fired = 0
    for _ in range(windows):
        runs = []
        for i in range(17):
            run = make_run(i)
            for qid, rate in NOISE_RATES.items():
                if rng.random() >= rate:
                    run["results"][qid]["pass"] = False
            runs.append(run)
        if any(a.rule == "quality" for a in evaluate(runs, delta)):
            fired += 1
    print(
        f"delta={delta:.4f} false alarms={fired}/{windows} in {time.perf_counter() - started:.1f}s"
    )
    assert fired / windows <= 0.01


def test_alpha_constant():
    assert ALPHA == 0.01
