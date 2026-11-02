from rail_rag_monitor.store import read_json, read_jsonl, upsert_jsonl, write_json


def test_read_missing_returns_empty(tmp_path):
    assert read_jsonl(tmp_path / "x.jsonl") == []


def test_upsert_appends_new_keys(tmp_path):
    p = tmp_path / "x.jsonl"
    upsert_jsonl(p, [{"k": 1}], key="k")
    upsert_jsonl(p, [{"k": 2}], key="k")
    assert read_jsonl(p) == [{"k": 1}, {"k": 2}]


def test_upsert_replaces_same_key_in_place(tmp_path):
    p = tmp_path / "x.jsonl"
    upsert_jsonl(p, [{"k": 1, "v": "a"}, {"k": 2}], key="k")
    upsert_jsonl(p, [{"k": 1, "v": "b"}], key="k")
    assert read_jsonl(p) == [{"k": 1, "v": "b"}, {"k": 2}]


def test_writes_one_line_per_row_utf8(tmp_path):
    p = tmp_path / "sub" / "dir" / "x.jsonl"
    upsert_jsonl(p, [{"k": 1, "t": "é"}, {"k": 2}], key="k")
    raw = p.read_bytes()
    assert raw.endswith(b"\n")
    assert len(raw.decode("utf-8").splitlines()) == 2
    assert read_jsonl(p)[0]["t"] == "é"


def test_read_json_default(tmp_path):
    assert read_json(tmp_path / "missing.json", {"a": 1}) == {"a": 1}


def test_write_json_roundtrip(tmp_path):
    p = tmp_path / "d" / "s.json"
    write_json(p, {"a": [1, 2], "b": "é"})
    assert read_json(p, None) == {"a": [1, 2], "b": "é"}
