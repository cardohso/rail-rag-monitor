import pytest

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


def test_upsert_duplicate_keys_in_one_batch_last_wins_first_position(tmp_path):
    p = tmp_path / "x.jsonl"
    upsert_jsonl(p, [{"k": 1, "v": "old"}, {"k": 2, "v": "x"}], key="k")
    upsert_jsonl(
        p,
        [{"k": 3, "v": "first"}, {"k": 1, "v": "mid"}, {"k": 3, "v": "last"}, {"k": 1, "v": "new"}],
        key="k",
    )
    assert read_jsonl(p) == [{"k": 1, "v": "new"}, {"k": 2, "v": "x"}, {"k": 3, "v": "last"}]
    fresh = tmp_path / "fresh.jsonl"
    upsert_jsonl(fresh, [{"k": 7, "v": "a"}, {"k": 8, "v": "b"}, {"k": 7, "v": "c"}], key="k")
    assert read_jsonl(fresh) == [{"k": 7, "v": "c"}, {"k": 8, "v": "b"}]


def test_corrupt_line_names_file_and_line_but_not_contents(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"k": 1}\n\n{"k": SECRET-CONTENT broken\n', encoding="utf-8")
    with pytest.raises(ValueError) as err:
        read_jsonl(p)
    message = str(err.value)
    assert str(p) in message
    assert "line 3" in message
    assert "SECRET" not in message
    assert err.value.__cause__ is None and err.value.__suppress_context__


def test_upsert_into_a_corrupt_file_fails_without_touching_it(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"k": 1}\nnot json\n', encoding="utf-8")
    before = p.read_bytes()
    with pytest.raises(ValueError, match="line 2"):
        upsert_jsonl(p, [{"k": 2}], key="k")
    assert p.read_bytes() == before


def test_stored_row_missing_the_key_names_file_and_line(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"k": 1}\n{"other": "SECRET-CONTENT"}\n', encoding="utf-8")
    with pytest.raises(ValueError) as err:
        upsert_jsonl(p, [{"k": 2}], key="k")
    message = str(err.value)
    assert str(p) in message
    assert "line 2" in message
    assert "SECRET" not in message


def test_stored_row_that_is_not_an_object_is_rejected(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"k": 1}\n[1, 2]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        upsert_jsonl(p, [{"k": 2}], key="k")
