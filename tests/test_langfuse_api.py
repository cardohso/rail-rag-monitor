import base64

import httpx
import pytest

from rail_rag_monitor.langfuse_api import LangfuseAPI, LangfuseError


def make_api(handler, sleeps=None):
    sleeps = sleeps if sleeps is not None else []
    api = LangfuseAPI(
        "pk",
        "sk",
        "https://lf.test",
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
    )
    return api, sleeps


def test_basic_auth_header():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"data": [], "meta": {}})

    api, _ = make_api(handler)
    assert list(api.iter_observations()) == []
    expected = "Basic " + base64.b64encode(b"pk:sk").decode()
    assert seen[0].headers["Authorization"] == expected
    assert seen[0].url.path == "/api/public/v2/observations"


def test_follows_cursor():
    seen = []

    def handler(request):
        seen.append(request)
        if "cursor" not in request.url.params:
            return httpx.Response(
                200, json={"data": [{"id": "a"}, {"id": "b"}], "meta": {"cursor": "c1"}}
            )
        return httpx.Response(200, json={"data": [{"id": "c"}], "meta": {}})

    api, _ = make_api(handler)
    rows = list(api.iter_observations(type="GENERATION"))
    assert [r["id"] for r in rows] == ["a", "b", "c"]
    assert len(seen) == 2
    assert seen[1].url.params["cursor"] == "c1"
    assert all(r.url.params["limit"] == "1000" for r in seen)
    assert all(r.url.params["type"] == "GENERATION" for r in seen)


def test_retries_429_then_succeeds():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return httpx.Response(200, json={"data": [{"id": "a"}], "meta": {"cursor": None}})

    api, sleeps = make_api(handler)
    assert [r["id"] for r in api.iter_observations()] == ["a"]
    assert sleeps == [2]


def test_backoff_without_retry_after():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(503)
        return httpx.Response(200, json={"data": [], "meta": {}})

    api, sleeps = make_api(handler)
    assert list(api.iter_observations()) == []
    assert sleeps == [2, 4]


def test_raises_after_5_failures():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503)

    api, _ = make_api(handler)
    with pytest.raises(LangfuseError) as exc:
        list(api.iter_observations())
    assert len(calls) == 5
    message = str(exc.value)
    assert "pk" not in message
    assert "sk" not in message
    assert base64.b64encode(b"pk:sk").decode() not in message


def test_client_error_not_retried():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(401)

    api, sleeps = make_api(handler)
    with pytest.raises(LangfuseError):
        list(api.iter_observations())
    assert len(calls) == 1
    assert sleeps == []


def test_from_env_none_without_keys():
    assert LangfuseAPI.from_env({}) is None
    assert LangfuseAPI.from_env({"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": ""}) is None
    api = LangfuseAPI.from_env(
        {
            "LANGFUSE_PUBLIC_KEY": "pk",
            "LANGFUSE_SECRET_KEY": "sk",
            "LANGFUSE_HOST": "https://eu.example",
        }
    )
    assert isinstance(api, LangfuseAPI)


def test_project_id():
    def handler(request):
        assert request.url.path == "/api/public/projects"
        return httpx.Response(200, json={"data": [{"id": "p1"}]})

    api, _ = make_api(handler)
    assert api.project_id() == "p1"
