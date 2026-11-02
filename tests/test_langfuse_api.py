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


# --- hardening (H-B): Retry-After, malformed bodies, env parsing, cursor loop guard ----------


def _flaky(retry_after):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            headers = {} if retry_after is None else {"Retry-After": retry_after}
            return httpx.Response(429, headers=headers)
        return httpx.Response(200, json={"data": [], "meta": {}})

    return handler


@pytest.mark.parametrize(
    ("retry_after", "expected"),
    [
        ("inf", 2),
        ("-inf", 2),
        ("nan", 2),
        ("-5", 0),
        ("3600", 60),
        ("60", 60),
        ("1.5", 1.5),
        ("0", 0),
        ("Wed, 21 Oct 2026 07:28:00 GMT", 2),
        ("soon", 2),
        ("", 2),
    ],
)
def test_retry_after_is_finite_and_clamped(retry_after, expected):
    api, sleeps = make_api(_flaky(retry_after))
    assert list(api.iter_observations()) == []
    assert sleeps == [expected]


@pytest.mark.parametrize("body", [b"<html>nope</html>", b"", b"[1, 2]", b'"text"', b"null", b"7"])
def test_non_object_2xx_body_raises_without_the_body(body):
    def handler(request):
        return httpx.Response(200, content=body, headers={"Content-Type": "application/json"})

    api, sleeps = make_api(handler)
    with pytest.raises(LangfuseError) as exc:
        list(api.iter_observations())
    message = str(exc.value)
    assert "/api/public/v2/observations" in message
    assert "unexpected response" in message
    assert "nope" not in message
    assert sleeps == []


@pytest.mark.parametrize(
    "error",
    [httpx.DecodingError("bad gzip pk sk"), httpx.TooManyRedirects("loop pk sk")],
)
def test_other_httpx_errors_become_langfuse_error(error):
    calls = []

    def handler(request):
        calls.append(request)
        raise error

    api, sleeps = make_api(handler)
    with pytest.raises(LangfuseError) as exc:
        list(api.iter_observations())
    assert len(calls) == 1
    assert sleeps == []
    assert type(error).__name__ in str(exc.value)
    assert "pk" not in str(exc.value).replace("path", "")
    assert exc.value.__cause__ is None


def test_transport_error_is_retried_then_succeeds():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            raise httpx.ConnectError("boom pk:sk")
        return httpx.Response(200, json={"data": [{"id": "a"}], "meta": {}})

    api, sleeps = make_api(handler)
    assert [r["id"] for r in api.iter_observations()] == ["a"]
    assert len(calls) == 3
    assert sleeps == [2, 4]


def test_transport_error_exhausts_attempts_without_leaking():
    def handler(request):
        raise httpx.ConnectError(f"boom {request.headers['Authorization']}")

    api, sleeps = make_api(handler)
    with pytest.raises(LangfuseError) as exc:
        list(api.iter_observations())
    assert "ConnectError" in str(exc.value)
    assert base64.b64encode(b"pk:sk").decode() not in str(exc.value)
    assert sleeps == [2, 4, 8, 16]


def test_from_env_strips_keys_and_host():
    api = LangfuseAPI.from_env(
        {
            "LANGFUSE_PUBLIC_KEY": " pk \n",
            "LANGFUSE_SECRET_KEY": "\tsk ",
            "LANGFUSE_HOST": "  https://eu.example/  ",
        }
    )
    assert api is not None
    assert api._host == "https://eu.example"
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"data": [{"id": "p1"}]})

    api2 = LangfuseAPI.from_env(
        {"LANGFUSE_PUBLIC_KEY": " pk ", "LANGFUSE_SECRET_KEY": " sk "},
        transport=httpx.MockTransport(handler),
    )
    assert api2 is not None
    assert api2.project_id() == "p1"
    expected = "Basic " + base64.b64encode(b"pk:sk").decode()
    assert seen[0].headers["Authorization"] == expected


@pytest.mark.parametrize(
    "env",
    [
        {"LANGFUSE_PUBLIC_KEY": "   ", "LANGFUSE_SECRET_KEY": "sk"},
        {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "\n\t "},
        {"LANGFUSE_PUBLIC_KEY": " ", "LANGFUSE_SECRET_KEY": " "},
    ],
)
def test_from_env_whitespace_only_keys_are_missing(env):
    assert LangfuseAPI.from_env(env) is None


def test_from_env_blank_host_falls_back_to_cloud():
    api = LangfuseAPI.from_env(
        {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk", "LANGFUSE_HOST": "  "}
    )
    assert api is not None
    assert api._host == "https://cloud.langfuse.com"


def test_repeated_cursor_raises_instead_of_looping():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) > 10:
            raise AssertionError("the client kept following the same cursor")
        return httpx.Response(200, json={"data": [{"id": "a"}], "meta": {"cursor": "same"}})

    api, _ = make_api(handler)
    with pytest.raises(LangfuseError) as exc:
        list(api.iter_observations())
    assert "cursor" in str(exc.value)
    assert len(calls) == 2  # first page, then the page that repeated the cursor


def test_project_id_empty_list_raises():
    api, _ = make_api(lambda request: httpx.Response(200, json={"data": []}))
    with pytest.raises(LangfuseError):
        api.project_id()


@pytest.mark.parametrize("first", [{}, {"name": "demo"}, {"id": None}, {"id": ""}, {"id": 7}, "x"])
def test_project_id_without_usable_id_raises(first):
    api, _ = make_api(lambda request: httpx.Response(200, json={"data": [first]}))
    with pytest.raises(LangfuseError):
        api.project_id()


def test_close_and_context_manager():
    api, _ = make_api(lambda request: httpx.Response(200, json={"data": []}))
    with api as same:
        assert same is api
    assert api._client.is_closed
