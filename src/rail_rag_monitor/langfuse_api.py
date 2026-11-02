"""Minimal Langfuse public API (v2 observations) client.

This is the only module that talks to Langfuse. Credentials are never included
in exception messages or log output.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator, Mapping

import httpx

MAX_ATTEMPTS = 5
PAGE_LIMIT = 1000  # fixed: `iter_observations` always asks for full pages
MAX_RETRY_AFTER_S = 60.0


class LangfuseError(RuntimeError):
    """Raised when the Langfuse API cannot be queried successfully."""


class LangfuseAPI:
    def __init__(
        self,
        public_key: str,
        secret_key: str,
        host: str = "https://cloud.langfuse.com",
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._host = host.rstrip("/")
        self._sleep = sleep
        self._client = httpx.Client(
            auth=httpx.BasicAuth(public_key, secret_key),
            transport=transport,
            timeout=30.0,
        )

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str],
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> LangfuseAPI | None:
        """The client for `LANGFUSE_*` variables, or None if either key is missing or blank."""
        public_key = env.get("LANGFUSE_PUBLIC_KEY", "").strip()
        secret_key = env.get("LANGFUSE_SECRET_KEY", "").strip()
        if not public_key or not secret_key:
            return None
        host = env.get("LANGFUSE_HOST", "").strip() or "https://cloud.langfuse.com"
        return cls(public_key, secret_key, host, transport=transport, sleep=sleep)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LangfuseAPI:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _get(self, path: str, params: Mapping[str, str | int] | None = None) -> dict:
        url = f"{self._host}{path}"
        last = "no response"
        for attempt in range(MAX_ATTEMPTS):
            retry_after: str | None = None
            try:
                response = self._client.get(url, params=params)
            except httpx.TransportError as exc:
                last = type(exc).__name__
            except httpx.HTTPError as exc:  # e.g. DecodingError, TooManyRedirects: not transient
                raise LangfuseError(
                    f"Langfuse request to {path} failed: {type(exc).__name__}"
                ) from None
            else:
                status = response.status_code
                if status == 429 or status >= 500:
                    last = f"HTTP {status}"
                    retry_after = response.headers.get("Retry-After")
                elif status >= 400:
                    raise LangfuseError(f"Langfuse request to {path} failed with HTTP {status}")
                else:
                    return _object_body(response, path)
            if attempt < MAX_ATTEMPTS - 1:
                self._sleep(_delay(retry_after, attempt))
        raise LangfuseError(
            f"Langfuse request to {path} failed after {MAX_ATTEMPTS} attempts ({last})"
        )

    def iter_observations(self, **params: str | int) -> Iterator[dict]:
        """Yield every observation, following the cursor. `limit` is fixed at `PAGE_LIMIT`."""
        query: dict[str, str | int] = {**params, "limit": PAGE_LIMIT}
        seen: set[str] = set()
        while True:
            body = self._get("/api/public/v2/observations", query)
            yield from body.get("data", [])
            cursor = (body.get("meta") or {}).get("cursor")
            if not cursor:
                return
            if cursor in seen:
                raise LangfuseError("Langfuse returned a repeated page cursor")
            seen.add(cursor)
            query["cursor"] = cursor

    def project_id(self) -> str:
        body = self._get("/api/public/projects")
        data = body.get("data") or []
        if not data:
            raise LangfuseError("Langfuse returned no projects for these credentials")
        first = data[0]
        project_id = first.get("id") if isinstance(first, dict) else None
        if not isinstance(project_id, str) or not project_id:
            raise LangfuseError("Langfuse project list has no project id")
        return project_id


def _object_body(response: httpx.Response, path: str) -> dict:
    """The JSON object in a successful response. The body is never put in an error."""
    try:
        body = response.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        raise LangfuseError(f"Langfuse request to {path} returned an unexpected response")
    return body


def _delay(retry_after: str | None, attempt: int) -> float:
    """Seconds to wait: a finite numeric Retry-After clamped to 0..60, else exponential backoff.

    An HTTP-date or any other non-numeric value falls back to the backoff.
    """
    if retry_after is not None:
        try:
            value = float(retry_after)
        except ValueError:
            pass
        else:
            if math.isfinite(value):
                value = min(max(value, 0.0), MAX_RETRY_AFTER_S)
                return int(value) if value.is_integer() else value
    return 2 * 2**attempt
