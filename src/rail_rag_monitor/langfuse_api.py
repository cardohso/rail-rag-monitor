"""Minimal Langfuse public API (v2 observations) client.

This is the only module that talks to Langfuse. Credentials are never included
in exception messages or log output.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping

import httpx

MAX_ATTEMPTS = 5
PAGE_LIMIT = 1000


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
    def from_env(cls, env: Mapping[str, str]) -> LangfuseAPI | None:
        public_key = env.get("LANGFUSE_PUBLIC_KEY", "")
        secret_key = env.get("LANGFUSE_SECRET_KEY", "")
        if not public_key or not secret_key:
            return None
        host = env.get("LANGFUSE_HOST") or "https://cloud.langfuse.com"
        return cls(public_key, secret_key, host)

    def _get(self, path: str, params: Mapping[str, str | int] | None = None) -> dict:
        url = f"{self._host}{path}"
        last = "no response"
        for attempt in range(MAX_ATTEMPTS):
            retry_after: str | None = None
            try:
                response = self._client.get(url, params=params)
            except httpx.TransportError as exc:
                last = type(exc).__name__
            else:
                status = response.status_code
                if status == 429 or status >= 500:
                    last = f"HTTP {status}"
                    retry_after = response.headers.get("Retry-After")
                elif status >= 400:
                    raise LangfuseError(f"Langfuse request to {path} failed with HTTP {status}")
                else:
                    return response.json()
            if attempt < MAX_ATTEMPTS - 1:
                self._sleep(_delay(retry_after, attempt))
        raise LangfuseError(
            f"Langfuse request to {path} failed after {MAX_ATTEMPTS} attempts ({last})"
        )

    def iter_observations(self, **params: str | int) -> Iterator[dict]:
        query: dict[str, str | int] = {**params, "limit": PAGE_LIMIT}
        while True:
            body = self._get("/api/public/v2/observations", query)
            yield from body.get("data", [])
            cursor = (body.get("meta") or {}).get("cursor")
            if not cursor:
                return
            query["cursor"] = cursor

    def project_id(self) -> str:
        body = self._get("/api/public/projects")
        data = body.get("data") or []
        if not data:
            raise LangfuseError("Langfuse returned no projects for these credentials")
        return data[0]["id"]


def _delay(retry_after: str | None, attempt: int) -> float:
    if retry_after is not None:
        try:
            value = float(retry_after)
        except ValueError:
            pass
        else:
            return int(value) if value.is_integer() else value
    return 2 * 2**attempt
