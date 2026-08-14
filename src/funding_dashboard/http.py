from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests
import requests_cache
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

LOGGER = logging.getLogger(__name__)
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
SENSITIVE_QUERY_KEYS = {
    "api_key",
    "apikey",
    "access_token",
    "token",
    "secret",
    "password",
}
SENSITIVE_TEXT_PATTERN = re.compile(
    r"(?i)((?:api[_-]?key|access[_-]?token|token|secret|password)=)[^&\s]+"
)


def redact_url(url: str) -> str:
    """Redact credentials embedded in URL query strings before logging."""
    try:
        parsed = urlsplit(url)
        query = [
            (key, "REDACTED" if key.lower() in SENSITIVE_QUERY_KEYS else value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        ]
        return urlunsplit(
            (parsed.scheme, parsed.netloc, parsed.path, urlencode(query, doseq=True), parsed.fragment)
        )
    except Exception:
        return SENSITIVE_TEXT_PATTERN.sub(r"\1REDACTED", url)


def redact_text(value: object) -> str:
    return SENSITIVE_TEXT_PATTERN.sub(r"\1REDACTED", str(value))


def _cache_response_filter(response: requests.Response) -> bool:
    """Do not persist responses whose prepared URL contains a secret query value."""
    request = getattr(response, "request", None)
    url = str(getattr(request, "url", ""))
    if not url:
        return True
    try:
        return not any(
            key.lower() in SENSITIVE_QUERY_KEYS
            for key, _ in parse_qsl(urlsplit(url).query, keep_blank_values=True)
        )
    except Exception:
        return not bool(SENSITIVE_TEXT_PATTERN.search(url))


class HttpClient:
    """Polite HTTP client for official public APIs."""

    def __init__(self, cache_dir: Path, cfg: dict):
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = int(os.getenv("HTTP_TIMEOUT_SECONDS", cfg.get("timeout_seconds", 30)))
        self.min_interval = float(cfg.get("min_seconds_between_requests", 0.15))
        self.max_retries = int(cfg.get("max_retries", 5))
        self.backoff_seconds = float(cfg.get("backoff_seconds", 1.0))
        self._last_request = 0.0
        self.session = requests_cache.CachedSession(
            cache_name=str(cache_dir / "http_cache"),
            backend="sqlite",
            allowable_methods=("GET",),
            stale_if_error=True,
            filter_fn=_cache_response_filter,
        )
        user_agent = os.getenv("SEC_USER_AGENT", "FundingDashboard/2.1 research@example.com")
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept": "application/json,text/csv,text/html;q=0.9,*/*;q=0.8",
        })

    def _pace(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def _request_once(
        self,
        url: str,
        *,
        params: dict[str, Any] | None,
        expire_after: int,
    ) -> requests.Response:
        self._pace()
        started = time.monotonic()
        response = self.session.get(url, params=params, timeout=self.timeout, expire_after=expire_after)
        self._last_request = time.monotonic()
        elapsed = time.monotonic() - started
        LOGGER.info(
            "HTTP GET status=%s cached=%s elapsed_seconds=%.3f url=%s",
            response.status_code,
            bool(getattr(response, "from_cache", False)),
            elapsed,
            redact_url(response.url),
        )
        if response.status_code == 429:
            retry_after = float(response.headers.get("Retry-After", "5"))
            time.sleep(min(max(retry_after, 1), 60))
        if response.status_code in TRANSIENT_STATUS_CODES:
            response.raise_for_status()
        response.raise_for_status()
        return response

    @staticmethod
    def _log_retry(state) -> None:
        error = state.outcome.exception() if state.outcome else "unknown"
        LOGGER.warning(
            "HTTP retry attempt=%s error=%s",
            state.attempt_number,
            redact_text(error),
        )

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        expire_after: int = 3600,
    ) -> requests.Response:
        retrying = Retrying(
            retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError, requests.HTTPError)),
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(
                multiplier=self.backoff_seconds,
                min=self.backoff_seconds,
                max=16,
            ),
            reraise=True,
            before_sleep=self._log_retry,
        )
        return retrying(
            self._request_once,
            url,
            params=params,
            expire_after=expire_after,
        )
