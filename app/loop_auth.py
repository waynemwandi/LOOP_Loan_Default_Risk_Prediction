import base64
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger("loop-risk.auth")


@dataclass
class TokenBundle:
    access_token: str
    token_type: str
    expires_at: float  # epoch seconds


class LoopAuth:
    """
    Minimal client-credentials OAuth2 helper with in-memory caching and
    thread-safe refresh. Designed for FastAPI but framework-agnostic.
    """

    def __init__(
        self,
        auth_url: str | None = None,
        client_key: str | None = None,
        client_secret: str | None = None,
        verify_tls: bool | None = None,
        early_refresh_seconds: int | None = None,
        timeout: float = 10.0,
    ):
        self.auth_url = auth_url or os.getenv("LOOP_AUTH_URL", "").strip()
        self.client_key = client_key or os.getenv("LOOP_CLIENT_KEY", "").strip()
        self.client_secret = (
            client_secret or os.getenv("LOOP_CLIENT_SECRET", "").strip()
        )
        self.verify_tls = (
            str(verify_tls).lower()
            if verify_tls is not None
            else os.getenv("LOOP_VERIFY_TLS", "true").lower()
        ) != "false"
        self.early_refresh = (
            int(early_refresh_seconds)
            if early_refresh_seconds is not None
            else int(os.getenv("LOOP_TOKEN_EARLY_REFRESH", "60"))
        )

        self.timeout = timeout

        if not self.auth_url or not self.client_key or not self.client_secret:
            raise ValueError(
                "LOOP auth config missing: ensure LOOP_AUTH_URL, LOOP_CLIENT_KEY, LOOP_CLIENT_SECRET are set"
            )

        self._lock = threading.RLock()
        self._token: Optional[TokenBundle] = None

    def _basic_header(self) -> str:
        raw = f"{self.client_key}:{self.client_secret}".encode("utf-8")
        return "Basic " + base64.b64encode(raw).decode("ascii")

    def _expired_or_missing(self) -> bool:
        if not self._token:
            return True
        now = time.time()
        return now >= (self._token.expires_at - self.early_refresh)

    def _fetch_token(self) -> TokenBundle:
        headers = {
            "Authorization": self._basic_header(),
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {"grant_type": "client_credentials"}

        logger.info("Requesting LOOP OAuth2 token from %s", self.auth_url)
        try:
            resp = requests.post(
                self.auth_url,
                headers=headers,
                data=data,
                timeout=self.timeout,
                verify=self.verify_tls,
            )
        except requests.RequestException as e:
            logger.error("Token request failed: %s", e)
            raise

        # Some sandboxes return text/html on error—log body to aid debugging
        content_type = resp.headers.get("content-type", "")
        body = resp.text
        if resp.status_code != 200:
            snippet = body[:400].replace("\n", " ")
            logger.error(
                "Token request HTTP %s. Body (truncated): %s", resp.status_code, snippet
            )
            resp.raise_for_status()

        try:
            payload = resp.json() if "json" in content_type else json.loads(body)
        except json.JSONDecodeError:
            logger.error(
                "Failed to parse token JSON. Raw body (truncated): %s", body[:400]
            )
            raise

        access = payload.get("access_token")
        ttype = payload.get("token_type", "Bearer")
        expires_in = int(payload.get("expires_in", 3600))

        if not access:
            logger.error("No access_token in response: %s", payload)
            raise RuntimeError("No access_token in LOOP OAuth2 response")

        bundle = TokenBundle(
            access_token=access, token_type=ttype, expires_at=time.time() + expires_in
        )
        logger.info(
            "Received token. Expires in %ss (early refresh %ss)",
            expires_in,
            self.early_refresh,
        )
        return bundle

    def get_token(self) -> str:
        """
        Returns a valid 'Authorization' header value: 'Bearer <token>'.
        Refreshes automatically when close to expiry.
        """
        with self._lock:
            if self._expired_or_missing():
                self._token = self._fetch_token()

            token = self._token  # satisfy type checker
            if token is None:
                raise RuntimeError("LOOP OAuth2 token unavailable after refresh")

            return f"{token.token_type} {token.access_token}"

    # Convenience for building headers for downstream LOOP API calls
    def auth_headers(self, extra: dict | None = None) -> dict:
        h = {"Authorization": self.get_token()}
        if extra:
            h.update(extra)
        return h
