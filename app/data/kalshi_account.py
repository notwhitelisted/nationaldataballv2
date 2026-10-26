"""Read-only access to your Kalshi portfolio (balance, fills, positions).

Kalshi API keys are an id plus an RSA private key; every request is signed with RSA-PSS
over timestamp + method + path. This module only ever issues GET requests to
/portfolio endpoints. It cannot place, change or cancel orders.

Setup: Kalshi -> Account & security -> API keys -> create key, save the .pem file, then in .env:
    KALSHI_API_KEY_ID=...
    KALSHI_PRIVATE_KEY_PATH=C:/path/to/kalshi_private_key.pem
"""
import base64
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from ..config import KALSHI_BASE, ROOT

API_PREFIX = urlparse(KALSHI_BASE).path  # /trade-api/v2


def _key_path() -> Path | None:
    p = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")
    if not p:
        return None
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def configured() -> bool:
    path = _key_path()
    return bool(os.getenv("KALSHI_API_KEY_ID")) and path is not None and path.exists()


_private_key = None


def _sign(method: str, path: str) -> dict:
    global _private_key
    if _private_key is None:
        _private_key = serialization.load_pem_private_key(_key_path().read_bytes(), password=None)
    ts = str(int(time.time() * 1000))
    sig = _private_key.sign(
        (ts + method + API_PREFIX + path).encode(),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY": os.environ["KALSHI_API_KEY_ID"],
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        "KALSHI-ACCESS-TIMESTAMP": ts,
    }


def _get(path: str, params: dict | None = None) -> dict:
    if not configured():
        raise RuntimeError("Kalshi API key not configured (see KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH in .env)")
    with httpx.Client(timeout=30) as client:
        r = client.get(KALSHI_BASE + path, params=params or {}, headers=_sign("GET", path))
    r.raise_for_status()
    return r.json()


def _paged(path: str, key: str, params: dict | None = None) -> list[dict]:
    out, cursor = [], None
    while True:
        p = {"limit": 200, **(params or {})}
        if cursor:
            p["cursor"] = cursor
        d = _get(path, p)
        out += d.get(key, [])
        cursor = d.get("cursor")
        if not cursor:
            return out


def balance() -> dict:
    d = _get("/portfolio/balance")
    # Balances are reported in cents.
    return {"cash": d.get("balance", 0) / 100, "portfolio_value": (d.get("portfolio_value") or 0) / 100}


def fills() -> list[dict]:
    return _paged("/portfolio/fills", "fills")


def positions() -> list[dict]:
    return _paged("/portfolio/positions", "market_positions")
