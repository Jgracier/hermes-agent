"""
API Client Enrollment Store

Manages one-time enrollment tokens and per-client API keys for the
external client enrollment flow (hermes client enroll).

Security model mirrors the DM pairing system (gateway/pairing.py):
  - Enrollment tokens stored as salted SHA-256 hashes — never plaintext
  - Client API keys stored as salted SHA-256 hashes — never plaintext
  - Tokens are one-time: burned on first successful claim
  - Tokens expire after a configurable TTL (default 10 minutes)
  - File permissions: chmod 0600 on all data files
  - Atomic writes via temp-file rename

Storage: ~/.hermes/platforms/api_clients/
  enrollment-pending.json  — active one-time enrollment tokens (hashed)
  clients.json             — issued client keys (hashed) + metadata
"""

import hashlib
import hmac
import json
import os
import secrets
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from hermes_constants import get_hermes_dir
from utils import atomic_replace


def get_tailscale_host() -> Optional[str]:
    """Return the Tailscale MagicDNS hostname, or None if not available."""
    try:
        result = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        dns_name = data.get("Self", {}).get("DNSName", "")
        return dns_name.rstrip(".") if dns_name else None
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError, OSError):
        return None

ENROLLMENT_TOKEN_PREFIX = "hqe_"
CLIENT_KEY_PREFIX = "hca_"
CLIENT_ID_PREFIX = "clt_"

DEFAULT_ENROLLMENT_TTL = 600  # 10 minutes

CLIENTS_DIR = get_hermes_dir("platforms/api_clients", "api_clients")


def _secure_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        atomic_replace(tmp_path, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise



class APIClientStore:
    """
    Manages enrollment tokens and per-client API keys.

    Thread-safe via a single RLock. Designed to be instantiated once by
    APIServerAdapter and shared across async handler calls via the event loop.
    """

    def __init__(self):
        CLIENTS_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _pending_path(self) -> Path:
        return CLIENTS_DIR / "enrollment-pending.json"

    def _clients_path(self) -> Path:
        return CLIENTS_DIR / "clients.json"

    def _load_json(self, path: Path) -> dict:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_json(self, path: Path, data: dict) -> None:
        _secure_write(path, json.dumps(data, indent=2, ensure_ascii=False))

    # ------------------------------------------------------------------
    # Enrollment tokens
    # ------------------------------------------------------------------

    def create_enrollment(self, ttl_seconds: int = DEFAULT_ENROLLMENT_TTL) -> str:
        """
        Mint a one-time enrollment token.

        Returns the plaintext token — shown to the user once, stored only
        as a salted SHA-256 hash.
        """
        token = ENROLLMENT_TOKEN_PREFIX + secrets.token_urlsafe(24)
        salt = os.urandom(16)
        token_hash = hashlib.sha256(salt + token.encode()).hexdigest()
        entry_id = secrets.token_hex(8)

        with self._lock:
            pending = self._load_json(self._pending_path())
            pending = self._purge_expired(pending)
            pending[entry_id] = {
                "hash": token_hash,
                "salt": salt.hex(),
                "expires_at": time.time() + ttl_seconds,
                "created_at": time.time(),
            }
            self._save_json(self._pending_path(), pending)

        return token

    def claim_enrollment(self, token: str, client_name: str) -> Optional[Dict]:
        """
        Validate and burn an enrollment token, then mint a per-client API key.

        Returns {"api_key": "hca_...", "client_id": "clt_..."} on success,
        or None if the token is invalid, expired, or already claimed.
        """
        with self._lock:
            pending = self._load_json(self._pending_path())
            pending = self._purge_expired(pending)

            matched_id = None
            for entry_id, entry in pending.items():
                salt = bytes.fromhex(entry["salt"])
                if hmac.compare_digest(hashlib.sha256(salt + token.encode()).hexdigest(), entry["hash"]):
                    matched_id = entry_id
                    break

            if matched_id is None:
                return None

            # Burn immediately — one-time use
            del pending[matched_id]
            self._save_json(self._pending_path(), pending)

            # Mint client key
            api_key = CLIENT_KEY_PREFIX + secrets.token_urlsafe(32)
            client_id = CLIENT_ID_PREFIX + secrets.token_hex(8)
            salt = os.urandom(16)
            key_hash = hashlib.sha256(salt + api_key.encode()).hexdigest()

            clients = self._load_json(self._clients_path())
            clients[client_id] = {
                "hash": key_hash,
                "salt": salt.hex(),
                "name": (client_name or "unknown").strip()[:64],
                "created_at": time.time(),
                "last_used": None,
            }
            self._save_json(self._clients_path(), clients)

            return {"api_key": api_key, "client_id": client_id}

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def verify_client_key(self, bearer_token: str) -> bool:
        """Return True if bearer_token matches an enrolled client key."""
        if not bearer_token.startswith(CLIENT_KEY_PREFIX):
            return False

        clients = self._load_json(self._clients_path())
        for client_id, entry in clients.items():
            salt = bytes.fromhex(entry["salt"])
            if hmac.compare_digest(hashlib.sha256(salt + bearer_token.encode()).hexdigest(), entry["hash"]):
                # Touch last_used without the lock — minor race is acceptable
                entry["last_used"] = time.time()
                try:
                    self._save_json(self._clients_path(), clients)
                except OSError:
                    pass
                return True
        return False

    def has_any_clients(self) -> bool:
        return bool(self._load_json(self._clients_path()))

    # ------------------------------------------------------------------
    # List / revoke
    # ------------------------------------------------------------------

    def list_clients(self) -> List[Dict]:
        clients = self._load_json(self._clients_path())
        result = []
        for client_id, entry in clients.items():
            result.append({
                "client_id": client_id,
                "name": entry.get("name", "unknown"),
                "created_at": entry.get("created_at"),
                "last_used": entry.get("last_used"),
            })
        return sorted(result, key=lambda x: x["created_at"] or 0)

    def revoke_client(self, client_id: str):
        with self._lock:
            clients = self._load_json(self._clients_path())
            if client_id in clients:
                name = clients[client_id].get("name", "unknown")
                del clients[client_id]
                self._save_json(self._clients_path(), clients)
                return True, name
        return False, None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _purge_expired(self, pending: dict) -> dict:
        now = time.time()
        return {k: v for k, v in pending.items() if v.get("expires_at", 0) > now}
