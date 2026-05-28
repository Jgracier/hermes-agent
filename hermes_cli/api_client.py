"""
CLI commands for external API client enrollment.

Usage:
    hermes client enroll              # Generate enrollment QR + URL
    hermes client list                # Show enrolled clients
    hermes client revoke <client_id>  # Revoke a client's access
"""

import json
import os
import shutil
import subprocess
import sys
import time
from typing import Optional


def client_command(args):
    action = getattr(args, "client_action", None)
    if action == "enroll":
        _cmd_enroll()
    elif action == "list":
        _cmd_list()
    elif action == "revoke":
        _cmd_revoke(args.client_id)
    else:
        print("Usage: hermes client {enroll|list|revoke}")
        print("Run 'hermes client --help' for details.")


# ------------------------------------------------------------------
# enroll
# ------------------------------------------------------------------

def _cmd_enroll():
    from gateway.api_client_store import APIClientStore, DEFAULT_ENROLLMENT_TTL, get_tailscale_host

    host = _ensure_tailscale()
    if not host:
        return

    _ensure_api_server()

    api_port = os.getenv("API_SERVER_PORT", "8642")

    store = APIClientStore()
    token = store.create_enrollment()
    claim_url = f"http://{host}:{api_port}/api/enroll/claim?token={token}"
    payload = json.dumps({"type": "hermes.api.enroll", "enroll_url": claim_url})

    print()
    print("  Scan to connect:\n")
    _print_qr(payload)
    print()
    print("  Or copy this URL:")
    print(f"  {claim_url}")
    print()
    print(f"  Expires in {DEFAULT_ENROLLMENT_TTL // 60} minutes.\n")


def _ensure_tailscale() -> Optional[str]:
    from gateway.api_client_store import get_tailscale_host

    host = get_tailscale_host()
    if host:
        return host

    installed = bool(shutil.which("tailscale"))
    if installed:
        prompt = "\n  Tailscale is installed but not connected.\n  Log in to Tailscale now? [y/N]: "
    else:
        prompt = "\n  Tailscale is required to connect external clients.\n  Install and log in to Tailscale now? [y/N]: "

    try:
        answer = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    if answer != "y":
        print("\n  Skipped. Run 'hermes client enroll' again when Tailscale is set up.\n")
        return None

    if not installed:
        if not _install_tailscale():
            return None

    print("  Opening tailscale.com in your browser to log in...")
    subprocess.run(["sudo", "tailscale", "up"])

    host = get_tailscale_host()
    if host:
        print(f"  Connected: {host}\n")
    else:
        print("\n  Tailscale authentication failed. Run 'sudo tailscale up' manually.\n")
    return host


def _ensure_api_server():
    """Auto-configure and start the API server if it isn't network-accessible."""
    import secrets as _secrets
    import urllib.request
    from hermes_constants import get_hermes_home
    from hermes_cli.memory_setup import _write_env_vars
    from hermes_cli.gateway import find_gateway_pids, _graceful_restart_via_sigusr1, _gateway_run_args_for_profile

    api_host = os.getenv("API_SERVER_HOST", "127.0.0.1")
    api_key = os.getenv("API_SERVER_KEY", "")
    api_port = os.getenv("API_SERVER_PORT", "8642")
    needs_config = not api_key or api_host in ("127.0.0.1", "localhost", "::1")

    if needs_config:
        env_path = get_hermes_home() / ".env"
        writes = {"API_SERVER_ENABLED": "true", "API_SERVER_HOST": "0.0.0.0"}
        if not api_key:
            writes["API_SERVER_KEY"] = _secrets.token_hex(32)
            os.environ["API_SERVER_KEY"] = writes["API_SERVER_KEY"]
        os.environ["API_SERVER_HOST"] = "0.0.0.0"
        _write_env_vars(env_path, writes)
        print("  API server configured for external access.")
    else:
        # Already configured — check if API server is already reachable
        try:
            urllib.request.urlopen(f"http://localhost:{api_port}/health", timeout=2)
            return  # Up and configured — nothing to do
        except Exception:
            pass  # Not reachable — fall through to start/restart below

    # Restart or start the gateway to pick up config changes (or bring it up)
    pids = find_gateway_pids()
    if pids:
        print("  Restarting gateway...", end=" ", flush=True)
        _graceful_restart_via_sigusr1(pids[0], drain_timeout=15)
    else:
        print("  Starting gateway...", end=" ", flush=True)
        subprocess.Popen(
            _gateway_run_args_for_profile("default"),
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    # Wait for API server to be ready
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://localhost:{api_port}/health", timeout=2)
            print("ready.")
            return
        except Exception:
            time.sleep(1)
    print("timed out.\n  Gateway did not start in time. Run 'hermes gateway' and try again.")


def _install_tailscale() -> bool:
    print("  Installing tailscale...", end=" ", flush=True)
    if sys.platform == "darwin" and shutil.which("brew"):
        result = subprocess.run(["brew", "install", "tailscale"], capture_output=True)
    elif shutil.which("apt-get"):
        subprocess.run(["sudo", "apt-get", "update", "-qq"])
        result = subprocess.run(["sudo", "apt-get", "install", "-y", "-qq", "tailscale"])
    elif shutil.which("dnf"):
        result = subprocess.run(["sudo", "dnf", "install", "-y", "-q", "tailscale"])
    elif shutil.which("yum"):
        result = subprocess.run(["sudo", "yum", "install", "-y", "-q", "tailscale"])
    else:
        print("failed.\n  Install Tailscale manually: https://tailscale.com/download")
        return False

    if result.returncode == 0:
        print("done.")
        return True
    print("failed.\n  Install Tailscale manually: https://tailscale.com/download")
    return False


def _print_qr(data: str):
    try:
        import qrcode
        qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=1, border=1)
        qr.add_data(data)
        qr.make(fit=True)
        qr.print_ascii(invert=True, tty=False)
    except ImportError:
        print("  (Install qrcode for QR display: pip install qrcode)")


# ------------------------------------------------------------------
# list
# ------------------------------------------------------------------

def _cmd_list():
    from gateway.api_client_store import APIClientStore

    clients = APIClientStore().list_clients()
    if not clients:
        print("\n  No enrolled clients.\n")
        return

    print(f"\n  Enrolled Clients ({len(clients)}):")
    print(f"  {'ID':<22} {'Name':<22} {'Enrolled':<16} Last used")
    print(f"  {'--':<22} {'----':<22} {'--------':<16} ---------")
    for c in clients:
        print(f"  {c['client_id']:<22} {c['name']:<22} {_age(c['created_at']):<16} {_age(c['last_used']) if c['last_used'] else 'never'}")
    print()


# ------------------------------------------------------------------
# revoke
# ------------------------------------------------------------------

def _cmd_revoke(client_id: str):
    from gateway.api_client_store import APIClientStore

    ok, name = APIClientStore().revoke_client(client_id.strip())
    if ok:
        label = f"'{name}'" if name and name != "unknown" else client_id
        print(f"\n  Revoked client {label}.\n")
    else:
        print(f"\n  Client '{client_id}' not found. Run 'hermes client list' to see enrolled clients.\n")


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

def _age(ts: Optional[float]) -> str:
    if ts is None:
        return "unknown"
    delta = int(time.time() - ts)
    if delta < 60:
        return f"{delta}s ago"
    if delta < 3600:
        return f"{delta // 60}m ago"
    if delta < 86400:
        return f"{delta // 3600}h ago"
    return f"{delta // 86400}d ago"
