"""Container-only networking, readiness and Streamlit process supervision.

The host .env and the existing three infrastructure containers are untouched.
Secrets are never included in startup diagnostics.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit, urlunsplit
from urllib.request import ProxyHandler, build_opener


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOOPBACK = {"localhost", "127.0.0.1", "::1"}


def container_environment(env: dict[str, str]) -> dict[str, str]:
    """Translate only loopback DB endpoints, preserving credentials and options."""
    result = dict(env)
    if result.get("APEXLOGIC_CONTAINER_MODE") != "1":
        return result
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    dsn = result.get("APEXLOGIC_POSTGRES_DSN", "")
    if dsn:
        info = conninfo_to_dict(dsn)
        if info.get("host") in LOOPBACK:
            info.update(host=result.get("APEXLOGIC_DOCKER_POSTGRES_HOST", "postgres"),
                        port=result.get("APEXLOGIC_DOCKER_POSTGRES_PORT", "5432"))
            result["APEXLOGIC_POSTGRES_DSN"] = make_conninfo(**info)
    if result.get("APEXLOGIC_REDIS_HOST", "127.0.0.1") in LOOPBACK:
        result["APEXLOGIC_REDIS_HOST"] = result.get("APEXLOGIC_DOCKER_REDIS_HOST", "redis")
        result["APEXLOGIC_REDIS_PORT"] = result.get("APEXLOGIC_DOCKER_REDIS_PORT", "6379")
    value = result.get("APEXLOGIC_QUEUE_REDIS_URL", "")
    if value:
        parsed = urlsplit(value)
        if parsed.hostname in LOOPBACK:
            auth = parsed.netloc.rsplit("@", 1)[0] + "@" if "@" in parsed.netloc else ""
            host = result.get("APEXLOGIC_DOCKER_QUEUE_HOST", "queue")
            port = result.get("APEXLOGIC_DOCKER_QUEUE_PORT", "6379")
            result["APEXLOGIC_QUEUE_REDIS_URL"] = urlunsplit(
                parsed._replace(netloc=f"{auth}{host}:{port}"))
    return result


def _pump(source: socket.socket, destination: socket.socket) -> None:
    try:
        while data := source.recv(65536):
            destination.sendall(data)
    except OSError:
        pass
    finally:
        try:
            destination.shutdown(socket.SHUT_WR)
        except OSError:
            pass


class ProxyRelay(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, upstream):
        self.upstream = upstream
        super().__init__(address, _RelayHandler)


class _RelayHandler(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            with socket.create_connection(self.server.upstream, timeout=5) as upstream:
                upstream.settimeout(None)
                outgoing = threading.Thread(target=_pump, args=(self.request, upstream), daemon=True)
                outgoing.start()
                _pump(upstream, self.request)
                outgoing.join(timeout=1)
        except OSError:
            pass


def start_proxy_relay(env: dict[str, str]):
    """Keep HTTP_PROXY/HTTPS_PROXY=127.0.0.1:7890 usable inside a bridge container."""
    proxies = [urlsplit(env[key]) for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
               if env.get(key)]
    if not any(p.hostname in LOOPBACK and (p.port or 80) == 7890 for p in proxies):
        return None
    relay = ProxyRelay(("127.0.0.1", 7890), (
        env.get("APEXLOGIC_PROXY_UPSTREAM_HOST", "host.docker.internal"),
        int(env.get("APEXLOGIC_PROXY_UPSTREAM_PORT", "7890"))))
    threading.Thread(target=relay.serve_forever, daemon=True).start()
    print("Local proxy relay ready on 127.0.0.1:7890", flush=True)
    return relay


def check_dependencies() -> None:
    from core.postgres import connect, require_schema
    from core.background import queue_client

    require_schema()  # Read-only validation; migrations remain an explicit action.
    with connect() as db:
        if not db.execute("SELECT 1 FROM schema_migrations WHERE version=7").fetchone():
            raise RuntimeError("Worker schema migration 7 is required")
    client = queue_client()
    client.close()


def wait_for_dependencies(seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while True:
        try:
            check_dependencies()
            return
        except Exception as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError("PostgreSQL/queue readiness timed out") from None
            print(f"Waiting for PostgreSQL/queue ({type(exc).__name__})", flush=True)
            time.sleep(2)


def start_managed_workers(*, announce: bool = True) -> None:
    if os.getenv("APEXLOGIC_UI_WORKER_CONTROL_ENABLED", "1").strip().lower() in {"0", "false", "off"}:
        return
    from core.worker_control import StreamlitWorkerManager, worker_limit

    enabled = os.getenv("APEXLOGIC_UI_AUTO_START_WORKERS", "1").strip().lower() not in {"0", "false", "off"}
    try:
        count = int(os.getenv("APEXLOGIC_UI_INITIAL_WORKERS", "2")) if enabled else 0
    except ValueError:
        count = 2 if enabled else 0
    state = StreamlitWorkerManager().reconcile(min(max(count, 0), worker_limit()))
    if announce:
        print(f"Worker startup checked: online={state['online']}, starting={state['starting']}", flush=True)


def healthcheck() -> int:
    try:
        with build_opener(ProxyHandler({})).open("http://127.0.0.1:8501/_stcore/health", timeout=3) as response:
            return 0 if response.status == 200 and response.read().strip() == b"ok" else 1
    except Exception:
        return 1


def stop_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--healthcheck", action="store_true")
    args = parser.parse_args()
    if args.healthcheck:
        return healthcheck()
    os.environ.update(container_environment(dict(os.environ)))
    relay = start_proxy_relay(dict(os.environ))
    child = None
    stopping = threading.Event()

    def stop(signum, frame):
        stopping.set()
        if child is not None:
            stop_group(child.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        wait_for_dependencies(float(os.getenv("APEXLOGIC_CONTAINER_WAIT_SECONDS", "120")))
        start_managed_workers()
        child = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true",
        ], cwd=ROOT, start_new_session=True)
        next_worker_check = time.monotonic() + 5
        while child.poll() is None:
            if stopping.wait(0.5):
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    stop_group(child.pid, signal.SIGKILL)
                break
            if time.monotonic() >= next_worker_check:
                try:
                    start_managed_workers(announce=False)
                except Exception as exc:
                    print(f"Worker reconciliation deferred ({type(exc).__name__})", flush=True)
                next_worker_check = time.monotonic() + 5
        return 0 if stopping.is_set() else child.wait()
    finally:
        if child is not None:
            stop_group(child.pid, signal.SIGTERM)
        if relay is not None:
            relay.shutdown()
            relay.server_close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Streamlit container startup failed ({type(exc).__name__}); check dependencies/configuration.", flush=True)
        raise SystemExit(1) from None
