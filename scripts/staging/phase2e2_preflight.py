"""Validate a staging env file. This script does not connect to anything.

It refuses --apply and any other action flag. It prints variable names and
host classes, never values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.queue.transport_security import queue_api_transport_error

REQUIRED = (
    "STAGING_MODE",
    "STAGING_BIND_IP",
    "QUEUE_API_DOMAIN",
    "PYTHON_API_DOMAIN",
    "NODE_API_DOMAIN",
    "AUTH_TOKEN_SECRET",
    "SESSION_SECRET",
    "QUEUE_API_SERVICE_TOKEN",
    "QUEUE_API_HMAC_SECRET",
    "QUEUE_API_BASE_URL",
    "REDIS_URL",
    "REMINDER_REDIS_URL",
    "GOAL_REDIS_URL",
    "MINDMAP_REDIS_URL",
    "MONGO_URI",
    "MONGODB_URL",
    "SPEECH_VECTOR_QUEUE_TRANSPORT",
    "WORKER_CONCURRENCY",
    "JOB_LEASE_TTL_SECONDS",
    "JOB_LEASE_HEARTBEAT_SECONDS",
)

PLACEHOLDER_MARKS = ("replace-with", "example.invalid", "<", "changeme")
PRIVATE_REDIS_HOSTS = {"redis", "redis-reminder", "redis-goals", "localhost", "127.0.0.1"}
PRIVATE_MONGO_HOSTS = {"staging-mongo", "localhost", "127.0.0.1"}


def _parse(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def _placeholder(value: str) -> bool:
    lowered = value.lower()
    return any(mark in lowered for mark in PLACEHOLDER_MARKS) or value == ""


def main(argv: list[str]) -> int:
    if any(arg.startswith("--") and arg.split("=", 1)[0] not in {"--env", "--expect", "--json", "--help", "-h"} for arg in argv):
        print("This preflight only reads a local env file. It does not accept an apply flag.", file=sys.stderr)
        return 2
    if "--help" in argv or "-h" in argv:
        print("usage: phase2e2_preflight.py --env <file> --expect template|ready")
        return 0
    expect = "template"
    env_path = ""
    as_json = "--json" in argv
    args = [arg for arg in argv[1:] if arg != "--json"]
    index = 0
    while index < len(args):
        if args[index] == "--expect" and index + 1 < len(args):
            expect = args[index + 1]
            index += 2
            continue
        if args[index] == "--env" and index + 1 < len(args):
            env_path = args[index + 1]
            index += 2
            continue
        print("unrecognized argument", file=sys.stderr)
        return 2
    if expect not in {"template", "ready"} or not env_path:
        print("usage: phase2e2_preflight.py --env <file> --expect template|ready", file=sys.stderr)
        return 2
    path = Path(env_path)
    if not path.is_file():
        print(f"missing env file: {path.name}", file=sys.stderr)
        return 2
    values = _parse(path)
    failures: list[str] = []
    for key in REQUIRED:
        if key not in values:
            failures.append(f"missing {key}")
    mode = values.get("STAGING_MODE", "")
    if mode not in {"local", "cloud"}:
        failures.append("STAGING_MODE must be local or cloud")
    if values.get("SPEECH_VECTOR_QUEUE_TRANSPORT") != "list":
        failures.append("SPEECH_VECTOR_QUEUE_TRANSPORT must be list")
    if values.get("WORKER_CONCURRENCY") != "1":
        failures.append("WORKER_CONCURRENCY must stay 1")
    if values.get("JOB_LEASE_TTL_SECONDS") != "90" or values.get("JOB_LEASE_HEARTBEAT_SECONDS") != "20":
        failures.append("lease settings must stay 90 and 20")
    if values.get("FCM_ENABLED", "false").lower() not in {"false", "0", ""}:
        failures.append("FCM_ENABLED must stay false until a staging Firebase project exists")
    queue_url = values.get("QUEUE_API_BASE_URL", "")
    transport = queue_api_transport_error(queue_url)
    if transport:
        failures.append("QUEUE_API_BASE_URL rejected")
    if mode == "cloud":
        if not queue_url.startswith("https://"):
            failures.append("cloud mode requires an https QUEUE_API_BASE_URL")
        if urlparse(queue_url).hostname != values.get("QUEUE_API_DOMAIN"):
            failures.append("QUEUE_API_BASE_URL host must equal QUEUE_API_DOMAIN")
        if values.get("STAGING_BIND_IP") in {"127.0.0.1", "localhost", ""}:
            failures.append("cloud mode must bind the staging host address, not loopback")
        if values.get("STAGING_HTTP_PORT") != "80" or values.get("STAGING_HTTPS_PORT") != "443":
            failures.append("cloud mode must publish ports 80 and 443")
    storage = values.get("STORAGE_PROVIDER", "local")
    if storage not in {"local", "s3"}:
        failures.append("STORAGE_PROVIDER must be local or s3")
    if storage == "s3":
        bucket = values.get("S3_BUCKET") or values.get("S3_AUDIO_BUCKET") or values.get("AWS_S3_BUCKET") or ""
        if "staging" not in bucket.lower():
            failures.append("S3 mode requires a new bucket whose name contains staging")
    if mode == "local" and _host(queue_url) not in {"queue-api", "localhost", "127.0.0.1"}:
        failures.append("local mode Queue API host must be the compose service or loopback")
    for key in ("REDIS_URL", "REMINDER_REDIS_URL", "GOAL_REDIS_URL", "MINDMAP_REDIS_URL"):
        host = _host(values.get(key, ""))
        if host and host not in PRIVATE_REDIS_HOSTS:
            failures.append(f"{key} host is not a staging compose Redis name")
    if _host(values.get("REDIS_URL", "")) != "redis":
        failures.append("REDIS_URL must target the worker redis service")
    if _host(values.get("GOAL_REDIS_URL", "")) != "redis-goals":
        failures.append("GOAL_REDIS_URL must target redis-goals")
    if _host(values.get("REMINDER_REDIS_URL", "")) != "redis-reminder":
        failures.append("REMINDER_REDIS_URL must target redis-reminder")
    for key in ("MONGO_URI", "MONGODB_URL"):
        host = _host(values.get(key, ""))
        if host and host not in PRIVATE_MONGO_HOSTS:
            failures.append(f"{key} host is not staging-mongo")
    domain = values.get("QUEUE_API_DOMAIN", "")
    if domain.startswith(":") or domain.replace(".", "").isdigit():
        failures.append("QUEUE_API_DOMAIN must be a DNS name")
    secret_keys = (
        "AUTH_TOKEN_SECRET",
        "SESSION_SECRET",
        "QUEUE_API_SERVICE_TOKEN",
        "QUEUE_API_HMAC_SECRET",
        "REDIS_PASSWORD",
        "MONGO_PASSWORD",
    )
    placeholder_count = sum(1 for key in secret_keys if _placeholder(values.get(key, "")))
    if expect == "ready" and placeholder_count:
        failures.append("ready check still has placeholder secrets")
    if expect == "template" and placeholder_count == 0:
        failures.append("template check expected placeholders and found none")
    if values.get("AUTH_TOKEN_SECRET") and values.get("AUTH_TOKEN_SECRET") == values.get("QUEUE_API_SERVICE_TOKEN"):
        failures.append("user HMAC secret and Queue API token must differ")
    report = {
        "check": "phase2e2_preflight",
        "expect": expect,
        "mode": mode or "missing",
        "connected": False,
        "infrastructure_changed": False,
        "failures": failures,
        "result": "pass" if not failures else "fail",
    }
    if as_json:
        print(json.dumps(report, indent=2))
    else:
        print(f"phase2e2_preflight expect={expect} mode={mode or 'missing'} failures={len(failures)}")
        for item in failures:
            print(f"fail {item}")
        if not failures:
            print("pass")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
