from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import httpx


def load_payload(args: argparse.Namespace) -> dict:
    if args.file:
        payload = json.loads(Path(args.file).read_text(encoding="utf-8"))
    elif args.payload:
        payload = json.loads(args.payload)
    else:
        payload = {
            "job_id": args.job_id,
            "user_id": args.user_id,
            "space_id": args.space_id,
            "file_path": args.file_path,
            "filename": args.filename or Path(args.file_path).name,
            "content_type": args.content_type,
            "status": args.status,
        }
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Send a local Pub/Sub push envelope to a worker endpoint.")
    parser.add_argument("--endpoint", required=True, help="Worker endpoint, for example http://localhost:8080/pubsub/speech")
    parser.add_argument("--payload", help="Inline JSON object payload")
    parser.add_argument("--file", help="Path to a JSON payload file")
    parser.add_argument("--job-id", help="Job id for generated speech payload")
    parser.add_argument("--user-id", default="user_1", help="User id for generated speech payload")
    parser.add_argument("--space-id", default="space_1", help="Space id for generated speech payload")
    parser.add_argument("--file-path", help="Audio file path for generated speech payload")
    parser.add_argument("--filename", help="Audio filename for generated speech payload")
    parser.add_argument("--content-type", default="audio/m4a", help="Audio content type for generated speech payload")
    parser.add_argument("--status", default="queued", help="Job status for generated speech payload")
    parser.add_argument("--event-type", default="test.requested")
    parser.add_argument("--message-id", default="local-test-message")
    args = parser.parse_args()

    explicit_payload_count = int(bool(args.payload)) + int(bool(args.file))
    if explicit_payload_count > 1:
        parser.error("Provide only one of --payload or --file")
    if explicit_payload_count == 0 and (not args.job_id or not args.file_path):
        parser.error("Provide --payload, --file, or both --job-id and --file-path")

    payload = load_payload(args)
    encoded = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    envelope = {
        "message": {
            "data": encoded,
            "messageId": args.message_id,
            "attributes": {"event_type": args.event_type, "source": "local"},
        },
        "subscription": "local-test-subscription",
        "deliveryAttempt": 1,
    }
    response = httpx.post(args.endpoint, json=envelope, timeout=60)
    print(f"{response.status_code} {response.text}")
    return 0 if response.status_code < 500 else 1


if __name__ == "__main__":
    sys.exit(main())
