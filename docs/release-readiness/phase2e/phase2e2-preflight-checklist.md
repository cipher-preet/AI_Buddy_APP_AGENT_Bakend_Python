# Phase 2E.2 preflight checklist

A row is `pass` only when this phase ran the check. Everything else is `not executed`.

## Executed here

| Check | Result | Evidence |
| --- | --- | --- |
| Template env file passes `--expect template` | pass | JSON `result=pass`, `mode=local`, `connected=false`, `infrastructure_changed=false`. Exit 0. |
| `--apply` is rejected | pass | Exit 2. Message says the preflight does not accept an apply flag. |
| `--expect ready` on the placeholder file | fail, as required | `ready check still has placeholder secrets`. Exit 1. |
| Public HTTP queue URL in cloud mode | fail, as required | Failures: URL rejected, https required, host must equal `QUEUE_API_DOMAIN`. The URL was not printed. Temporary file deleted. |
| `docker compose config` | pass | Exit 0. Project `kukunotes-staging`. Services: `redis`, `redis-reminder`, `redis-goals`, `staging-mongo`, `qdrant`, `queue-api`, `buddy-worker`, `python-api`, `node-api`, `caddy`. Redis, MongoDB, Qdrant, and the APIs publish no host ports. Caddy publishes `127.0.0.1:8088` and `127.0.0.1:8443` only. Worker and Queue API `REDIS_URL` host is `redis`, `QUEUE_PROVIDER=redis`, `SPEECH_VECTOR_QUEUE_TRANSPORT=list`, `WORKER_CONCURRENCY=1`. Caddy's environment is the five domain settings, not the secret file. |

## Infrastructure — not executed

| Check | Required before cloud start |
| --- | --- |
| Redis host is the staging container | `INFO server` from inside project `kukunotes-staging` only |
| MongoDB host is `staging-mongo` and database name is `buddy` | `db.getName()` in that container |
| S3 | Bucket variables empty, or a new bucket whose name contains `staging` and whose key cannot read production |
| Queue API HTTPS | `curl` against the operator hostname with certificate verification on. No name exists yet. |
| Cloud Run staging URLs | Not part of the minimum design. Templates are not deployed. |
| Docker host ports | `compose ps` shows no `6379` or `27017` |
| Volumes | `staging_redis_data`, `staging_reminder_redis_data`, `staging_goal_redis_data`, `staging_mongo_data` exist only after a start |
| Memory | `t3.medium` 4 GiB for cloud. Local mode uses the developer machine. Not measured here. |

## Security — not executed on a filled env

| Check | How |
| --- | --- |
| Node and Python HMAC parity | Phase 9 of the runbook, plus `tests/test_p0_api_auth.py` when the operator reruns tests |
| Queue service auth | Phase 9 |
| No public HTTP credentials | Preflight plus `queue_api_transport_error`. Cloud mode requires `https` |
| No public Redis or MongoDB | Compose file has no `ports` on those services. Confirm with `compose ps` after start |
| No production secrets | `--expect ready` fails while any `replace-with` value remains. The operator still has to avoid pasting a production value over a placeholder |
| No production API URL | Cloud URL host must equal `QUEUE_API_DOMAIN`, which the operator sets to the new name |

## Application — not executed

Node, Python, Queue API, and the worker were not started. Consumer groups were not created. Indexes were not created. `SPEECH_VECTOR_QUEUE_TRANSPORT=list` and `WORKER_CONCURRENCY=1` are pinned in the template and forced on the worker service. The goal worker code on disk is the recovery implementation from Phase 2C.1. The daily briefing flag is `true` with `FCM_ENABLED=false`. None of that was observed in a running process.

## Reliability — not executed

Stream trim, lease heartbeat, ack order, retry, dead letter, goal recovery, dedupe, and worker restart were proven in earlier local tests on disposable Redis. They have not been proven on the staging project, because that project does not exist yet. Do not mark them passed from those older runs.
