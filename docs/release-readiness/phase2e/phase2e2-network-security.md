# Phase 2E.2 network and security

No hostname is invented here. `example.invalid` does not resolve and is only a placeholder. The operator supplies the real names before any cloud start.

## Paths

Local mode, all on the compose network:

```
browser on the operator machine
  -> 127.0.0.1:8088 or :8443 Caddy
      -> node-api:5000
      -> python-api:8080
  -> node and python call http://queue-api:8080
  -> queue-api and buddy-worker use redis:6379
  -> node, python, and the worker use staging-mongo:27017
  -> node and the goal worker use redis-goals:6379
  -> reminders and mind maps use redis-reminder:6379
```

Cloud mode, one new instance, security group 80 and 443 from the internet, 22 from the operator address only:

```
client
  -> https://NODE_API_DOMAIN      Caddy -> node-api:5000
  -> https://PYTHON_API_DOMAIN    Caddy -> python-api:8080
  -> https://QUEUE_API_DOMAIN     Caddy -> queue-api:8080
       only /internal/events and /health/*
  -> queue-api -> redis:6379          (no host port)
  -> worker    -> redis:6379
  -> worker    -> staging-mongo:27017 (no host port)
  -> APIs      -> redis-goals, redis-reminder, staging-mongo
```

`QUEUE_API_BASE_URL` in cloud mode is `https://` plus the operator's queue hostname, so the call leaves the container, hits Caddy, and comes back. That is the certificate test. If the instance cannot hairpin to its own public name, the operator fixes the instance DNS or security group. The fallback is not cleartext HTTP on a public address. The code rejects that URL.

## What each hop requires

| Hop | Transport | Auth |
| --- | --- | --- |
| Browser to Node or Python | HTTPS in cloud mode. Loopback HTTP or the local Caddy ports in local mode. | User bearer. `AUTH_TOKEN_SECRET` or else `SESSION_SECRET`, same value on Node and Python. |
| Node or Python to Queue API | HTTPS on a public name. HTTP only for `queue-api`, loopback, or an explicit private range. TLS verification stays on. Redirects are errors. | `QUEUE_API_SERVICE_TOKEN` and `QUEUE_API_HMAC_SECRET`. A user bearer is not accepted. |
| Queue API and worker to worker Redis | Docker network only. | `requirepass`. URL forced by the compose file to `redis:6379/0`. |
| Goal and reminder clients | Docker network only. | Separate passwords. |
| Any service to MongoDB | Docker network only. | Username `staging` and the staging password, `authSource=admin`. |
| S3, only after a later approval | HTTPS to AWS. | A new key that cannot read the production bucket. Python has no custom S3 endpoint. |
| External STT, LLM, embeddings, FCM | Disabled. Empty keys. `FCM_ENABLED=false`. | Do not call paid providers during staging tests. |

## Caddy

`deploy/staging/Caddyfile` terminates TLS for three names: `QUEUE_API_DOMAIN`, `PYTHON_API_DOMAIN`, `NODE_API_DOMAIN`. The queue site returns 404 for every path except `/internal/events` and `/health/*`. Redis, MongoDB, Qdrant, and the worker have no Caddy route and no published host port.

`:80` is not a legal domain value. The preflight rejects a domain that is only a port or only digits.

Local mode binds Caddy to `127.0.0.1` and high ports so a mistaken start does not take over a machine's public 80/443. Cloud mode must set ports 80 and 443 and a non-loopback bind. Let's Encrypt needs those ports and a name that resolves to the staging instance. Do not disable certificate checks to work around a missing name.

## Firewall

On the staging instance only, after the operator approves provisioning:

- 80 and 443 from the internet, to Caddy.
- 22 from the operator's address only.
- No 6379, 27017, 6333, 5000, or 8080 on the security group. Those ports are not published by the compose file either.
- No Docker socket published.

The production security group is not changed by this design.

## Cloud Run templates

`deploy/staging/cloud-run-api.staging.env` and `deploy/staging/cloud-run-node.staging.yaml` are placeholders. They are not deploy inputs. The production scripts remain `AI_Orchestration/deploy.bat` (`buddy-ai-api`) and `Node_Backend/deploy.bat` (`buddy-node-backend`). Do not point either script at staging, and do not run them.

A future Cloud Run service would be a new name (`buddy-ai-api-staging`, `buddy-node-backend-staging`), minimum instances 0, and CPU throttling on. It stays blocked while its database URLs would have to be public.
