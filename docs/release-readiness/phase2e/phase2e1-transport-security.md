# Phase 2E.1 Queue API transport

No production endpoint was called. No credential is recorded here.

## Path

| Hop | Where it runs | What it carries | Transport found |
| --- | --- | --- | --- |
| Desktop, Android, extension to Node | public internet to Cloud Run | user HMAC bearer or a session cookie | Cloud Run HTTPS. UNVERIFIED against the live revision |
| Node to Python | public internet to Cloud Run | user bearer, or a 600-second token Node mints | `BUDDY_API_BASE`. UNVERIFIED live scheme |
| Node meeting publisher to Queue API | `MeetingExtension/queue.ts` `fetch` | service bearer and HMAC of `timestamp.body` | The workstation file `cloud-run-env.yaml` (gitignored, not in the index) sets `QUEUE_API_BASE_URL` to `http` and a public address. Python `.env` has the same class of URL and `QUEUE_PROVIDER=redis`, so the worker does not call it |
| Python `QueueApiPublisher` to Queue API | only when `QUEUE_PROVIDER=queue_api` | same service bearer and HMAC | Same URL class if that provider is turned on |
| Caddy to `queue-api:8080` | compose network | the request after TLS termination, when TLS exists | HTTP on the Docker network. The hostname `queue-api` is not public |
| Queue API to Redis | compose network | queue payload | `redis://` to `redis:6379/0`, password in the URL, not on the public internet |
| Node goal publisher | to `GOAL_REDIS_URL` | goal event | Direct Redis, not the Queue API. No `MAXLEN` after this change |

HMAC shows the body was signed. It does not hide the bearer, the signature, or the body. A public `http://` URL is a credential leak.

## What Caddy is actually configured to do

`deploy/aws/Caddyfile` uses `{$QUEUE_API_DOMAIN}` as the site address and proxies `/internal/events` and `/health/*` to `queue-api:8080`. Compose publishes ports 80 and 443.

The gitignored `.env.aws.example` on this workstation had `QUEUE_API_DOMAIN=:80`. A site address of `:80` is HTTP on every interface. It does not request a certificate. Caddy being installed is not HTTPS.

That example file is matched by `.gitignore` rule `.env.*`, so the local edit that sets `QUEUE_API_DOMAIN=queue-api.example.com` will not be in git. The value operators must set is a DNS name they control. This task does not know that name and does not invent a production one.

`httpx` follows redirects only when asked. The publisher now sets `follow_redirects=False` and `verify=True`. Node `fetch` defaults to following redirects, which can resend a POST and, depending on the target, the `Authorization` header. `queueRequestInit` sets `redirect: "error"`.

## Code behavior

`services/queue/transport_security.py` and `MeetingExtension/queueTransport.ts`:

- `https://` is allowed. Certificate checks stay on.
- `http://` is allowed for `localhost`, `127.0.0.1`, `::1`, `host.docker.internal`, single-label names such as `queue-api`, and these networks only: `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `127.0.0.0/8`, `169.254.0.0/16`, IPv6 `::1`, `fe80::/10`, `fc00::/7`.
- Any other `http://` host, including a public address, is rejected.
- Userinfo in the URL is rejected.
- The error text does not include the credential or the raw URL.

Python applies this when `QUEUE_PROVIDER=queue_api`, and again inside `QueueApiPublisher.publish` before a client is opened. A process with `QUEUE_PROVIDER=redis` still starts if an unused public HTTP URL is present, which is the current local worker. The moment something publishes through the Queue API client, the public HTTP URL fails closed.

Node applies it in `publishMeetingEvent` before `fetch`.

Queue API authentication itself is unchanged: service bearer compare, and HMAC over `timestamp + "." + body` when the secret is set. A user bearer does not pass. Tests cover a bad HMAC and a user bearer with a stand-in token, not a live secret.

## Operator prerequisite

Staging is blocked until:

1. A DNS name points at the staging Queue API.
2. `QUEUE_API_DOMAIN` is that name, so Caddy can serve HTTPS.
3. Node and Python `QUEUE_API_BASE_URL` are `https://` plus that name.
4. The service token and HMAC secret are new staging values, not the example placeholders.

The current workstation URL must not be copied.
