# Phase 2E security integration matrix

Local tests exercised the contract. No staging or production request was sent. Do not weaken a check to make a client pass.

## Token contract

| Case | Expected | Local evidence |
| --- | --- | --- |
| Valid HMAC bearer | Identity returned | `tests/test_p0_api_auth.py` and Node `Buddy/homeSecurity.test.ts` |
| Expired `exp` | Rejected | Both verifiers. Python compares `exp` with `floor(time)`. |
| Payload byte flipped | Rejected | Python test. The assertion now flips the payload's own last character. |
| Signature byte flipped | Rejected | Python test. The old assertion reused the payload's last character and was a no-op when the signature already ended with that character. That was a false failure, not an accepted bad signature. |
| Extra segment, empty token, non-string `userId` | Rejected | Python test |
| Wrong secret | Signature compare fails, result is null, route returns 401 | Verifier uses `compare_digest` on the signature string |
| Missing secret | Exception, route 401 | `AuthNotConfigured` / Node throw. Not a bypass |
| Other user's token on this user's resource | 403 or 401 | Node home tests and Python `require_matching_user` |
| Header and query tokens for different users | 401 | `require_user` in `apps/api_gateway/auth.py` |
| Session-only daily briefing | Node mints a 600-second token for the session user | `DailyBriefing.repository.test.ts` |
| Presented bearer for a different user | Not replaced with the session token | `resolveOutboundUserToken` |

Node and Python must use the same effective secret. That equality on the live services is UNVERIFIED. The Node workstation env file has `SESSION_SECRET` and does not define `AUTH_TOKEN_SECRET`.

## Authorization matrix

| Surface | Rule in the current code | How to prove it on staging |
| --- | --- | --- |
| Node Home routes | Every protected route requires a session or a bearer. Anonymous is 401. | Repeat `Buddy/homeSecurity.test.ts` against staging with a test user. |
| Spaces | Owner reads and writes. Active viewer reads. Unknown space does not reveal the resource. Client `userId` must match the token. | Two test users. User B requests user A's space. |
| Transcripts | Bound to the authenticated user, limit 50 default and 100 max, transcript text is not logged. | Node test `secures transcript retrieval`. Confirm staging logs do not contain the fixture text. |
| Python user routes | `require_user` requires a bearer or `access_token` query whose `userId` is a valid ObjectId. | Call with no header, an expired token, and a token signed by a different secret. All 401. |
| Documents, meetings, goals, briefing, speech | Handlers use the authenticated id, not a client-supplied id. | Call with user A's token and user B's id in the body. Expect 403. |
| Private download | Ownership check stays in the Python document path. | User B requests user A's object. Expect denial and no bytes. |
| Queue API | Service token and/or HMAC. User bearer is the wrong credential. | Send a user bearer and no HMAC. Expect 401. Send the staging service credential. Expect 202. |
| Internal token as a user | A Queue API service token is not an `exp`/`userId` HMAC payload, so `require_user` rejects it. | Send the service token to a Python user route. Expect 401. |

## Client compatibility

| Client | HEAD | Authorization behavior in source | Rollout note |
| --- | --- | --- | --- |
| BuddyDesktopApp renderer | `cffd49c` | `api.ts` sets `Authorization: Bearer` when a token is stored. Speech, document, and SSE helpers omit the header when the token is missing. | A logged-in current build matches the API. A request with no token fails closed. |
| AIAssistantApp | `244a32f` | `baseApi.ts` and the voice/status helpers set `Authorization: Bearer` when a token exists. | Same as desktop. Installed older APKs were not run. If an older APK omits the header, protected routes return 401. Do not remove the requirement. Ship an updated APK if a field build has no token header. |
| chromeExtension meeting recorder | `9250437` | Upload uses `Authorization: Bearer` from the stored connection. | A disconnected extension cannot upload. That is the intended failure. |
| Buddywebsite | not in this release | No matching API authorization usage. | No change. |

Session cookies still authenticate Node. Node then mints the short-lived bearer when it calls Python for daily briefing. A browser that only has a cookie can use Node. A browser that calls Python directly still needs the bearer.

## Service-to-service calls

| Call | Credential | Must not accept |
| --- | --- | --- |
| Client to Node | Session cookie or user HMAC bearer | Another user's token |
| Node to Python | Forwarded user bearer, or 600-second session token | Body `userId`, Queue API token |
| Node or Python to Queue API | Service bearer and HMAC headers | User bearer alone |
| Queue API to Redis | Compose network password | A client-supplied Redis URL |
| Worker to MongoDB and Redis | Task role credentials inside the compose network | End-user tokens |

## Staging proof list

Run these only after Stage 0 isolation in the runbook. Use two synthetic users.

1. Login as user A. Load spaces, tasks, and notes. Response contains only A's rows.
2. Replay A's request with no cookie and no bearer. Expect 401.
3. Replay with an expired token and with a token signed by a different secret. Expect 401.
4. Request A's space, transcript, goal, briefing, and private document as user B. Expect 403 or 404, and no document bytes.
5. Add user B as an active viewer. B can read the shared space and cannot rename it.
6. Force-generate a briefing from a session with no bearer. Capture the Python access log and confirm the outbound token's `exp` is about 600 seconds out and the token is not in the HTTP response body.
7. POST `/internal/events` with A's user bearer and without the service HMAC. Expect 401.
8. POST the same event with the staging service credential. Expect 202, one Redis entry, and a duplicate POST that does not push a second entry.
