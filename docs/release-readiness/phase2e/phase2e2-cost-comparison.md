# Phase 2E.2 cost comparison

Retrieved 2026-11-05. Figures below are public list prices or arithmetic on those prices. They are not an invoice. Taxes, credits, and an existing free-tier balance were not queried.

## Sources

Verified from the provider page in this session:

- Cloud Run request-based, Tier 1, which includes `asia-south1`: active CPU `$0.000024` per vCPU-second, active memory `$0.0000025` per GiB-second, requests `$0.40` per million. Free tier per billing account: 180,000 vCPU-seconds, 360,000 GiB-seconds, 2 million requests. Instance-based free tier: 240,000 vCPU-seconds and 450,000 GiB-seconds, then `$0.000018` per vCPU-second and `$0.000002` per GiB-second. Source: `https://cloud.google.com/run/pricing`. Production on the same billing account may already consume the free tier. Do not treat staging as free.
- AWS T3 marketing prices are US East only. `t3.medium` there is `$0.0418` per hour. Source: `https://aws.amazon.com/ec2/instance-types/t3/`. That page is not the Mumbai price.

Mumbai prices below come from public price-list mirrors, not from an AWS bill:

- `t3.medium` Linux on-demand `ap-south-1`: `$0.0448` per hour, `$32.70` for 730 hours. Same figure on cloud-credits.org (updated 2026-10-19), trucost.cloud (row dated 2026-11-03), instances.usage.ai, and doit.com.
- `t3.small`: `$0.0224` per hour. `t3.micro`: `$0.0112` per hour. cloud-credits.org updated 2026-10-19 and the doit.com t3 family table. A Vantage snippet showed about `$0.0208`, which matches the US East `t3.small` list more closely, so it is not used for Mumbai.
- gp3: `$0.0912` per GB-month, and a public IPv4 address: `$0.005` per hour. Both are from the cloud-credits.org t3.micro Mumbai page, attributed there to the AWS Price List API on 2026-10-19. The AWS VPC pricing page was not opened in this session, so the IPv4 line stays a mirror figure.
- S3 Standard `ap-south-1`, first 50 TB: `$0.025` per GB-month. PUT/COPY/POST/LIST `$0.005` per 1,000. GET `$0.0004` per 1,000. cloud-credits.org updated 2026-10-19. The AWS S3 pricing page describes the meters and does not print the Mumbai number in the fetched text. US East GET on that page's secondary write-ups is the same `$0.0004` per 1,000.

Not priced in this session, so no number is stated: Atlas, Cloud NAT, a Serverless VPC connector, CloudWatch ingestion, ECR storage, Route 53, and the Mumbai internet-egress rate. Synthetic tests should stay small. A hosted zone the operator already pays for does not add a new zone fee.

## 1. Local-only testing

Cloud spend: `$0`.

The stack is `docker-compose.staging.yml` on the developer machine. Images are pulled once. `STORAGE_PROVIDER=local`. Provider keys in the template are empty, so the tests must use mocks. No notification provider is enabled (`FCM_ENABLED=false`).

This does not buy a public certificate or a Cloud Run test.

## 2. Cloud staging running continuously

Recommended shape, one new `t3.medium`, not the production instance:

| Line | Arithmetic | Estimate |
| --- | --- | --- |
| EC2 Linux on-demand | `730 * 0.0448` | `$32.70` |
| 30 GB gp3, kept while stopped too | `30 * 0.0912` | `$2.74` |
| Elastic IPv4 kept for stable DNS | `730 * 0.005` | `$3.65` |
| S3 | unused while `STORAGE_PROVIDER=local` | `$0` |
| Cloud Run | not in this design | `$0` |
| Paid AI | not invoked | `$0` |

About `$39` per month before tax, from the mirror rates above. The EC2 line matches the `$32.70` figure published next to `$0.0448`.

A continuous Cloud Run pair is the expensive variant and is not recommended. vmprices.com, checked 4 Nov 2026, lists `asia-south1` request-based busy-all-month at `$76.21` and instance-based at `$57.82` for 1 vCPU and 2 GiB before free tier. Two such services would dominate the bill, and they still could not reach private Redis or MongoDB.

## 3. Cloud staging only during test sessions

Assumption: 20 hours per month, instance stopped otherwise, 30 GB volume retained, Elastic IP retained so the operator DNS names stay valid.

| Line | Arithmetic | Estimate |
| --- | --- | --- |
| EC2 | `20 * 0.0448` | `$0.90` |
| 30 GB gp3 | `30 * 0.0912` | `$2.74` |
| Elastic IPv4 | `730 * 0.005` | `$3.65` |
| S3, only if a later test stores about 1 GB | `1 * 0.025` plus a few thousand requests | under `$0.10` |

About `$7.30` per month before tax. Stopping the instance does not stop the disk charge or an allocated Elastic IP. An auto-assigned public IP is released on stop and avoids the `$3.65`, and it also breaks DNS until the operator edits the records. Keep the Elastic IP if the names must survive a stop.

If Cloud Run were added later for a 4-hour request-based test of two 1 vCPU / 1 GiB services, and the free tier were already gone: CPU `14400 * 0.000024 * 2 = $0.69`, memory `14400 * 0.0000025 * 2 = $0.07`. That addition is pointless until a private data path exists.

## What was left out on purpose

No second production-sized server. No Memorystore. No Atlas. No NAT gateway. No load balancer. No always-on Cloud Run minimum instances. No paid STT, LLM, or embedding calls. Logs stay in the container json files on the staging disk until an operator asks for a log agent.

`t3.small` at `730 * 0.0224 = $16.35` continuous is cheaper and is rejected because 2 GiB is not enough for this stack. A smaller bill that OOMs the worker is not the low-cost safe option.
