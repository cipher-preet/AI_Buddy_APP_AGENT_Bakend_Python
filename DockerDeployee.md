# Buddy AI Orchestration Deployment Guide

## Step 1: Open Google Cloud SDK Shell

```bat
cd /d D:\AI_Personal_Buddy\AI_Orchestration
```

```bat
deploy.bat
```

## Step 2: Build Docker image

Use a new version every deployment.

Example:

```bash
docker build -t buddy-ai-orchestration:v3 .
```

Don't use

```
--no-cache
```

unless you changed dependencies.

## Step 3: Tag the image

```bash
docker tag buddy-ai-orchestration:v3 asia-south1-docker.pkg.dev/python-microservice-hub/buddy-backend-repo/buddy-ai-orchestration:v3
```

## Step 4: Push to Artifact Registry

```bash
docker push asia-south1-docker.pkg.dev/python-microservice-hub/buddy-backend-repo/buddy-ai-orchestration:v3
```

Wait until every layer says:

```
Pushed
```

## Deploy API

```bat
gcloud run deploy buddy-ai-api ^
  --image asia-south1-docker.pkg.dev/python-microservice-hub/buddy-backend-repo/buddy-ai-orchestration:v3 ^
  --region asia-south1 ^
  --service-account buddy-api-sa@python-microservice-hub.iam.gserviceaccount.com ^
  --env-vars-file=cloud-run-api.env ^
  --allow-unauthenticated ^
  --port 8080 ^
  --memory 1Gi ^
  --cpu 1 ^
  --timeout 300 ^
  --min-instances 0 ^
  --max-instances 5
```

## Deploy Worker

```bat
gcloud run deploy buddy-ai-worker ^
  --image asia-south1-docker.pkg.dev/python-microservice-hub/buddy-backend-repo/buddy-ai-orchestration:v3 ^
  --region asia-south1 ^
  --env-vars-file=cloud-run-worker.env ^
  --no-allow-unauthenticated ^
  --command uvicorn ^
  --args apps.api_gateway.workers.http_app:app,--host,0.0.0.0,--port,8080 ^
  --port 8080 ^
  --memory 2Gi ^
  --cpu 2 ^
  --timeout 600 ^
  --concurrency 4 ^
  --min-instances 0 ^
  --max-instances 10
```

## API Logs

```bat
gcloud run services logs read buddy-ai-api ^
  --region asia-south1 ^
  --freshness 10m ^
  --limit 100
```

## Worker Logs

```bat
gcloud run services logs read buddy-ai-worker ^
  --region asia-south1 ^
  --freshness 10m ^
  --limit 100
```

## Live API Logs

```bat
gcloud beta run services logs tail buddy-ai-api ^
  --region asia-south1
```

## Live Worker Logs

```bat
gcloud beta run services logs tail buddy-ai-worker ^
  --region asia-south1
```

## Verify deployed image

```bat
gcloud run services describe buddy-ai-api ^
  --region asia-south1 ^
  --format="value(spec.template.spec.containers[0].image)"
```

Worker:

```bat
gcloud run services describe buddy-ai-worker ^
  --region asia-south1 ^
  --format="value(spec.template.spec.containers[0].image)"
```

## Verify Pub/Sub subscriptions

```bat
gcloud pubsub subscriptions list ^
  --format="table(name,topic,pushConfig.pushEndpoint)"
```

## Publish a manual speech test

```bash
gcloud pubsub topics publish buddy-speech-jobs ^
  --message="{\"job_id\":\"test001\"}"
```

## API URL

https://buddy-ai-api-710178903619.asia-south1.run.app

## Worker URL

https://buddy-ai-worker-710178903619.asia-south1.run.app