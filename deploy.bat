@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM Buddy AI Orchestration - Cloud Run API Deployment
REM
REM This script deploys only the public FastAPI API to Google Cloud Run.
REM Background processing now runs on AWS with:
REM   docker compose -f docker-compose.aws.yml --env-file .env.aws up -d
REM
REM Required local files:
REM   Dockerfile
REM   requirements.txt
REM   cloud-run-api.env
REM ============================================================

set "PROJECT_ID=python-microservice-hub"
set "REGION=asia-south1"
set "REPOSITORY=buddy-backend-repo"

set "LOCAL_IMAGE=buddy-ai-orchestration"
set "REMOTE_IMAGE=buddy-ai-orchestration"
set "API_SERVICE=buddy-ai-api"
set "API_SERVICE_ACCOUNT=buddy-api-sa@python-microservice-hub.iam.gserviceaccount.com"
set "API_ENV_FILE=cloud-run-api.env"

for /f "usebackq delims=" %%V in (`powershell -NoProfile -Command "Get-Date -Format 'yyyyMMdd-HHmmss'"`) do set "BUILD_VERSION=%%V"

if not defined BUILD_VERSION (
    echo ERROR: Could not generate deployment version.
    exit /b 1
)

set "VERSION=v%BUILD_VERSION%"
set "IMAGE_URI=%REGION%-docker.pkg.dev/%PROJECT_ID%/%REPOSITORY%/%REMOTE_IMAGE%:%VERSION%"

echo.
echo ============================================================
echo Buddy AI Orchestration - Cloud Run API Deployment
echo ============================================================
echo Project:  %PROJECT_ID%
echo Region:   %REGION%
echo Service:  %API_SERVICE%
echo Version:  %VERSION%
echo Image:    %IMAGE_URI%
echo ============================================================
echo.

if not exist "Dockerfile" (
    echo ERROR: Dockerfile was not found in %CD%.
    exit /b 1
)

if not exist "requirements.txt" (
    echo ERROR: requirements.txt was not found.
    exit /b 1
)

if not exist "%API_ENV_FILE%" (
    echo ERROR: %API_ENV_FILE% was not found.
    exit /b 1
)

findstr /R /I /C:"^[ ]*PORT[ ]*=" "%API_ENV_FILE%" >nul 2>&1
if not errorlevel 1 (
    echo ERROR: Remove PORT from %API_ENV_FILE%. Cloud Run provides it.
    exit /b 1
)

findstr /R /I /C:"^[ ]*SERVICE_ROLE[ ]*=[ ]*worker" "%API_ENV_FILE%" >nul 2>&1
if not errorlevel 1 (
    echo ERROR: %API_ENV_FILE% must deploy SERVICE_ROLE=api.
    exit /b 1
)

where docker >nul 2>&1
if errorlevel 1 (
    echo ERROR: docker was not found in PATH.
    exit /b 1
)

where gcloud >nul 2>&1
if errorlevel 1 (
    echo ERROR: gcloud was not found in PATH.
    exit /b 1
)

echo [1/7] Checking Docker...
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker Desktop is not running.
    exit /b 1
)

echo [2/7] Selecting Google Cloud project...
call gcloud config set project "%PROJECT_ID%"
if errorlevel 1 exit /b 1

echo [3/7] Configuring Artifact Registry authentication...
call gcloud auth configure-docker "%REGION%-docker.pkg.dev" --quiet
if errorlevel 1 exit /b 1

echo [4/7] Building Docker image...
docker build -t "%LOCAL_IMAGE%:%VERSION%" .
if errorlevel 1 (
    echo ERROR: Docker build failed.
    exit /b 1
)

echo [5/7] Tagging image...
docker tag "%LOCAL_IMAGE%:%VERSION%" "%IMAGE_URI%"
if errorlevel 1 exit /b 1

echo [6/7] Pushing image...
docker push "%IMAGE_URI%"
if errorlevel 1 (
    echo ERROR: Docker push failed.
    exit /b 1
)

echo [7/7] Deploying Cloud Run API...
call gcloud run deploy "%API_SERVICE%" ^
  --image "%IMAGE_URI%" ^
  --region "%REGION%" ^
  --service-account "%API_SERVICE_ACCOUNT%" ^
  --env-vars-file "%API_ENV_FILE%" ^
  --allow-unauthenticated ^
  --port 8080 ^
  --memory 1Gi ^
  --cpu 1 ^
  --timeout 300 ^
  --concurrency 80 ^
  --min-instances 0 ^
  --max-instances 5 ^
  --quiet

if errorlevel 1 (
    echo ERROR: API deployment failed.
    call gcloud run services logs read "%API_SERVICE%" --region "%REGION%" --limit 50
    exit /b 1
)

for /f "usebackq delims=" %%A in (`gcloud run services describe "%API_SERVICE%" --region "%REGION%" --format^="value(status.url)"`) do set "API_URL=%%A"

echo.
echo ============================================================
echo DEPLOYMENT COMPLETED
echo ============================================================
echo Version:
echo   %VERSION%
echo.
echo Image:
echo   %IMAGE_URI%
echo.
echo API:
echo   !API_URL!
echo.
echo AWS worker and Queue API are not deployed by this script.
echo Use docker-compose.aws.yml on the AWS host.
echo ============================================================
echo.

endlocal
exit /b 0
