@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM Buddy AI Orchestration - Automatic Version Deployment
REM
REM Usage:
REM   deploy.bat
REM
REM The version is generated automatically using date and time.
REM Example tag:
REM   v20260804-164530
REM ============================================================

set "PROJECT_ID=python-microservice-hub"
set "REGION=asia-south1"
set "REPOSITORY=buddy-backend-repo"

set "LOCAL_IMAGE=buddy-ai-orchestration"
set "REMOTE_IMAGE=buddy-ai-orchestration"

set "API_SERVICE=buddy-ai-api"
set "WORKER_SERVICE=buddy-ai-worker"

set "API_SERVICE_ACCOUNT=buddy-api-sa@python-microservice-hub.iam.gserviceaccount.com"

set "API_ENV_FILE=cloud-run-api.env"
set "WORKER_ENV_FILE=cloud-run-worker.env"

set "WORKER_APP=apps.api_gateway.workers.http_app:app"

REM Generate a safe unique version using PowerShell.
for /f "usebackq delims=" %%V in (`powershell -NoProfile -Command "Get-Date -Format 'yyyyMMdd-HHmmss'"`) do set "BUILD_VERSION=%%V"

if not defined BUILD_VERSION (
    echo ERROR: Could not generate deployment version.
    exit /b 1
)

set "VERSION=v%BUILD_VERSION%"
set "IMAGE_URI=%REGION%-docker.pkg.dev/%PROJECT_ID%/%REPOSITORY%/%REMOTE_IMAGE%:%VERSION%"

echo.
echo ============================================================
echo Buddy AI Orchestration Deployment
echo ============================================================
echo Project:  %PROJECT_ID%
echo Region:   %REGION%
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

if not exist "%WORKER_ENV_FILE%" (
    echo ERROR: %WORKER_ENV_FILE% was not found.
    exit /b 1
)

findstr /R /I /C:"^[ ]*PORT[ ]*=" "%API_ENV_FILE%" >nul 2>&1
if not errorlevel 1 (
    echo ERROR: Remove PORT from %API_ENV_FILE%. Cloud Run provides it.
    exit /b 1
)

findstr /R /I /C:"^[ ]*PORT[ ]*=" "%WORKER_ENV_FILE%" >nul 2>&1
if not errorlevel 1 (
    echo ERROR: Remove PORT from %WORKER_ENV_FILE%. Cloud Run provides it.
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

echo [1/8] Checking Docker...
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker Desktop is not running.
    exit /b 1
)

echo [2/8] Selecting Google Cloud project...
call gcloud config set project "%PROJECT_ID%"
if errorlevel 1 exit /b 1

echo [3/8] Configuring Artifact Registry authentication...
call gcloud auth configure-docker "%REGION%-docker.pkg.dev" --quiet
if errorlevel 1 exit /b 1

echo [4/8] Building Docker image...
docker build -t "%LOCAL_IMAGE%:%VERSION%" .
if errorlevel 1 (
    echo ERROR: Docker build failed.
    exit /b 1
)

echo [5/8] Tagging image...
docker tag "%LOCAL_IMAGE%:%VERSION%" "%IMAGE_URI%"
if errorlevel 1 exit /b 1

echo [6/8] Pushing image...
docker push "%IMAGE_URI%"
if errorlevel 1 (
    echo ERROR: Docker push failed.
    exit /b 1
)

echo [7/8] Deploying API service...
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
  --min-instances 0 ^
  --max-instances 5 ^
  --quiet

if errorlevel 1 (
    echo ERROR: API deployment failed.
    call gcloud run services logs read "%API_SERVICE%" --region "%REGION%" --limit 50
    exit /b 1
)

echo [8/8] Deploying worker service...
call gcloud run deploy "%WORKER_SERVICE%" ^
  --image "%IMAGE_URI%" ^
  --region "%REGION%" ^
  --env-vars-file "%WORKER_ENV_FILE%" ^
  --no-allow-unauthenticated ^
  --command uvicorn ^
  --args "%WORKER_APP%,--host,0.0.0.0,--port,8080" ^
  --port 8080 ^
  --memory 2Gi ^
  --cpu 2 ^
  --timeout 600 ^
  --concurrency 4 ^
  --min-instances 0 ^
  --max-instances 10 ^
  --quiet

if errorlevel 1 (
    echo ERROR: Worker deployment failed.
    call gcloud run services logs read "%WORKER_SERVICE%" --region "%REGION%" --limit 50
    exit /b 1
)

for /f "usebackq delims=" %%A in (`gcloud run services describe "%API_SERVICE%" --region "%REGION%" --format^="value(status.url)"`) do set "API_URL=%%A"
for /f "usebackq delims=" %%A in (`gcloud run services describe "%WORKER_SERVICE%" --region "%REGION%" --format^="value(status.url)"`) do set "WORKER_URL=%%A"

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
echo Worker:
echo   !WORKER_URL!
echo ============================================================
echo.
echo Future deployments require only:
echo   deploy.bat
echo.

endlocal
exit /b 0
