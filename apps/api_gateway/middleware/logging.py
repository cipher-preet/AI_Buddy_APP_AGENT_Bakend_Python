from fastapi import Request
import time

from apps.api_gateway.config.setting import settings


async def log_requests(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time

    if settings.ENABLE_REQUEST_LOGS:
        print(f"{request.method} {request.url} completed in {process_time}")

    return response
