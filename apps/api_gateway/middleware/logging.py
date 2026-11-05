from fastapi import Request
import time

from apps.api_gateway.config.setting import settings


async def log_requests(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time

    if settings.ENABLE_REQUEST_LOGS:
        status_code = getattr(response, "status_code", 0)
        print(f"{request.method} {request.url.path} {status_code} {process_time:.3f}s")

    return response
