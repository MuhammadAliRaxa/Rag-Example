from fastapi import Request
import time
from src.config.logging_config import logger

async def logging_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Path: {request.url.path} | Duration: {process_time:.4f}s | Status: {response.status_code}")
    return response
