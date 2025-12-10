"""
Custom middleware for the API.
"""

import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from utils.logger import logger


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all requests and responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and log details.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response
        """
        # Start timer
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url.path}")
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            logger.info(
                f"Response: {request.method} {request.url.path} "
                f"- Status: {response.status_code} - Duration: {duration:.3f}s"
            )
            
            # Add custom headers
            response.headers["X-Process-Time"] = str(duration)
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Error: {request.method} {request.url.path} "
                f"- Error: {str(e)} - Duration: {duration:.3f}s"
            )
            raise


class CORSMiddleware:
    """Custom CORS middleware with configuration."""
    
    def __init__(self, app, allowed_origins: list):
        self.app = app
        self.allowed_origins = allowed_origins
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope["headers"])
            origin = headers.get(b"origin", b"").decode()
            
            # Check if origin is allowed
            if origin in self.allowed_origins or "*" in self.allowed_origins:
                # Add CORS headers to response
                async def send_wrapper(message):
                    if message["type"] == "http.response.start":
                        headers = message.get("headers", [])
                        headers.extend([
                            (b"access-control-allow-origin", origin.encode()),
                            (b"access-control-allow-credentials", b"true"),
                            (b"access-control-allow-methods", b"GET, POST, PUT, DELETE, OPTIONS"),
                            (b"access-control-allow-headers", b"*"),
                        ])
                        message["headers"] = headers
                    await send(message)
                
                await self.app(scope, receive, send_wrapper)
            else:
                await self.app(scope, receive, send)
        else:
            await self.app(scope, receive, send)
