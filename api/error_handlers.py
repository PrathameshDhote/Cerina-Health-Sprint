"""
Custom error handlers for the API.
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from utils.logger import logger
from state.schemas import ErrorResponse
from datetime import datetime


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """
    Handle HTTP exceptions.
    
    Args:
        request: Request that caused the exception
        exc: HTTP exception
        
    Returns:
        JSON error response
    """
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    
    error_response = ErrorResponse(
        error=str(exc.detail),  # Convert to string
        detail=None,
        thread_id=None,
        timestamp=datetime.now()
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(mode='json')  # Use Pydantic JSON serialization
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle request validation errors.
    
    Args:
        request: Request that caused the exception
        exc: Validation exception
        
    Returns:
        JSON error response
    """
    logger.error(f"Validation Error: {exc.errors()}")
    
    error_response = ErrorResponse(
        error="Validation error",
        detail=str(exc.errors()),
        thread_id=None,
        timestamp=datetime.now()
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.model_dump(mode='json')
    )


async def generic_exception_handler(request: Request, exc: Exception):
    """
    Handle all other exceptions.
    
    Args:
        request: Request that caused the exception
        exc: Exception
        
    Returns:
        JSON error response
    """
    logger.error(f"Unhandled Exception: {type(exc).__name__} - {str(exc)}")
    
    error_response = ErrorResponse(
        error="Internal server error",
        detail=str(exc),
        thread_id=None,
        timestamp=datetime.now()
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump(mode='json')
    )
