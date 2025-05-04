"""
Query Router Service - Main Application
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.query_router.api.router import router as api_router
from src.query_router.core.config import load_config
from src.query_router.core.logging import setup_logging
from src.query_router.core.exceptions import AppException

# Load configuration
config = load_config()

# Setup logging
logger = setup_logging()

# Create FastAPI app
app = FastAPI(
    title=config["app"]["name"],
    version=config["app"]["version"],
    description="Query classification and routing service",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix="/api/v1")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

# @app.get("/test-error/{error_type}")
# async def test_error(error_type: str):
#     """Test error endpoint."""
#     if error_type == "auth":
#         from src.query_router.core.exceptions import AuthenticationException
#         raise AuthenticationException("Test: Not authenticated", details={"user": "anonymous"})
#     elif error_type == "validation":
#         from src.query_router.core.exceptions import ValidationException
#         raise ValidationException("Test: Invalid input", details={"field": "query"})
#     elif error_type == "notfound":
#         from src.query_router.core.exceptions import ResourceNotFoundException
#         raise ResourceNotFoundException("Test: Resource missing", details={"resource": "item"})
#     elif error_type == "external":
#         from src.query_router.core.exceptions import ExternalServiceException
#         raise ExternalServiceException("Test: External service down", details={"service": "llm"})
#     elif error_type == "app":
#         from src.query_router.core.exceptions import AppException
#         raise AppException("Test: Generic app error", status_code=418, error_type="Teapot", details={"fun": "true"})
#     elif error_type == "generic":
#         raise Exception("Test: Unhandled generic exception")
#     else:
#         return {"message": "No error triggered"}

@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    logger.error(
        "app_exception",
        error_type=exc.error_type,
        message=exc.message,
        details=exc.details,
        request_id=getattr(request.state, "trace_id", None),
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_type": exc.error_type,
            "message": exc.message,
            "request_id": getattr(request.state, "trace_id", None),
            "details": exc.details,
        },
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception(
        "unhandled_exception",
        error=str(exc),
        request_id=getattr(request.state, "trace_id", None),
    )
    return JSONResponse(
        status_code=500,
        content={
            "error_type": "InternalServerError",
            "message": "An unexpected error occurred.",
            "request_id": getattr(request.state, "trace_id", None),
        },
    ) 