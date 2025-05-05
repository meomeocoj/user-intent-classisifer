"""
API router and endpoint definitions.
"""
from uuid import uuid4
import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from src.query_router.api.models import RouteRequest, RouteResponse
from src.query_router.core.logging import get_logger
from src.query_router.services.router_service import RouterService

# Initialize router
router = APIRouter()

# Initialize logger
logger = get_logger(__name__)

# Instantiate RouterService once (could be improved with dependency injection)
router_service = RouterService()

@router.post("/route", response_model=RouteResponse)
async def route_query(request: RouteRequest, req: Request) -> RouteResponse:
    """
    Classify an incoming query and determine its routing using RouterService.
    
    Args:
        request: The RouteRequest containing query and optional history
        req: The FastAPI Request object
        
    Returns:
        RouteResponse with the classification results
        
    Raises:
        HTTPException: When processing fails or input is invalid
    """
    trace_id = str(uuid4())
    start_time = time.time()
    logger.info(
        "route_request_received", 
        trace_id=trace_id, 
        query=request.query, 
        has_history=request.history is not None,
        client_ip=req.client.host if req.client else "unknown"
    )
    
    try:
        # Use RouterService for all routing logic
        result, timing = await router_service.route_query(request.query, request.history, trace_id=trace_id, timing_enabled=True)
        total_time = (time.time() - start_time) * 1000
        logger.info(
            "route_total_time_ms",
            trace_id=trace_id,
            total_time_ms=f"{total_time:.2f}",
            **timing
        )
        
        # Check for error or blocked routes
        if result.get("route") == "blocked":
            logger.warning(
                "query_blocked",
                trace_id=result.get("trace_id", trace_id),
                reason=result.get("reason", "Unsafe content")
            )
            raise HTTPException(
                status_code=400, 
                detail=result.get("reason", "Query contains unsafe content")
            )
            
        if result.get("route") == "error":
            logger.error(
                "query_processing_error",
                trace_id=result.get("trace_id", trace_id),
                error=result.get("error", "Unknown error")
            )
            raise HTTPException(
                status_code=500, 
                detail=result.get("error", "Failed to process query")
            )
            
        # Log successful classification
        logger.info(
            "query_classified",
            trace_id=result.get("trace_id", trace_id),
            route=result.get("route"),
            confidence=result.get("confidence"),
            classification_model=result.get("model", "default")
        )
        
        # Create and return response
        return RouteResponse(
            route=result["route"],
            confidence=result["confidence"],
            trace_id=result.get("trace_id", trace_id)
        )
        
    except ValidationError as e:
        logger.warning(
            "invalid_request_format",
            trace_id=trace_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=422,
            detail="Invalid request format"
        ) from e
        
    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
        
    except Exception as e:
        logger.exception(
            "route_request_failed",
            trace_id=trace_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to process query"
        ) from e 