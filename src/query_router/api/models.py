"""
Pydantic models for API requests and responses.
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class RouteRequest(BaseModel):
    """Request model for route endpoint."""
    query: str = Field(..., description="The query to classify")
    history: Optional[List[Dict[str, str]]] = Field(
        default=None, 
        description="Optional conversation history as a list of message objects"
    )


class RouteResponse(BaseModel):
    """Response model for route endpoint."""
    route: str = Field(..., description="Classified route: simple, semantic, agent, or blocked")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the classification")
    trace_id: str = Field(..., description="Unique trace ID for request tracking") 