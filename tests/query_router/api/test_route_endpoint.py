"""
Tests for the /route API endpoint.
"""
import uuid
from unittest.mock import patch, AsyncMock

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

API_ROUTE = "/api/v1/route"  # Endpoint path


def test_route_valid_query():
    """Test routing a simple valid query."""
    response = client.post(
        API_ROUTE, 
        json={"query": "What is the capital of France?"}
    )
    
    # Verify the response
    assert response.status_code == 200
    data = response.json()
    assert "route" in data
    assert "confidence" in data
    assert "trace_id" in data
    assert data["route"] in ["simple", "semantic", "agent"]
    assert 0.0 <= data["confidence"] <= 1.0


def test_route_with_structured_history():
    """Test routing a query with properly formatted conversation history."""
    # Create a conversation history with proper format
    history = [
        {"user": "What are the best places to visit in France?"},
        {"assistant": "Paris, Nice, and Bordeaux are excellent choices in France."},
        {"user": "Tell me about Paris"},
        {"assistant": "Paris is the capital of France and known for the Eiffel Tower, art, and cuisine."}
    ]
    
    # Make the request
    response = client.post(
        API_ROUTE, 
        json={"query": "What's the weather like there?", "history": history}
    )
    
    # Verify the response
    assert response.status_code == 200
    data = response.json()
    assert "route" in data
    assert "confidence" in data
    assert "trace_id" in data
    assert data["route"] in ["simple", "semantic", "agent"]
    assert 0.0 <= data["confidence"] <= 1.0


def test_route_with_empty_history():
    """Test routing with an empty history list."""
    response = client.post(
        API_ROUTE, 
        json={"query": "What is the capital of France?", "history": []}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "route" in data
    assert "confidence" in data
    assert "trace_id" in data


def test_route_without_history():
    """Test routing without including history field."""
    response = client.post(
        API_ROUTE, 
        json={"query": "What is the capital of France?"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "route" in data
    assert "confidence" in data
    assert "trace_id" in data


def test_trace_id_format():
    """Test that the trace_id is a valid UUID."""
    response = client.post(
        API_ROUTE, 
        json={"query": "What is the capital of France?"}
    )
    
    assert response.status_code == 200
    data = response.json()
    trace_id = data["trace_id"]
    
    # Verify that the trace_id is a valid UUID
    try:
        uuid_obj = uuid.UUID(trace_id)
        assert str(uuid_obj) == trace_id
    except ValueError:
        assert False, f"trace_id '{trace_id}' is not a valid UUID"


def test_empty_query_validation():
    """Test validation for empty queries."""
    response = client.post(
        API_ROUTE, 
        json={"query": ""}
    )
    
    # The updated service should handle this case with a 400 error
    assert response.status_code in [400, 422]  # Either validation or business logic error
    
    error_data = response.json()
    assert "detail" in error_data


def test_invalid_query_type():
    """Test validation for non-string queries."""
    response = client.post(
        API_ROUTE, 
        json={"query": 12345}  # Number instead of string
    )
    
    # Should fail validation
    assert response.status_code == 422
    error_data = response.json()
    assert "detail" in error_data


def test_invalid_history_format():
    """Test validation for incorrectly formatted history."""
    response = client.post(
        API_ROUTE, 
        json={"query": "What's the weather like?", "history": "This should be a list"}
    )
    
    # Should fail validation
    assert response.status_code == 422
    error_data = response.json()
    assert "detail" in error_data


def test_missing_query():
    """Test request without a query field."""
    response = client.post(
        API_ROUTE, 
        json={"history": []}
    )
    
    # Should fail validation
    assert response.status_code == 422
    error_data = response.json()
    assert "detail" in error_data


@patch('query_router.services.router_service.RouterService.route_query')
def test_blocked_query(mock_route_query):
    """Test handling of blocked queries."""
    # Mock the service to return a blocked result
    mock_route_query.return_value = {
        "route": "blocked",
        "confidence": 1.0,
        "reason": "Query contains unsafe content",
        "trace_id": str(uuid.uuid4())
    }
    
    response = client.post(
        API_ROUTE, 
        json={"query": "This is a potentially unsafe query"}
    )
    
    # Should return a 400 Bad Request
    assert response.status_code == 400
    error_data = response.json()
    assert "detail" in error_data
    assert "unsafe content" in error_data["detail"]


@patch('query_router.services.router_service.RouterService.route_query')
def test_error_handling(mock_route_query):
    """Test proper handling of service errors."""
    # Mock the service to return an error result
    mock_route_query.return_value = {
        "route": "error",
        "confidence": 0.0,
        "error": "Classification service unavailable",
        "trace_id": str(uuid.uuid4())
    }
    
    response = client.post(
        API_ROUTE, 
        json={"query": "This should trigger an error"}
    )
    
    # Should return a 500 Internal Server Error
    assert response.status_code == 500
    error_data = response.json()
    assert "detail" in error_data
    assert "Classification service unavailable" in error_data["detail"]


@patch('query_router.services.router_service.RouterService.route_query')
def test_service_exception(mock_route_query):
    """Test handling of exceptions thrown by the service."""
    # Mock the service to raise an exception
    mock_route_query.side_effect = Exception("Unexpected service error")
    
    response = client.post(
        API_ROUTE, 
        json={"query": "This should trigger an exception"}
    )
    
    # Should return a 500 Internal Server Error
    assert response.status_code == 500
    error_data = response.json()
    assert "detail" in error_data
    assert "Failed to process query" in error_data["detail"] 