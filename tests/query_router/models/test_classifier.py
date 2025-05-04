"""
Tests for the query depth classifier.
"""
import pytest

from query_router.models.classifier import QueryClassifier

@pytest.fixture
def classifier():
    """Create a QueryClassifier instance for testing."""
    return QueryClassifier()

@pytest.mark.asyncio
async def test_simple_query_classification(classifier):
    """Test classification of a simple query."""
    query = "What is the capital of France?"
    route, confidence = await classifier.classify(query)
    
    assert route == "simple"
    assert confidence >= 0.75

@pytest.mark.asyncio
async def test_semantic_query_classification(classifier):
    """Test classification of a semantic query."""
    query = "Can you summarize the main findings from the latest research papers on quantum computing?"
    route, confidence = await classifier.classify(query)
    
    assert route == "semantic"
    assert confidence >= 0.5

@pytest.mark.asyncio
async def test_agent_query_classification(classifier):
    """Test classification of an agent query."""
    query = "Help me design a research plan to study the effects of climate change on marine ecosystems."
    route, confidence = await classifier.classify(query)
    
    assert route == "agent"
    assert confidence >= 0.5

@pytest.mark.asyncio
async def test_query_with_history(classifier):
    """Test classification with conversation history."""
    query = "What about Paris?"
    history = ["Tell me about France", "France is a country in Europe"]
    route, confidence = await classifier.classify(query, history)
    
    assert route in ["simple", "semantic", "agent"]
    assert 0 <= confidence <= 1

@pytest.mark.asyncio
async def test_empty_query(classifier):
    """Test handling of empty query."""
    with pytest.raises(ValueError):
        await classifier.classify("")

@pytest.mark.asyncio
async def test_classifier_callable(classifier):
    """Test that classifier instance can be called directly."""
    query = "What time is it?"
    route, confidence = await classifier(query)
    
    assert route == "simple"
    assert confidence >= 0.75 