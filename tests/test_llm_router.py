import pytest
from unittest.mock import patch, MagicMock
from src.query_router.models.llm_router import LLMRouter, get_llm_router_from_config
import os
from src.query_router.core.config import load_config

@pytest.fixture
def router_config():
    return {
        "provider": "openai",
        "model": "gpt-4o",
        "api_key": "fake-key",
        "temperature": 0.7,
        "max_tokens": 128,
        "base_url": None,
        "extra_args": {},
    }

@patch("src.query_router.models.llm_router.litellm.completion")
def test_route_valid_json(mock_completion, router_config):
    mock_completion.return_value = {
        "choices": [
            {"message": {"content": '{"route": "simple"}'}}
        ]
    }
    router = LLMRouter(router_config)
    result = router.route("What is the capital of France?")
    assert result["route"] == "simple"
    assert result["raw"]["route"] == "simple"
    mock_completion.assert_called_once()

@patch("src.query_router.models.llm_router.litellm.completion")
def test_route_malformed_json(mock_completion, router_config):
    mock_completion.return_value = {
        "choices": [
            {"message": {"content": 'not a json'}}
        ]
    }
    router = LLMRouter(router_config)
    result = router.route("What is the capital of France?")
    assert result["route"] == "semantic"
    assert result["error"] == "json_parse_error"
    assert "raw" in result
    mock_completion.assert_called_once()

@patch("src.query_router.models.llm_router.litellm.completion")
def test_route_api_error(mock_completion, router_config):
    mock_completion.side_effect = Exception("API error!")
    router = LLMRouter(router_config)
    result = router.route("What is the capital of France?")
    assert result["route"] == "semantic"
    assert "API error!" in result["error"]
    mock_completion.assert_called_once()

def test_get_llm_router_from_config(router_config):
    router = get_llm_router_from_config(router_config)
    assert isinstance(router, LLMRouter)
    assert router.model == "gpt-4o"

def test_build_prompt(router_config):
    router = LLMRouter(router_config)
    prompt = router._build_prompt("What is the capital of France?", history=["Hi", "How are you?"])
    assert prompt[0]["role"] == "system"
    assert any("Hi" in m["content"] for m in prompt)
    assert any("How are you?" in m["content"] for m in prompt)
    assert prompt[-1]["content"] == "What is the capital of France?"

@pytest.mark.integration
def test_llm_router_real_openai():
    # Skip this test in CI environments
    if os.environ.get("CI") == "true":
        pytest.skip("Skipping LLMRouter integration test in CI environment.")
    config = load_config()["models"]["llm_router"]
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        config = dict(config)  # make a copy to avoid mutating global config
        config["api_key"] = api_key
    elif not config.get("api_key"):
        pytest.skip("No API key set in config or environment for integration test")
    router = LLMRouter(config)
    result = router.route("What is the capital of France?")
    print("Integration result:", result)    
    assert "route" in result
    assert result["route"] in ["simple", "semantic", "agent"]
