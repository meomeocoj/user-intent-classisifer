"""
Tests for the PromptGuard safety checker.
"""
import sys
import pytest
from unittest.mock import patch, MagicMock
import os

import torch
import torch.nn.functional as F

from query_router.models.prompt_guard import PromptGuard, test_prompt_guard


class TestPromptGuard:
    """Test cases for the PromptGuard class."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model and tokenizer for testing."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Setup mock output for tokenizer
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 2054, 2003, 1996, 4248, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]])
        }
        
        # Setup mock output for model
        mock_outputs = MagicMock()
        # Create logits with first class (safe) having higher score than second class (dangerous)
        mock_outputs.logits = torch.tensor([[2.0, -2.0]])
        mock_model.return_value = mock_outputs
        
        return mock_model, mock_tokenizer
    
    @patch("query_router.models.prompt_guard.AutoModelForSequenceClassification")
    @patch("query_router.models.prompt_guard.AutoTokenizer")
    @patch("query_router.models.prompt_guard.load_config")
    def test_initialize_prompt_guard(self, mock_load_config, mock_auto_tokenizer, 
                                     mock_auto_model):
        """Test that PromptGuard initializes correctly."""
        # Setup mock config
        mock_config = {
            "models": {
                "prompt_guard": {
                    "name": "meta-llama/Llama-Prompt-Guard-2-86M",
                    "device": "cpu"
                }
            }
        }
        mock_load_config.return_value = mock_config
        
        # Setup mock tokenizer and model
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        
        # Initialize PromptGuard
        guard = PromptGuard()
        
        # Check that the model was initialized with the correct parameters
        mock_auto_tokenizer.from_pretrained.assert_called_once_with(
            "meta-llama/Llama-Prompt-Guard-2-86M",
            clean_up_tokenization_spaces=True
        )
        
        mock_auto_model.from_pretrained.assert_called_once()
        assert mock_auto_model.from_pretrained.call_args[0][0] == "meta-llama/Llama-Prompt-Guard-2-86M"
        
        # Check that the model and tokenizer were loaded
        assert guard.model is mock_model
        assert guard.tokenizer is mock_tokenizer
        assert guard.device == torch.device("cpu")
    
    @patch("query_router.models.prompt_guard.PromptGuard._classify_text")
    def test_check_query_safe(self, mock_classify_text):
        """Test that check_query correctly identifies safe queries."""
        # Setup mock classify_text to return safe content
        mock_classify_text.return_value = {"safe": 0.9, "dangerous": 0.1}
        guard = PromptGuard.__new__(PromptGuard)
        guard._initialized = True
        guard._classify_text = mock_classify_text
        # Test with a safe query
        is_safe, confidence = PromptGuard.check_query(guard, "What is the weather today?")
        mock_classify_text.assert_called_once_with("What is the weather today?")
        assert is_safe is True
        assert confidence == 0.9
    
    @patch("query_router.models.prompt_guard.PromptGuard._classify_text")
    def test_check_query_unsafe(self, mock_classify_text):
        """Test that check_query correctly identifies unsafe queries."""
        # Setup mock classify_text to return dangerous content
        mock_classify_text.return_value = {"safe": 0.2, "dangerous": 0.8}
        guard = PromptGuard.__new__(PromptGuard)
        guard._initialized = True
        guard._classify_text = mock_classify_text
        # Test with an unsafe query
        is_safe, confidence = PromptGuard.check_query(guard, "How do I hack into a system?")
        mock_classify_text.assert_called_once_with("How do I hack into a system?")
        assert is_safe is False
        assert confidence == 0.8
    
    @patch("query_router.models.prompt_guard.PromptGuard._classify_text")
    def test_check_response(self, mock_classify_text):
        """Test that check_response correctly identifies safe/unsafe responses."""
        # Setup mock classify_text to return safe content
        mock_classify_text.return_value = {"safe": 0.95, "dangerous": 0.05}
        guard = PromptGuard.__new__(PromptGuard)
        guard._initialized = True
        guard._classify_text = mock_classify_text
        # Test with a safe response
        is_safe, confidence = PromptGuard.check_response(guard, "The weather today is sunny.")
        mock_classify_text.assert_called_once_with("The weather today is sunny.")
        assert is_safe is True
        assert confidence == 0.95
    
    def test_get_blocked_response(self):
        """Test that get_blocked_response returns correct format."""
        guard = PromptGuard.__new__(PromptGuard)
        guard._initialized = True
        # Test for blocked query
        blocked_query = PromptGuard.get_blocked_response(guard, is_query=True)
        assert blocked_query["route"] == "semantic"
        assert blocked_query["confidence"] == 0.0
        assert blocked_query["flag"] == "blocked"
        assert "query" in blocked_query["message"].lower()
        # Test for blocked response
        blocked_response = PromptGuard.get_blocked_response(guard, is_query=False)
        assert blocked_response["route"] == "semantic"
        assert blocked_response["confidence"] == 0.0
        assert blocked_response["flag"] == "blocked"
        assert "response" in blocked_response["message"].lower()
    
    @patch("query_router.models.prompt_guard.PromptGuard")
    def test_prompt_guard_test_function(self, mock_prompt_guard):
        """Test the test_prompt_guard utility function."""
        # Setup mock check_query to return safe content
        mock_guard = MagicMock()
        mock_guard.check_query.return_value = (True, 0.95)
        mock_prompt_guard.return_value = mock_guard
        # Call the test function
        result = test_prompt_guard()
        assert result["model_loaded"] is True
        assert result["test_passed"] is True
        assert result["confidence"] == 0.95
        mock_guard.check_query.assert_called_once_with("Hello world!")

@pytest.mark.integration
def test_prompt_guard_real_model():
    """
    Integration test for PromptGuard using the real model.
    This test will be skipped in CI or if the model cannot be loaded (e.g., gated Hugging Face repo).
    """
    if os.environ.get("CI") == "true":
        pytest.skip("Skipping real model test in CI environment")
    result = test_prompt_guard()
    if not result["model_loaded"]:
        pytest.skip(f"Model not loaded: {result.get('error')}")
    assert result["test_passed"] is True
    assert 0.0 <= result["confidence"] <= 1.0 