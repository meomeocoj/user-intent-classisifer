"""
Prompt Guard for safety checking of queries and responses.

This module provides a PromptGuard class that uses the meta-llama/Llama-Prompt-Guard-2-86M
model to detect potentially unsafe or harmful content in both user queries and 
LLM-generated responses.
"""
import torch
from functools import lru_cache
from typing import Dict, Tuple, Optional, Union, List

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from query_router.core.config import load_config
from query_router.core.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

class PromptGuard:
    """
    Safety checker for queries and responses using the Llama-Prompt-Guard model.
    
    This class loads the meta-llama/Llama-Prompt-Guard-2-86M model and provides
    methods to check if text content is safe or potentially harmful.
    """
    
    # Instance for singleton pattern
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern to avoid loading multiple model instances."""
        if cls._instance is None:
            cls._instance = super(PromptGuard, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize the PromptGuard with model and tokenizer."""
        # Only initialize once (singleton pattern)
        if self._initialized:
            return
            
        self.config = load_config()
        self.model_config = self.config["models"]["prompt_guard"]
        
        logger.info(
            "initializing_prompt_guard",
            model=self.model_config["name"],
            device=self.model_config["device"],
        )
        
        try:
            # Determine device
            self.device_str = self.model_config["device"]
            self.device = self._determine_device(self.device_str)
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config["name"],
                clean_up_tokenization_spaces=True
            )
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_config["name"],
                torch_dtype=torch.float16 if self.device.type != "cpu" else torch.float32,
                device_map="auto" if self.device.type != "cpu" else None,
                trust_remote_code=True
            )
            
            # Move model to device if not using device_map="auto"
            if self.device.type != "cpu" and not hasattr(self.model, "hf_device_map"):
                self.model.to(self.device)
                
            logger.info("prompt_guard_initialized")
            self._initialized = True
            
        except OSError as e:
            logger.error("prompt_guard_model_download_failed", error=str(e))
            raise RuntimeError(f"Failed to download PromptGuard model: {str(e)}")
        except ValueError as e:
            logger.error("prompt_guard_model_config_error", error=str(e))
            raise RuntimeError(f"Invalid PromptGuard model configuration: {str(e)}")
        except Exception as e:
            logger.error("prompt_guard_initialization_failed", error=str(e))
            raise RuntimeError(f"Failed to initialize PromptGuard: {str(e)}")
    
    def _determine_device(self, device_str: str) -> torch.device:
        """
        Determine the appropriate device based on configuration and availability.
        
        Args:
            device_str: Device string from configuration
            
        Returns:
            torch.device: The device to use
        """
        if device_str == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info("using_cuda_device", device_id=0)
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("using_mps_device")
            else:
                device = torch.device("cpu")
                logger.info("using_cpu_device_fallback")
        else:
            device = torch.device(device_str)
            logger.info("using_specified_device", device=device_str)
            
        return device
    
    @lru_cache(maxsize=128)
    def _classify_text(self, text: str, max_length: int = 512) -> Dict[str, float]:
        """
        Classify text as safe or dangerous using the Prompt Guard model.
        
        Args:
            text: Text to classify
            max_length: Maximum sequence length for tokenization
            
        Returns:
            Dict with classification results including safety score
        """
        if not text or not text.strip():
            logger.warning("empty_text_for_safety_check")
            return {"safe": 1.0, "dangerous": 0.0}
            
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=max_length
            )
            
            # Move inputs to device
            if self.device.type != "cpu" and not hasattr(self.model, "hf_device_map"):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Process logits
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)[0].tolist()
            
            # Return safety classification
            result = {
                "safe": probabilities[0],
                "dangerous": probabilities[1]
            }
            
            logger.debug(
                "safety_check_result", 
                safe_score=result["safe"],
                dangerous_score=result["dangerous"]
            )
            
            return result
            
        except Exception as e:
            logger.error("safety_check_failed", error=str(e), text=text[:100])
            # Default to safe in case of errors
            return {"safe": 1.0, "dangerous": 0.0}
    
    def check_query(self, query: str) -> Tuple[bool, float]:
        """
        Check if a user query is safe.
        
        Args:
            query: User query to check
            
        Returns:
            Tuple of (is_safe: bool, confidence: float)
        """
        logger.info("checking_query_safety", query=query[:100])
        logger.debug("checking_query_safety", query=query[:100])
        result = self._classify_text(query)
        
        is_safe = result["safe"] > result["dangerous"]
        confidence = max(result["safe"], result["dangerous"])
        
        if not is_safe:
            logger.warning(
                "unsafe_query_detected", 
                dangerous_score=result["dangerous"], 
                query=query[:100]
            )
            
        return is_safe, confidence
    
    def check_response(self, response: str) -> Tuple[bool, float]:
        """
        Check if an LLM response is safe.
        
        Args:
            response: LLM response to check
            
        Returns:
            Tuple of (is_safe: bool, confidence: float)
        """
        logger.debug("checking_response_safety", response=response[:100])
        result = self._classify_text(response)
        
        is_safe = result["safe"] > result["dangerous"]
        confidence = max(result["safe"], result["dangerous"])
        
        if not is_safe:
            logger.warning(
                "unsafe_response_detected", 
                dangerous_score=result["dangerous"], 
                response=response[:100]
            )
            
        return is_safe, confidence
    
    def get_blocked_response(self, is_query: bool = True) -> Dict[str, Union[str, float, str]]:
        """
        Generate a standard blocked response for unsafe content.
        
        Args:
            is_query: Whether the blocked content was a query (True) or response (False)
            
        Returns:
            Dict containing the standardized blocked response format
        """
        block_type = "query" if is_query else "response"
        logger.info(f"blocked_{block_type}_response_generated")
        
        return {
            "route": "semantic",
            "confidence": 0.0,
            "flag": "blocked",
            "message": f"The {block_type} was flagged as potentially unsafe and has been blocked."
        }
        
    def __call__(self, text: str, is_query: bool = True) -> Tuple[bool, float]:
        """
        Convenience method to allow using instance as a callable.
        
        Args:
            text: Text to check
            is_query: Whether the text is a query (True) or response (False)
            
        Returns:
            Result of check_query or check_response
        """
        if is_query:
            return self.check_query(text)
        else:
            return self.check_response(text)


# Simple test function to verify model loading
def test_prompt_guard():
    """Test function to verify that the PromptGuard loads correctly."""
    try:
        guard = PromptGuard()
        test_query = "Hello world!"
        is_safe, confidence = guard.check_query(test_query)
        return {
            "model_loaded": True,
            "test_passed": is_safe,
            "confidence": confidence
        }
    except Exception as e:
        logger.error("prompt_guard_test_failed", error=str(e))
        return {
            "model_loaded": False,
            "error": str(e)
        } 