"""
Query depth classifier using zero-shot classification.
"""
import re
from typing import Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Pipeline
from transformers.pipelines import pipeline

from query_router.core.config import load_config
from query_router.core.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

class QueryClassifier:
    """Classifies query depth using zero-shot classification."""
    
    LABELS = ["simple", "semantic", "agent"]
    HYPOTHESES = {
        "agent":   "This question involves complex planning, multi-step reasoning, or designing a detailed strategy.",
        "semantic": "This question requires retrieving and synthesizing information from external sources or documents.",
        "simple":  "This question requires a brief, factual answer that can be provided directly without research or analysis.",
    }

    def __init__(self) -> None:
        """Initialize the classifier"""
        self.config = load_config()
        self.model_config = self.config["models"]["classifier"]
        
        logger.info(
            "initializing_classifier",
            model=self.model_config["name"],
            device=self.model_config["device"],
        )
        
        # Initialize model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_config["name"]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config["name"],
            clean_up_tokenization_spaces=True  # Explicitly set to True to handle deprecation warning
        )
        
        # Move model to specified device
        self.device = torch.device(self.model_config["device"])
        self.model.to(self.device)
        
        # Create classification pipeline
        self.classifier: Pipeline = pipeline(
            "zero-shot-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device if self.device.type == "cuda" else -1,
        )
        
        logger.info("classifier_initialized")
    
    async def classify(self, query: str, history: list[str] | None = None) -> Tuple[str, float]:
        """
        Classify the depth of a query using zero-shot classification.
        
        Args:
            query: The query to classify
            history: Optional conversation history (currently unused)
            
        Returns:
            Tuple of (route_label, confidence_score)
        """
        # Create hypothesis for each label
        query = self.preprocess_query(query)

        logger.debug("classifying_query", query=query)
        
        # Run zero-shot classification
        result = self.classifier(
            query,
            list(self.HYPOTHESES.values()),
            multi_label=False,
        )
        
        # Get predicted label and confidence
        label_idx = result["scores"].index(max(result["scores"]))
        route = self.LABELS[label_idx]
        confidence = result["scores"][label_idx]
        
        logger.debug(
            "classification_result",
            route=route,
            confidence=confidence,
            all_scores=dict(zip(self.LABELS, result["scores"], strict=False))
        )
        
        return route, confidence
    
    def preprocess_query(self, query: str) -> str:
        """Clean and normalize the query."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Remove extra whitespace and normalize
        query = re.sub(r'\s+', ' ', query.strip())
        # Remove excessive punctuation but keep basic structure
        query = re.sub(r'[.!?]{2,}', '.', query)
        query = re.sub(r'[^\w\s.,!?]', '', query)
        
        # Augment query to emphasize complexity
        if any(keyword in query.lower() for keyword in ["plan", "design", "strategy", "research"]):
            query = f"Complex task: {query}"
        
        return query
    def __call__(self, query: str, history: list[str] | None = None) -> Tuple[str, float]:
        """Convenience method to allow using instance as a callable."""
        return self.classify(query, history) 