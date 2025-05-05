import logging
import uuid
import time
from typing import Any, Dict, List, Optional

from src.query_router.models.classifier import QueryClassifier
from src.query_router.models.llm_router import LLMRouter, get_llm_router_from_config
from src.query_router.core.config import load_config


class RouterService:
    """
    Orchestrates the routing logic: (stubbed) safety check -> BART classifier (QueryClassifier) -> LLM fallback (now implemented).
    """
    def __init__(self, llm_router: Optional[LLMRouter] = None, logger: Optional[logging.Logger] = None):
        self.classifier = QueryClassifier()
        if llm_router is not None:
            self.llm_router = llm_router
        else:
            self.llm_router = get_llm_router_from_config()
        self.logger = logger or logging.getLogger(__name__)
        self.config = load_config()
        self.classifier_model_name = self.config["models"]["classifier"]["name"]

    async def route_query(self, query: str, history: Optional[List[Dict[str, str]]] = None, trace_id: Optional[str] = None, timing_enabled: bool = False) -> Dict[str, Any]:
        """
        Process a query through the routing pipeline.
        
        Args:
            query: The user's query string
            history: Optional conversation history as a list of message objects
            trace_id: Optional unique identifier for this request
            timing_enabled: Whether to log timing information
            
        Returns:
            Dict containing the routing result with the following keys:
            - route: The determined route ("simple", "semantic", "agent", "blocked", or "error")
            - confidence: Confidence score from the classifier (0-1)
            - trace_id: Unique identifier for this request
            - Additional optional keys based on processing results
        """
        timing = {}
        router_start = time.time()
        if trace_id is None:
            trace_id = str(uuid.uuid4())
        
        # Parameter validation
        if not query or not isinstance(query, str):
            self.logger.warning(f"[{trace_id}] Invalid query format")
            return {
                "route": "error", 
                "confidence": 0.0, 
                "error": "Invalid query format", 
                "trace_id": trace_id
            }, timing
            
        # Convert history format if needed
        processed_history = None
        if history:
            try:
                # Extract just the text from history objects
                processed_history = []
                for msg in history:
                    if isinstance(msg, dict) and any(key in msg for key in ('user', 'assistant')):
                        processed_history.append(msg)
            except Exception as e:
                self.logger.warning(f"[{trace_id}] Invalid history format: {e}")
                # Continue without history rather than failing
        
        # (Stub) Safety check
        # TODO: Replace with real PromptGuard logic when available
        is_safe = True
        if not is_safe:
            self.logger.warning(f"[{trace_id}] Query blocked by safety check")
            return {
                "route": "blocked", 
                "confidence": 1.0, 
                "reason": "Query contains unsafe content", 
                "trace_id": trace_id
            }, timing

        try:
            classifier_start = time.time()
            route, confidence = await self.classifier.classify(query, processed_history)
            classifier_time = (time.time() - classifier_start) * 1000
            if timing_enabled:
                timing["classifier_time_ms"] = f"{classifier_time:.2f}"
                self.logger.info(f"[{trace_id}] classifier_time_ms: {classifier_time:.2f}")
        except Exception as e:
            self.logger.error(f"[{trace_id}] Classifier error: {e}")
            return {
                "route": "error", 
                "confidence": 0.0, 
                "error": str(e), 
                "trace_id": trace_id
            }, timing

        self.logger.info(f"[{trace_id}] {self.classifier_model_name} classified: {route} (conf: {confidence:.3f})")

        if route == "simple" and confidence >= 0.75:
            router_time = (time.time() - router_start) * 1000
            if timing_enabled:
                timing["llm_time_ms"] = "0.00"
                timing["total_router_time_ms"] = f"{router_time:.2f}"
            return {
                "route": route,
                "confidence": confidence,
                "trace_id": trace_id,
                "model": self.classifier_model_name
            }, timing
        else:
            # LLM fallback logic
            self.logger.info(f"[{trace_id}] Using LLM fallback via LLMRouter")
            llm_start = time.time()
            llm_result = self.llm_router.route(query, processed_history)
            llm_time = (time.time() - llm_start) * 1000
            if timing_enabled:
                timing["llm_time_ms"] = f"{llm_time:.2f}"
                self.logger.info(f"[{trace_id}] llm_time_ms: {llm_time:.2f}")
            router_time = (time.time() - router_start) * 1000
            if timing_enabled:
                timing["total_router_time_ms"] = f"{router_time:.2f}"
            llm_route = llm_result.get("route", route)
            llm_conf = llm_result.get("confidence", confidence)
            result = {
                "route": llm_route,
                "confidence": llm_conf,
                "trace_id": trace_id,
                "model": "llm",
                "llm_raw": llm_result
            }
            return result, timing 