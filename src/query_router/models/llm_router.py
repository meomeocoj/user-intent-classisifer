import json
from typing import Any, Dict, List, Optional

import litellm

from query_router.core.config import load_config
from query_router.core.logging import get_logger

logger = get_logger(__name__)

class LLMRouter:
    """
    Unified LLM router using LiteLLM for multi-provider/model support.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = load_config()["models"]["llm_router"]
        self.provider = config.get("provider", "openai")
        self.model = config.get("model", "gpt-4o")
        self.api_key = config.get("api_key")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1024)
        self.base_url = config.get("base_url")
        self.extra_args = config.get("extra_args", {})

    def _build_prompt(self, query: str, history: Optional[List[str]] = None) -> List[Dict[str, str]]:
        # Few-shot prompt for routing
        system = (
            "You are a router. Classify the user's query into one of: simple, semantic, or agent. "
            "Respond ONLY with a JSON object: {\"route\": \"simple|semantic|agent\"}"
        )
        messages = [
            {"role": "system", "content": system},
        ]
        if history:
            for h in history:
                messages.append({"role": "user", "content": h})
        messages.append({"role": "user", "content": query})
        return messages

    def route(self, query: str, history: Optional[List[str]] = None) -> Dict[str, Any]:
        messages = self._build_prompt(query, history)
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        kwargs.update(self.extra_args)
        try:
            logger.info("llm_router_call", provider=self.provider, model=self.model)
            response = litellm.completion(**kwargs)
            content = response["choices"][0]["message"]["content"]
            logger.debug("llm_router_raw_response", content=content)
            # Parse JSON from response
            try:
                result = json.loads(content)
                if "route" not in result:
                    raise ValueError("No 'route' in LLM response")
                return {"route": result["route"], "raw": result}
            except Exception as e:
                logger.error("llm_router_json_parse_error", error=str(e), content=content)
                return {"route": "semantic", "error": "json_parse_error", "raw": content}
        except Exception as e:
            logger.error("llm_router_api_error", error=str(e))
            return {"route": "semantic", "error": str(e)}

def get_llm_router_from_config(config: Optional[Dict[str, Any]] = None) -> LLMRouter:
    return LLMRouter(config)

# Simple test function for manual verification
def test_llm_router():
    router = get_llm_router_from_config()
    result = router.route("What is the capital of France?")
    print("LLMRouter test result:", result)
    return result 