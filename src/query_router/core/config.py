"""
Configuration loading and management.
"""
import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_config() -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Returns:
        Dict containing configuration
    """
    config_path = os.getenv("CONFIG_PATH", "config/config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def _override_from_env(config: Dict[str, Any], prefix: str = "") -> None:
    """
    Recursively override config values from environment variables.
    
    Environment variables should be in the format:
    APP_SERVER_HOST=0.0.0.0
    APP_MODELS_CLASSIFIER_NAME=facebook/bart-large-mnli
    
    Args:
        config: Configuration dictionary to update
        prefix: Current prefix for nested config keys
    """
    for key, value in config.items():
        env_key = f"{prefix}_{key}".upper().strip("_")
        
        if isinstance(value, dict):
            _override_from_env(value, env_key)
        else:
            env_value = os.getenv(env_key)
            if env_value is not None:
                # Convert environment value to appropriate type
                if isinstance(value, bool):
                    config[key] = env_value.lower() in ("true", "1", "yes")
                elif isinstance(value, int):
                    config[key] = int(env_value)
                elif isinstance(value, float):
                    config[key] = float(env_value)
                else:
                    config[key] = env_value 