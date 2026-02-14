"""Configuration management utilities."""

import os
import logging
from typing import Dict, Any
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

# Default configuration
DEFAULTS = {
    'youtube': {'api_key': ''},
    'database': {'url': 'sqlite:///data/youtube_analytics.db'},
    'models': {
        'engagement_predictor': {
            'params': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42}
        },
        'bandit': {'algorithm': 'thompson_sampling', 'params': {'alpha': 1.0, 'beta': 1.0}}
    },
    'scheduling': {'top_k_recommendations': 3, 'confidence_threshold': 0.7},
    'features': {'text_embedding': {'model': 'all-MiniLM-L6-v2'}},
    'logging': {'level': 'INFO', 'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'}
}


class Config:
    """Lightweight configuration manager."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            # Merge with defaults
            return self._merge_dicts(DEFAULTS.copy(), config)
        except FileNotFoundError:
            logger.warning(f"Config file not found. Using defaults.")
            return DEFAULTS.copy()
    
    def _merge_dicts(self, base: dict, override: dict) -> dict:
        """Recursively merge dictionaries."""
        for key, value in override.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                base[key] = self._merge_dicts(base[key], value)
            else:
                base[key] = value
        return base
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation key."""
        value = self.config
        for k in key.split('.'):
            try:
                value = value[k]
            except (KeyError, TypeError):
                return default
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by dot notation key."""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = value


def setup_logging(config: Config = None) -> None:
    """Setup logging configuration."""
    log_level = config.get('logging.level', 'INFO') if config else 'INFO'
    log_format = config.get('logging.format', '%(asctime)s - %(levelname)s - %(message)s') if config else '%(asctime)s - %(levelname)s - %(message)s'
    
    Path('logs').mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[logging.StreamHandler(), logging.FileHandler('logs/app.log', mode='a')]
    )


def create_directories() -> None:
    """Create necessary directories."""
    for directory in ['data', 'logs', 'models', 'outputs', 'config']:
        Path(directory).mkdir(exist_ok=True)