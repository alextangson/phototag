"""Configuration loading with deep-merge defaults."""

import copy
import os
from pathlib import Path

import yaml

DEFAULT_DATA_DIR = os.path.expanduser("~/.phototag")
DEFAULT_CONFIG_PATH = os.path.join(DEFAULT_DATA_DIR, "config.yaml")

DEFAULT_CONFIG = {
    "ollama": {
        "host": "http://localhost:11434",
        "model": "gemma4:e4b",
        "timeout": 60,
    },
    "schedule": {
        "start_hour": 1,
        "end_hour": 7,
        "check_interval": 10,
    },
    "load": {
        "max_memory_pressure": "warn",
        "min_cpu_idle": 30,
        "pause_seconds": 300,
    },
    "dedup": {
        "hash_threshold": 5,
    },
    "layers": {
        "layer1_organize": True,
        "layer2_analyze": False,
        "layer3_brief": False,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(config_path: str | None = None, data_dir: str | None = None) -> dict:
    """Load config from YAML file, deep-merged with defaults."""
    config = copy.deepcopy(DEFAULT_CONFIG)

    path = config_path or DEFAULT_CONFIG_PATH
    if os.path.isfile(path):
        with open(path) as f:
            user_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, user_config)

    data = data_dir or DEFAULT_DATA_DIR
    os.makedirs(data, exist_ok=True)
    config["data_dir"] = data

    return config
