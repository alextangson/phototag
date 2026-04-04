import os
import tempfile
import pytest
import yaml


@pytest.fixture
def tmp_db_path(tmp_path):
    """Return a temporary database file path."""
    return str(tmp_path / "test_progress.db")


@pytest.fixture
def sample_config(tmp_path):
    """Return a minimal config dict and write it to a temp file."""
    config = {
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
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config, str(config_path)
