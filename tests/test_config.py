import os
import pytest
from photo_memory.config import load_config, DEFAULT_CONFIG


def test_load_config_from_file(sample_config):
    config_dict, config_path = sample_config
    config = load_config(config_path)
    assert config["ollama"]["model"] == "gemma4:e4b"
    assert config["schedule"]["start_hour"] == 1
    assert config["dedup"]["hash_threshold"] == 5


def test_load_config_defaults_when_no_file(tmp_path):
    missing_path = str(tmp_path / "nonexistent.yaml")
    config = load_config(missing_path)
    assert config == {**DEFAULT_CONFIG, "data_dir": config["data_dir"]}


def test_load_config_merges_partial(tmp_path):
    import yaml
    partial = {"ollama": {"model": "custom-model"}}
    path = tmp_path / "partial.yaml"
    with open(path, "w") as f:
        yaml.dump(partial, f)
    config = load_config(str(path))
    assert config["ollama"]["model"] == "custom-model"
    assert config["ollama"]["host"] == DEFAULT_CONFIG["ollama"]["host"]
    assert config["schedule"] == DEFAULT_CONFIG["schedule"]


def test_data_dir_created(tmp_path):
    config = load_config(None, data_dir=str(tmp_path / "photo-memory-data"))
    assert os.path.isdir(config["data_dir"])
