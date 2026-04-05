import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from photo_memory.cli import main


def test_cli_scan_command():
    runner = CliRunner()
    with patch("photo_memory.cli.load_config") as mock_config, \
         patch("photo_memory.cli.Database") as mock_db, \
         patch("photo_memory.cli.scan_photos_into_db", return_value=100):
        mock_config.return_value = {"data_dir": "/tmp/test"}
        result = runner.invoke(main, ["scan"])
    assert result.exit_code == 0
    assert "100" in result.output


def test_cli_status_command():
    runner = CliRunner()
    with patch("photo_memory.cli.load_config") as mock_config, \
         patch("photo_memory.cli.Database") as mock_db_cls:
        mock_config.return_value = {"data_dir": "/tmp/test"}
        mock_db = MagicMock()
        mock_row = {"total": 100, "done": 50, "error": 2, "pending": 48}
        mock_db.execute.return_value.fetchone.return_value = mock_row
        mock_db_cls.return_value = mock_db
        result = runner.invoke(main, ["status"])
    assert result.exit_code == 0
    assert "100" in result.output


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "scan" in result.output
    assert "run" in result.output
    assert "status" in result.output


def test_reprocess_command_resets_done_photos(tmp_path, sample_config):
    """reprocess should reset 'done' photos back to 'pending'."""
    from unittest.mock import patch
    from click.testing import CliRunner
    from photo_memory.cli import main
    from photo_memory.db import Database

    config, config_path = sample_config
    config["data_dir"] = str(tmp_path)

    db_path = str(tmp_path / "progress.db")
    db = Database(db_path)
    db.upsert_photo("uuid-1")
    db.upsert_photo("uuid-2")
    db.upsert_photo("uuid-3")
    db.update_photo_status("uuid-1", "done")
    db.update_photo_status("uuid-2", "done")
    db.close()

    runner = CliRunner()
    with patch("photo_memory.cli.load_config", return_value=config):
        result = runner.invoke(main, ["--config", config_path, "reprocess"])

    assert result.exit_code == 0
    assert "2" in result.output  # 2 photos reset

    db = Database(db_path)
    assert db.get_photo("uuid-1")["status"] == "pending"
    assert db.get_photo("uuid-2")["status"] == "pending"
    assert db.get_photo("uuid-3")["status"] == "pending"  # was already pending
    db.close()


def test_events_command_builds_and_lists(tmp_path, sample_config):
    """`phototag events` should build events and print a summary."""
    from unittest.mock import patch
    from click.testing import CliRunner
    from photo_memory.cli import main
    from photo_memory.db import Database
    import json

    config, config_path = sample_config
    config["data_dir"] = str(tmp_path)
    db_path = str(tmp_path / "progress.db")

    db = Database(db_path)
    db.upsert_photo("p1", date_taken="2024-09-28T13:00:00",
                    face_cluster_ids='[]', ai_result=json.dumps({"narrative": "test"}))
    db.update_photo_status("p1", "done")
    db.close()

    runner = CliRunner()
    with patch("photo_memory.cli.load_config", return_value=config), \
         patch("photo_memory.events.summarize_event",
               return_value={"summary": "测试", "mood": ""}):
        result = runner.invoke(main, ["--config", config_path, "events"])

    assert result.exit_code == 0
    assert "event" in result.output.lower() or "事件" in result.output


def test_people_command_builds_and_lists(tmp_path, sample_config):
    from unittest.mock import patch
    from click.testing import CliRunner
    from photo_memory.cli import main
    from photo_memory.db import Database

    config, config_path = sample_config
    config["data_dir"] = str(tmp_path)
    db_path = str(tmp_path / "progress.db")

    db = Database(db_path)
    db.upsert_photo("p1", date_taken="2024-01-01T12:00:00",
                    face_cluster_ids='["fc_001"]',
                    named_faces='["唐嘉鑫"]')
    db.upsert_photo("p2", date_taken="2024-02-01T12:00:00",
                    face_cluster_ids='["fc_001"]',
                    named_faces='["唐嘉鑫"]')
    for uuid in ["p1", "p2"]:
        db.update_photo_status(uuid, "done")
    db.close()

    runner = CliRunner()
    with patch("photo_memory.cli.load_config", return_value=config):
        result = runner.invoke(main, ["--config", config_path, "people", "--min-photos", "1"])

    assert result.exit_code == 0
    assert "fc_001" in result.output or "唐嘉鑫" in result.output
    assert "2" in result.output


def test_people_name_command_sets_user_name(tmp_path, sample_config):
    from unittest.mock import patch
    from click.testing import CliRunner
    from photo_memory.cli import main
    from photo_memory.db import Database

    config, config_path = sample_config
    config["data_dir"] = str(tmp_path)
    db_path = str(tmp_path / "progress.db")

    db = Database(db_path)
    db.upsert_photo("p1", date_taken="2024-01-01T12:00:00",
                    face_cluster_ids='["fc_001"]', named_faces='[]')
    db.update_photo_status("p1", "done")
    db.close()

    runner = CliRunner()
    with patch("photo_memory.cli.load_config", return_value=config):
        runner.invoke(main, ["--config", config_path, "people", "--min-photos", "1"])
        result = runner.invoke(main, ["--config", config_path, "people", "--name", "fc_001", "阿菁"])

    assert result.exit_code == 0
    db = Database(db_path)
    row = db.execute("SELECT user_name FROM people WHERE face_cluster_id = ?", ("fc_001",)).fetchone()
    assert row["user_name"] == "阿菁"
    db.close()
