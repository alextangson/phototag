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
                    named_faces='{"fc_001": "张三"}')
    db.upsert_photo("p2", date_taken="2024-02-01T12:00:00",
                    face_cluster_ids='["fc_001"]',
                    named_faces='{"fc_001": "张三"}')
    for uuid in ["p1", "p2"]:
        db.update_photo_status(uuid, "done")
    db.close()

    runner = CliRunner()
    with patch("photo_memory.cli.load_config", return_value=config):
        result = runner.invoke(main, ["--config", config_path, "people", "--min-photos", "1"])

    assert result.exit_code == 0
    assert "fc_001" in result.output or "张三" in result.output
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
                    face_cluster_ids='["fc_001"]', named_faces='{}')
    db.update_photo_status("p1", "done")
    db.close()

    runner = CliRunner()
    with patch("photo_memory.cli.load_config", return_value=config):
        runner.invoke(main, ["--config", config_path, "people", "--min-photos", "1"])
        result = runner.invoke(main, ["--config", config_path, "people", "--name", "fc_001", "李四"])

    assert result.exit_code == 0
    db = Database(db_path)
    row = db.execute("SELECT user_name FROM people WHERE face_cluster_id = ?", ("fc_001",)).fetchone()
    assert row["user_name"] == "李四"
    db.close()


def test_story_year_command_outputs_markdown(tmp_path, sample_config):
    from unittest.mock import patch
    from click.testing import CliRunner
    from photo_memory.cli import main
    from photo_memory.db import Database

    config, config_path = sample_config
    config["data_dir"] = str(tmp_path)
    db_path = str(tmp_path / "progress.db")

    db = Database(db_path)
    db.upsert_event(
        event_id="e1", start_time="2024-05-15T12:00:00", end_time="2024-05-15T13:00:00",
        location_city="北京", location_state=None, photo_count=3,
        face_cluster_ids='[]', summary="颐和园", mood="愉快", cover_photo_uuid="p1",
    )
    db.close()

    runner = CliRunner()
    with patch("photo_memory.cli.load_config", return_value=config), \
         patch("photo_memory.story.generate_period_narrative", return_value="春日回忆"):
        result = runner.invoke(main, ["--config", config_path, "story", "--year", "2024"])

    assert result.exit_code == 0
    assert "2024 年度回忆" in result.output
    assert "春日回忆" in result.output


def test_story_person_command_with_output_file(tmp_path, sample_config):
    from unittest.mock import patch
    from click.testing import CliRunner
    from photo_memory.cli import main
    from photo_memory.db import Database

    config, config_path = sample_config
    config["data_dir"] = str(tmp_path)
    db_path = str(tmp_path / "progress.db")

    db = Database(db_path)
    db.upsert_person(
        face_cluster_id="fc_001", apple_name="张三", user_name=None,
        photo_count=5, event_count=2,
        first_seen="2024-01-01T12:00:00", last_seen="2024-06-01T12:00:00",
        co_appearances="{}", top_locations="[]", appearance_trend="stable",
    )
    db.upsert_event(
        event_id="e1", start_time="2024-01-15T12:00:00", end_time="2024-01-15T13:00:00",
        location_city="北京", location_state=None, photo_count=3,
        face_cluster_ids='["fc_001"]', summary="颐和园", mood="愉快", cover_photo_uuid="p1",
    )
    db.close()

    output_file = tmp_path / "story.md"
    runner = CliRunner()
    with patch("photo_memory.cli.load_config", return_value=config), \
         patch("photo_memory.story.generate_period_narrative", return_value="一段回忆"):
        result = runner.invoke(main, [
            "--config", config_path, "story",
            "--person", "张三",
            "--output", str(output_file),
        ])

    assert result.exit_code == 0
    assert output_file.exists()
    content = output_file.read_text()
    assert "和张三的回忆" in content
    assert "一段回忆" in content


def test_story_requires_one_mode(tmp_path, sample_config):
    """Running `phototag story` without --person/--year/--relationship should error."""
    from unittest.mock import patch
    from click.testing import CliRunner
    from photo_memory.cli import main

    config, config_path = sample_config
    config["data_dir"] = str(tmp_path)

    runner = CliRunner()
    with patch("photo_memory.cli.load_config", return_value=config):
        result = runner.invoke(main, ["--config", config_path, "story"])

    assert result.exit_code != 0
    assert "需要指定" in result.output or "required" in result.output.lower()


def test_search_command_filters_photos(tmp_path, sample_config):
    from unittest.mock import patch
    from click.testing import CliRunner
    from photo_memory.cli import main
    from photo_memory.db import Database

    config, config_path = sample_config
    config["data_dir"] = str(tmp_path)
    db_path = str(tmp_path / "progress.db")

    db = Database(db_path)
    db.upsert_photo("p1", date_taken="2024-03-15T12:00:00",
                    location_city="北京", description="颐和园散步")
    db.upsert_photo("p2", date_taken="2023-05-10T12:00:00",
                    location_city="上海", description="外滩")
    db.update_photo_status("p1", "done")
    db.update_photo_status("p2", "done")
    db.close()

    runner = CliRunner()
    with patch("photo_memory.cli.load_config", return_value=config):
        result = runner.invoke(main, [
            "--config", config_path, "search",
            "--year", "2024",
            "--city", "北京",
        ])

    assert result.exit_code == 0
    assert "p1" in result.output or "颐和园" in result.output
    assert "p2" not in result.output


def test_cleanup_command_shows_counts(tmp_path, sample_config):
    from unittest.mock import patch
    from click.testing import CliRunner
    from photo_memory.cli import main
    from photo_memory.db import Database

    config, config_path = sample_config
    config["data_dir"] = str(tmp_path)
    db_path = str(tmp_path / "progress.db")

    db = Database(db_path)
    db.upsert_photo("p1", importance="cleanup", status="done", description="模糊")
    db.upsert_photo("p2", importance="cleanup", status="done", description="重复")
    db.upsert_photo("p3", importance="review", status="done", description="不确定")
    db.close()

    runner = CliRunner()
    with patch("photo_memory.cli.load_config", return_value=config):
        result = runner.invoke(main, ["--config", config_path, "cleanup"])

    assert result.exit_code == 0
    assert "2" in result.output
    assert "1" in result.output


def test_people_merge_sets_same_user_name(tmp_path, sample_config):
    from unittest.mock import patch
    from click.testing import CliRunner
    from photo_memory.cli import main
    from photo_memory.db import Database

    config, config_path = sample_config
    config["data_dir"] = str(tmp_path)
    db_path = str(tmp_path / "progress.db")

    db = Database(db_path)
    db.upsert_person(face_cluster_id="fc_001", apple_name=None, user_name=None,
                     photo_count=10, event_count=1, first_seen=None, last_seen=None,
                     co_appearances="{}", top_locations="[]", appearance_trend="stable")
    db.upsert_person(face_cluster_id="fc_002", apple_name=None, user_name=None,
                     photo_count=5, event_count=1, first_seen=None, last_seen=None,
                     co_appearances="{}", top_locations="[]", appearance_trend="stable")
    db.close()

    runner = CliRunner()
    with patch("photo_memory.cli.load_config", return_value=config):
        result = runner.invoke(main, [
            "--config", config_path, "people",
            "--merge", "fc_001", "fc_002", "张三",
        ])

    assert result.exit_code == 0
    db = Database(db_path)
    a = db.execute("SELECT user_name FROM people WHERE face_cluster_id='fc_001'").fetchone()
    b = db.execute("SELECT user_name FROM people WHERE face_cluster_id='fc_002'").fetchone()
    assert a["user_name"] == "张三"
    assert b["user_name"] == "张三"
    db.close()


def test_story_html_output_creates_html_file(tmp_path, sample_config):
    from unittest.mock import patch
    from click.testing import CliRunner
    from photo_memory.cli import main
    from photo_memory.db import Database
    from PIL import Image

    config, config_path = sample_config
    config["data_dir"] = str(tmp_path)
    db_path = str(tmp_path / "progress.db")

    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    img_path = img_dir / "photo1.jpg"
    Image.new("RGB", (100, 100), "red").save(img_path, "JPEG")

    db = Database(db_path)
    db.upsert_photo("p1", date_taken="2024-03-15T12:00:00",
                    file_path=str(img_path), description="颐和园")
    db.update_photo_status("p1", "done")
    db.upsert_event(
        event_id="e1", start_time="2024-03-15T12:00:00", end_time="2024-03-15T13:00:00",
        location_city="北京", location_state=None, photo_count=1,
        face_cluster_ids='[]', summary="颐和园散步", mood="愉快", cover_photo_uuid="p1",
    )
    db.link_photos_to_event("e1", ["p1"])
    db.close()

    output_file = tmp_path / "story.html"
    runner = CliRunner()
    with patch("photo_memory.cli.load_config", return_value=config), \
         patch("photo_memory.story.generate_period_narrative", return_value="春日回忆"):
        result = runner.invoke(main, [
            "--config", config_path, "story",
            "--year", "2024",
            "--output", str(output_file),
        ])

    assert result.exit_code == 0
    assert output_file.exists()
    content = output_file.read_text()
    assert "<!DOCTYPE html>" in content
    assert "2024 年度回忆" in content
    assert 'data:image/jpeg;base64,' in content
