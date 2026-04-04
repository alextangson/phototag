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
