import pytest
from unittest.mock import patch
from photo_memory.load_monitor import LoadMonitor, LoadDecision


def test_should_continue_when_load_is_low():
    monitor = LoadMonitor(max_memory_pressure="warn", min_cpu_idle=30)
    with patch.object(monitor, "_get_memory_pressure", return_value="normal"), \
         patch.object(monitor, "_get_cpu_idle", return_value=80.0):
        assert monitor.check() == LoadDecision.CONTINUE


def test_should_pause_when_cpu_busy():
    monitor = LoadMonitor(max_memory_pressure="warn", min_cpu_idle=30)
    with patch.object(monitor, "_get_memory_pressure", return_value="normal"), \
         patch.object(monitor, "_get_cpu_idle", return_value=20.0):
        assert monitor.check() == LoadDecision.PAUSE


def test_should_pause_when_memory_warn():
    monitor = LoadMonitor(max_memory_pressure="warn", min_cpu_idle=30)
    with patch.object(monitor, "_get_memory_pressure", return_value="warn"), \
         patch.object(monitor, "_get_cpu_idle", return_value=80.0):
        assert monitor.check() == LoadDecision.PAUSE


def test_should_stop_when_memory_critical():
    monitor = LoadMonitor(max_memory_pressure="warn", min_cpu_idle=30)
    with patch.object(monitor, "_get_memory_pressure", return_value="critical"), \
         patch.object(monitor, "_get_cpu_idle", return_value=80.0):
        assert monitor.check() == LoadDecision.STOP


def test_time_limit_not_reached_during_run():
    monitor = LoadMonitor(max_memory_pressure="warn", min_cpu_idle=30)
    with patch("photo_memory.load_monitor.datetime") as mock_dt:
        from datetime import datetime
        mock_dt.now.return_value = datetime(2026, 4, 5, 3, 0)  # 3 AM — running
        assert monitor.is_past_deadline(end_hour=7, start_hour=1) is False


def test_time_limit_reached_morning():
    monitor = LoadMonitor(max_memory_pressure="warn", min_cpu_idle=30)
    with patch("photo_memory.load_monitor.datetime") as mock_dt:
        from datetime import datetime
        mock_dt.now.return_value = datetime(2026, 4, 5, 7, 1)  # 7:01 AM — past deadline
        assert monitor.is_past_deadline(end_hour=7, start_hour=1) is True


def test_time_limit_not_reached_evening_manual_run():
    """Running manually at 10 PM should NOT trigger deadline."""
    monitor = LoadMonitor(max_memory_pressure="warn", min_cpu_idle=30)
    with patch("photo_memory.load_monitor.datetime") as mock_dt:
        from datetime import datetime
        mock_dt.now.return_value = datetime(2026, 4, 5, 22, 0)  # 10 PM
        assert monitor.is_past_deadline(end_hour=7, start_hour=1) is False


def test_time_limit_before_start_hour():
    """Midnight (before start_hour) should NOT trigger deadline."""
    monitor = LoadMonitor(max_memory_pressure="warn", min_cpu_idle=30)
    with patch("photo_memory.load_monitor.datetime") as mock_dt:
        from datetime import datetime
        mock_dt.now.return_value = datetime(2026, 4, 5, 0, 30)  # 0:30 AM
        assert monitor.is_past_deadline(end_hour=7, start_hour=1) is False
