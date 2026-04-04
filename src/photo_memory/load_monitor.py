"""System load monitoring for adaptive processing."""

import subprocess
from datetime import datetime
from enum import Enum


class LoadDecision(Enum):
    CONTINUE = "continue"
    PAUSE = "pause"
    STOP = "stop"


PRESSURE_LEVELS = {"normal": 0, "warn": 1, "critical": 2}


class LoadMonitor:
    def __init__(self, max_memory_pressure: str = "warn", min_cpu_idle: float = 30.0):
        self.max_memory_pressure = max_memory_pressure
        self.min_cpu_idle = min_cpu_idle

    def check(self) -> LoadDecision:
        mem_pressure = self._get_memory_pressure()
        cpu_idle = self._get_cpu_idle()

        mem_level = PRESSURE_LEVELS.get(mem_pressure, 0)
        threshold = PRESSURE_LEVELS.get(self.max_memory_pressure, 1)

        if mem_level > threshold:
            return LoadDecision.STOP

        if mem_level == threshold:
            return LoadDecision.PAUSE

        if cpu_idle < self.min_cpu_idle:
            return LoadDecision.PAUSE

        return LoadDecision.CONTINUE

    def is_past_deadline(self, end_hour: int, start_hour: int = 1) -> bool:
        """Check if current time is past the deadline.

        The valid run window is [start_hour, end_hour). Outside this window
        we do NOT enforce the deadline — so manual runs at any time work fine.
        Only within the run window do we check if we've passed end_hour.

        Example: start_hour=1, end_hour=7
          - 0:30  → outside window → False (allow manual run)
          - 3:00  → inside window, before deadline → False
          - 7:01  → inside window, past deadline → True
          - 22:00 → outside window → False (allow manual run)
        """
        now = datetime.now()
        hour = now.hour
        # Only enforce deadline if we're in the run window [start_hour, end_hour]
        if start_hour <= hour < end_hour:
            return False  # still within window
        elif hour >= end_hour and hour < end_hour + 2:
            # Grace period: just past deadline (e.g. 7-9 AM), enforce stop
            return True
        else:
            # Outside run window entirely (evening, night) — don't block
            return False

    def _get_memory_pressure(self) -> str:
        try:
            result = subprocess.run(
                ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                capture_output=True, text=True, timeout=5,
            )
            level = int(result.stdout.strip())
            if level == 0:
                return "normal"
            elif level == 1:
                return "warn"
            else:
                return "critical"
        except Exception:
            return "normal"

    def _get_cpu_idle(self) -> float:
        try:
            result = subprocess.run(
                ["top", "-l", "1", "-n", "0"],
                capture_output=True, text=True, timeout=10,
            )
            for line in result.stdout.splitlines():
                if "CPU usage" in line:
                    parts = line.split(",")
                    for part in parts:
                        if "idle" in part:
                            return float(part.strip().split("%")[0])
            return 100.0
        except Exception:
            return 100.0
