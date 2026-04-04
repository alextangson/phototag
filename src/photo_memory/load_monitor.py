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

        if PRESSURE_LEVELS.get(mem_pressure, 2) >= PRESSURE_LEVELS["critical"]:
            return LoadDecision.STOP

        if PRESSURE_LEVELS.get(mem_pressure, 0) >= PRESSURE_LEVELS[self.max_memory_pressure]:
            return LoadDecision.PAUSE

        if cpu_idle < self.min_cpu_idle:
            return LoadDecision.PAUSE

        return LoadDecision.CONTINUE

    def is_past_deadline(self, end_hour: int) -> bool:
        now = datetime.now()
        return now.hour >= end_hour

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
