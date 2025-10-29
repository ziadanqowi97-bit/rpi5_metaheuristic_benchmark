import os, time, math, glob, subprocess, psutil, numpy as np
from typing import Optional, Dict, Any

def set_single_core_affinity(core_id: int = 0):
    try:
        p = psutil.Process()
        p.cpu_affinity([core_id])
    except Exception:
        pass  # Best-effort; using `taskset -c` is recommended

def read_cpu_temp():
    # Try vcgencmd first
    try:
        out = subprocess.check_output(["/usr/bin/vcgencmd", "measure_temp"], stderr=subprocess.DEVNULL, timeout=0.2)
        s = out.decode("utf-8").strip()
        if s.startswith("temp=") and s.endswith("'C"):
            return float(s[5:-2])
    except Exception:
        pass
    # Fallback to thermal zones
    for path in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            with open(path, "r") as f:
                v = float(f.read().strip())
                # heuristic: values may be millidegree
                if v > 1000: v = v/1000.0
            return v
        except Exception:
            continue
    return float("nan")

def read_cpu_freq():
    # Return average current freq across CPUs in MHz
    vals = []
    for path in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq"):
        try:
            with open(path, "r") as f:
                v = float(f.read().strip())/1000.0  # kHz -> MHz
            vals.append(v)
        except Exception:
            continue
    if vals:
        return sum(vals)/len(vals)
    # Try psutil as fallback (may be None on ARM)
    try:
        f = psutil.cpu_freq()
        if f and f.current:
            return f.current
    except Exception:
        pass
    return float("nan")

class IterationMonitor:
    def __init__(self, log_every: int = 1):
        self.log_every = max(1, int(log_every))
        self.proc = psutil.Process()
        self.last_cpu_times = self.proc.cpu_times()
        self.last_wall = time.perf_counter()

    def snapshot(self) -> Dict[str, Any]:
        # CPU time delta over wall time since last snapshot
        now_cpu = self.proc.cpu_times()
        now_wall = time.perf_counter()
        user_delta = now_cpu.user - self.last_cpu_times.user
        sys_delta = now_cpu.system - self.last_cpu_times.system
        wall_delta = max(1e-9, now_wall - self.last_wall)
        cpu_util_est = 100.0 * (user_delta + sys_delta) / wall_delta
        mem_rss = self.proc.memory_info().rss  # bytes
        self.last_cpu_times = now_cpu
        self.last_wall = now_wall
        return {
            "cpu_util_est": cpu_util_est,
            "mem_rss": mem_rss,
        }
