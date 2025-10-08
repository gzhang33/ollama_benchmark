#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resource Monitor Module
Resource monitoring module providing GPU and CPU usage monitoring functionality
"""

import subprocess
import sys
from typing import Any, Dict, List, Optional

try:
    import psutil
except ImportError:
    print("Installing psutil for system monitoring...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
    import psutil


def get_gpu_usage() -> Optional[Dict[str, Any]]:
    """
    Get GPU usage via nvidia-smi
    
    Returns:
        GPU usage dictionary, returns None if unavailable
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        
        entries = []
        for line in result.stdout.strip().split("\n"):
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 3:
                entries.append(
                    {
                        "gpu_util": float(parts[0]),
                        "memory_used": float(parts[1]),
                        "memory_total": float(parts[2]),
                    }
                )
        return {"gpus": entries} if entries else None
    except Exception as exc:
        print(f"  [Warning] GPU monitoring unavailable: {exc}")
        return None


def get_cpu_usage() -> Optional[float]:
    """
    Get CPU usage percentage
    
    Returns:
        CPU usage percentage, returns None if unavailable
    """
    try:
        return psutil.cpu_percent(interval=None)
    except Exception as exc:
        print(f"  [Warning] CPU monitoring unavailable: {exc}")
        return None


def extract_gpu_util(snapshot: Any) -> float:
    """
    Extract GPU utilization from collected snapshot
    
    Args:
        snapshot: GPU snapshot data
        
    Returns:
        GPU utilization percentage
    """
    if not snapshot:
        return 0.0
    
    if isinstance(snapshot, dict):
        if "gpus" in snapshot and snapshot["gpus"]:
            return float(snapshot["gpus"][0].get("gpu_util", 0))
        return float(snapshot.get("gpu_util", 0))
    
    return float(snapshot) if snapshot else 0.0


class ResourceMonitor:
    """Resource monitor class"""
    
    def __init__(self, enable_gpu: bool = True, enable_cpu: bool = True):
        """
        Initialize resource monitor
        
        Args:
            enable_gpu: Whether to enable GPU monitoring
            enable_cpu: Whether to enable CPU monitoring
        """
        self.enable_gpu = enable_gpu
        self.enable_cpu = enable_cpu
    
    def get_snapshot(self) -> Dict[str, Any]:
        """Get current resource usage snapshot"""
        snapshot = {}
        
        if self.enable_gpu:
            snapshot["gpu"] = get_gpu_usage()
        
        if self.enable_cpu:
            snapshot["cpu"] = get_cpu_usage()
        
        return snapshot
    
    def get_gpu_util(self) -> float:
        """Get current GPU utilization"""
        if not self.enable_gpu:
            return 0.0
        gpu_data = get_gpu_usage()
        return extract_gpu_util(gpu_data)
    
    def get_cpu_util(self) -> float:
        """Get current CPU utilization"""
        if not self.enable_cpu:
            return 0.0
        cpu_data = get_cpu_usage()
        return float(cpu_data) if cpu_data is not None else 0.0

