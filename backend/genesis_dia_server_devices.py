from __future__ import annotations

from typing import Dict, List

import torch

from .genesis_dia_server_storage import normalize_device_setting


class DeviceUnavailableError(RuntimeError):
    """The configured device does not exist on this host/container."""


def list_device_options() -> List[Dict[str, str]]:
    """Dropdown options for the admin UI: auto, CPU and every visible CUDA GPU."""

    options = [
        {"label": "Auto (first GPU, else CPU)", "value": "auto"},
        {"label": "CPU", "value": "cpu"},
    ]
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            total_gb = properties.total_memory / (1024 ** 3)
            options.append({"label": f"GPU {index}: {properties.name} ({total_gb:.0f} GB)", "value": f"cuda:{index}"})
    return options


def resolve_device(setting: str) -> torch.device:
    """Map a saved ``gpu_device`` setting to a torch device, failing loudly on a missing GPU."""

    normalized = normalize_device_setting(setting)
    if normalized == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        if normalized == "auto":
            return torch.device("cpu")
        raise DeviceUnavailableError(
            f"GPU '{normalized}' ist ausgewaehlt, aber CUDA ist nicht verfuegbar "
            "(NVIDIA-Treiber / NVIDIA Container Toolkit pruefen oder 'Auto'/'CPU' waehlen)."
        )
    if normalized == "auto":
        return torch.device(f"cuda:{torch.cuda.current_device()}")

    index = int(normalized.split(":", 1)[1])
    count = torch.cuda.device_count()
    if index >= count:
        raise DeviceUnavailableError(
            f"GPU '{normalized}' existiert nicht, es wurden {count} GPU(s) erkannt. Bitte in den Einstellungen neu waehlen."
        )
    return torch.device(normalized)


__all__ = ["DeviceUnavailableError", "list_device_options", "resolve_device"]
