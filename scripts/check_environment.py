#!/usr/bin/env python3
"""Report the current execution host; never assume it is the intended target server."""

from __future__ import annotations

import importlib.util
import json
import platform
import subprocess
import sys


def main() -> None:
    output = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "architecture": platform.machine(),
        "pytorch_installed": importlib.util.find_spec("torch") is not None,
        "gpu_query": None,
        "target_is_verified": False,
    }
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        )
        output["gpu_query"] = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        output["gpu_query"] = "unavailable; check on the target host"
    if output["pytorch_installed"]:
        try:
            import torch
            output["pytorch_version"] = torch.__version__
            output["pytorch_cuda_build"] = torch.version.cuda
            output["cuda_available"] = bool(torch.cuda.is_available())
        except Exception as exc:  # Broken binary dependencies should be reported, not hidden.
            output["pytorch_import_error"] = type(exc).__name__
    json.dump(output, sys.stdout, ensure_ascii=False, indent=2)
    print()


if __name__ == "__main__":
    main()
