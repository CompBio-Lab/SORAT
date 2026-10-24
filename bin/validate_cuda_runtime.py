#!/usr/bin/env python3
"""Validate that the PyTorch CUDA runtime can execute on the allocated GPU.

This is intentionally a small, model-independent check. It catches the
common HPC failure mode where a container has a newer PyTorch build than the
GPU architecture on the cluster (for example, a V100/SM70 with a build that
only contains SM75+ kernels).
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch


def _device_index(device: torch.device | int | None) -> int:
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, int):
        return device
    return device.index if device.index is not None else torch.cuda.current_device()


def _architecture_names(capability: tuple[int, int]) -> set[str]:
    major, minor = capability
    code = f"{major}{minor}"
    return {f"sm_{code}", f"compute_{code}"}


def _runtime_report(device_index: int) -> dict[str, Any]:
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(device_index))
    return {
        "device_index": device_index,
        "device_name": torch.cuda.get_device_name(device_index),
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "requested_architecture": f"sm_{capability[0]}{capability[1]}",
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "compiled_architectures": list(torch.cuda.get_arch_list()),
    }


def validate_cuda_runtime(device: torch.device | int | None = None) -> dict[str, Any]:
    """Return runtime details or raise a descriptive compatibility error."""

    if not torch.cuda.is_available():
        raise RuntimeError(
            "VSA-3L requires CUDA, but PyTorch cannot access a CUDA device. "
            "Check the SLURM GPU request and that Apptainer is launched with --nv."
        )

    device_index = _device_index(device)
    report = _runtime_report(device_index)
    supported = set(report["compiled_architectures"])
    required = _architecture_names(
        tuple(int(part) for part in report["compute_capability"].split("."))
    )

    if supported and not supported.intersection(required):
        raise RuntimeError(
            "Incompatible VSA-3L CUDA/PyTorch runtime: "
            f"GPU {report['device_name']} has compute capability "
            f"{report['compute_capability']} ({report['requested_architecture']}), "
            f"but PyTorch {report['torch_version']} (CUDA {report['torch_cuda_version']}) "
            f"was built for {', '.join(report['compiled_architectures'])}. "
            "Rebuild or replace the VSA-3L container with a PyTorch build that "
            "includes the GPU architecture; do not fall back to CPU."
        )

    # Architecture lists are useful diagnostics, but the executable smoke test
    # is the authority because CUDA extensions can still be broken in an image
    # whose torch metadata looks plausible.
    try:
        with torch.no_grad():
            sample = torch.zeros((1, 1, 8, 8), device=f"cuda:{device_index}")
            torch.nn.Conv2d(1, 1, kernel_size=3, device=sample.device)(sample)
        torch.cuda.synchronize(device_index)
    except Exception as exc:  # pragma: no cover - exercised on GPU nodes
        raise RuntimeError(
            "VSA-3L CUDA runtime smoke test failed on "
            f"{report['device_name']} (CC {report['compute_capability']}) with "
            f"PyTorch {report['torch_version']} / CUDA {report['torch_cuda_version']}. "
            "Rebuild or replace the container for the allocated GPU architecture."
        ) from exc

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the successful runtime report as JSON",
    )
    args = parser.parse_args()

    report = validate_cuda_runtime()
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(
            "CUDA runtime compatible: "
            f"{report['device_name']} (CC {report['compute_capability']}); "
            f"PyTorch {report['torch_version']} / CUDA {report['torch_cuda_version']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
