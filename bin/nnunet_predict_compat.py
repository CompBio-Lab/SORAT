#!/usr/bin/env python3
"""TEMPORARY WORKAROUND: nnUNet predictor wrapper for NumPy checkpoint compatibility.

Why temporary:
- Some checkpoints require `numpy._core` during torch unpickling.
- Current runtime may only expose `numpy.core`.

Important stability note:
- Import torch first.
- Only install the alias when numpy._core is genuinely unavailable.

Removal plan:
- Remove this wrapper after rebuilding the atrial container with compatible
  dependency versions and switch back to direct nnUNetv2_predict calls.
"""

import sys
import importlib


def _install_numpy_core_compat_alias() -> None:
    """Install numpy._core compatibility aliases for NumPy 1.x runtimes.

    Some checkpoints reference numpy._core symbols at unpickle time, while
    NumPy 1.x exposes numpy.core. We alias both the top-level module and key
    submodules required by scipy/compiled extensions.
    """
    try:
        import numpy._core  # noqa: F401
        return
    except Exception:
        import numpy.core as np_core

        sys.modules.setdefault("numpy._core", np_core)

        for sub in ["multiarray", "_multiarray_umath", "umath", "numerictypes", "_dtype_ctypes"]:
            try:
                mod = importlib.import_module(f"numpy.core.{sub}")
                sys.modules.setdefault(f"numpy._core.{sub}", mod)
            except Exception:
                continue


def main() -> int:
    import torch  # noqa: F401  # Import first to avoid runtime segfault in this environment.

    _install_numpy_core_compat_alias()

    from nnunetv2.inference.predict_from_raw_data import predict_entry_point

    return predict_entry_point()


if __name__ == "__main__":
    raise SystemExit(main())
