"""Compatibility helpers for loading legacy checkpoints.

Released model checkpoints were serialized with ``torch.save(model)`` from
older versions of this project, so unpickling them requires the *original*
module paths (``workspace.latex_ocr...``) to be importable. This package
registers lightweight alias modules that re-export the current
``latex_ocr`` implementations, letting old checkpoints load unmodified.

Import :mod:`latex_ocr.checkpoints` (done automatically by
``latex_ocr/__init__.py``) to activate.
"""

from . import legacy_workspace  # noqa: F401  (import registers the alias)

__all__ = ["legacy_workspace"]
