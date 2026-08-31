"""Register ``workspace.latex_ocr.*`` as an alias of ``latex_ocr.*``.

Importing this module installs :class:`_AliasFinder` on ``sys.meta_path`` so
that ``import workspace.latex_ocr.models.coca.model`` resolves to the real
``latex_ocr.models.coca.model`` module object. Unpickling checkpoints saved
by pre-split versions of the codebase therefore keeps working even though
the package was renamed/moved.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
from types import ModuleType

_LEGACY_ROOT = "workspace"
_LEGACY_PKG = "workspace.latex_ocr"


def _real_name(legacy_name: str) -> str:
    """Map a legacy ``workspace.latex_ocr[.<rest>]`` name to ``latex_ocr[.<rest>]``."""
    if legacy_name == _LEGACY_PKG:
        return "latex_ocr"
    return "latex_ocr." + legacy_name[len(_LEGACY_PKG) + 1 :]


class _AliasLoader(importlib.abc.Loader):
    """Loader that resolves legacy names to already-executed real modules."""

    def create_module(self, spec):
        if spec.name == _LEGACY_ROOT:
            # Plain "workspace" has no real counterpart; expose a namespace stub.
            dummy = ModuleType(spec.name)
            dummy.__path__ = []  # mark as package
            return dummy
        return importlib.import_module(_real_name(spec.name))

    def exec_module(self, module) -> None:
        # The real module was already executed during create_module.
        return None


class _AliasFinder(importlib.abc.MetaPathFinder):
    """Resolve ``workspace`` / ``workspace.latex_ocr.*`` imports to real modules."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _LEGACY_ROOT and not fullname.startswith(_LEGACY_ROOT + "."):
            return None
        if fullname != _LEGACY_ROOT:
            # Import the real module first so its package __init__ chain runs.
            importlib.import_module(_real_name(fullname))
        return importlib.machinery.ModuleSpec(fullname, _AliasLoader(), is_package=True)


def install() -> None:
    """Install the alias finder once (idempotent)."""
    for finder in sys.meta_path:
        if isinstance(finder, _AliasFinder):
            return
    sys.meta_path.insert(0, _AliasFinder())


install()
