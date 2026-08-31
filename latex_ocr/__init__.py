"""latex_ocr: standalone LaTeX formula OCR (training + inference).

Importing this package activates the legacy-checkpoint compatibility shim
(``latex_ocr._compat``), which lets checkpoints pickled with the old
``workspace.latex_ocr.*`` module paths unpickle correctly.
"""

from . import _compat  # noqa: F401

__version__ = "0.1.0"
