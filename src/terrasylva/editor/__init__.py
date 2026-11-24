"""Visual editor. Needs the ``editor`` extra: ``pip install terrasylva[editor]``."""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Open the editor, or explain what is missing."""
    try:
        from .window import main as run
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] not in ("PySide6", "shiboken6", "moderngl"):
            raise
        print(f'The editor needs {e.name}. Install it with: pip install "terrasylva[editor]"',
              file=sys.stderr)
        return 1
    return run(argv)
