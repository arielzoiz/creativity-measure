import sys
from pathlib import Path

# Put the repo root on sys.path so top-level packages (e.g. `refset`, used like in the
# notebooks) import under pytest, which otherwise only sees the tests/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jaxtyping import install_import_hook

install_import_hook("creativity_measure", "beartype.beartype")
