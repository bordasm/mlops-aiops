"""REST API package for the coffee quality benchmark model.

Importing the package makes the production-ready training package
(``productionReadyCode/src``) importable, so the API reuses the same
training code instead of duplicating it.
"""

import sys
from pathlib import Path

__all__ = ["__version__"]
__version__ = "1.0.0"

_PRODUCTION_SRC_DIR = (
    Path(__file__).resolve().parents[2] / "productionReadyCode" / "src"
)
if str(_PRODUCTION_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_PRODUCTION_SRC_DIR))
