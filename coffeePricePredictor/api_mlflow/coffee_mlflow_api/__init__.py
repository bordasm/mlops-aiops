"""REST API package for the coffee quality model tracked with MLflow.

Importing the package makes the production-ready training package
(``productionReadyCode/src``) importable, so the API reuses its data
preparation and modeling code.
"""

import sys
from pathlib import Path

__all__ = ["__version__"]
__version__ = "2.0.0"

_PRODUCTION_SRC_DIR = (
    Path(__file__).resolve().parents[2] / "productionReadyCode" / "src"
)
if str(_PRODUCTION_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_PRODUCTION_SRC_DIR))
