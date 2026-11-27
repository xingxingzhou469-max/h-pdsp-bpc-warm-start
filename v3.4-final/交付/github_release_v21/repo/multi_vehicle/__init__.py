"""Multi-vehicle H-PDSP v2.1 solver, corrected paper baselines, and independent proof replay."""

from pathlib import Path
import sys

PACKAGE_DIR = Path(__file__).resolve().parent
ENGINE_DIR = PACKAGE_DIR / "engine"
BUILD_DIR = PACKAGE_DIR / "build"


def ensure_engine_on_path():
    """Place multi_vehicle/engine on sys.path so frozen modules import verbatim."""
    engine = str(ENGINE_DIR)
    if engine not in sys.path:
        sys.path.insert(0, engine)
