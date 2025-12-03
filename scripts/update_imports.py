#!/usr/bin/env python3
"""
Automated import updater for nwc_webapp refactoring.
Run after each file move to update all imports throughout the codebase.

Usage:
    python scripts/update_imports.py [--phase PHASE]

Phases:
    utils     - Remove utils.py imports
    logging   - Update logging config imports
    all       - Apply all package rename updates
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# Phase 1: Remove utils.py imports
UTILS_REMOVALS = {
    'from nwc_webapp.utils import load_gif_as_bytesio': 'from nwc_webapp.data.gifs import load_gif_as_bytesio',
    'from nwc_webapp.utils import lincol_2_yx, yx_2_latlon': 'from nwc_webapp.geo.coordinates import lincol_2_yx, yx_2_latlon',
    'from nwc_webapp.utils import lincol_2_yx': 'from nwc_webapp.geo.coordinates import lincol_2_yx',
    'from nwc_webapp.utils import yx_2_latlon': 'from nwc_webapp.geo.coordinates import yx_2_latlon',
    'from nwc_webapp.utils import compute_figure_gpd': 'from nwc_webapp.rendering.figures import compute_figure_gpd',
    'from nwc_webapp.utils import cmap, norm': 'from nwc_webapp.rendering.colormaps import cmap, norm',
    'from nwc_webapp.utils import cmap': 'from nwc_webapp.rendering.colormaps import cmap',
    'from nwc_webapp.utils import norm': 'from nwc_webapp.rendering.colormaps import norm',
    'from nwc_webapp.utils import worker_thread': 'from nwc_webapp.core.workers import worker_thread',
    'from nwc_webapp.utils import load_prediction_thread': 'from nwc_webapp.core.workers import load_prediction_thread',
    'from nwc_webapp.utils import setup_logger': 'from nwc_webapp.logging_config import setup_logger',
}

# Phase 2: Update logging config imports
LOGGING_UPDATES = {
    'from nwc_webapp.config.logging_config import get_logger': 'from nwc_webapp.logging_config import setup_logger',
    'from nwc_webapp.config.logging_config import setup_logging': 'from nwc_webapp.logging_config import setup_logger',
}

# Phase 3: All package renames
ALL_PACKAGE_RENAMES = {
    # Page modules
    'from nwc_webapp.page_modules import': 'from nwc_webapp.pages import',
    'from nwc_webapp.page_modules.': 'from nwc_webapp.pages.',
    'import nwc_webapp.page_modules.': 'import nwc_webapp.pages.',
    'page_modules.nowcasting_utils': 'pages.nowcasting_helpers',
    'page_modules.csi_utils': 'pages.csi_helpers',

    # Services
    'from nwc_webapp.services.parallel_code import': 'from nwc_webapp.rendering.gifs import',
    'from nwc_webapp.services.pbs import': 'from nwc_webapp.hpc.pbs import',
    'from nwc_webapp.services.prediction_service import': 'from nwc_webapp.models.services import',
    'from nwc_webapp.services.data_service import': 'from nwc_webapp.models.data_service import',
    'from nwc_webapp.services.workers import': 'from nwc_webapp.core.workers import',
    'from nwc_webapp.services.mock_realtime_service import': 'from nwc_webapp.mock.realtime import',

    # Prediction
    'from nwc_webapp.prediction.loaders import': 'from nwc_webapp.models.predictions import',
    'from nwc_webapp.prediction.jobs import': 'from nwc_webapp.core.jobs import',
    'from nwc_webapp.prediction.visualization import': 'from nwc_webapp.rendering.visualization import',

    # Visualization
    'from nwc_webapp.visualization.figures import': 'from nwc_webapp.rendering.figures import',
    'from nwc_webapp.visualization.colormaps import': 'from nwc_webapp.rendering.colormaps import',
    'from nwc_webapp.visualization.fit_diagram import': 'from nwc_webapp.rendering.fit_diagram import',

    # Background
    'from nwc_webapp.background.workers import': 'from nwc_webapp.core.workers import',

    # UI
    'from nwc_webapp.ui.layouts import': 'from nwc_webapp.ui.components import',
    'from nwc_webapp.ui.state import': 'from nwc_webapp.core.session import',

    # Data
    'from nwc_webapp.data.gif_utils import': 'from nwc_webapp.data.gifs import',
    'from nwc_webapp.data.create_dataset import': 'from nwc_webapp.data.dataset import',

    # Mock
    'from nwc_webapp.services.mock.mock_data_generator import': 'from nwc_webapp.mock.generator import',
    'from nwc_webapp.services.mock.mock import': 'from nwc_webapp.mock.predictions import',
}


def update_imports_in_file(file_path: Path, mapping: Dict[str, str]) -> bool:
    """Update imports in a single file."""
    try:
        content = file_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"⚠️  Error reading {file_path}: {e}")
        return False

    original = content

    for old_import, new_import in mapping.items():
        content = content.replace(old_import, new_import)

    if content != original:
        try:
            file_path.write_text(content, encoding='utf-8')
            print(f"✓ Updated: {file_path}")
            return True
        except Exception as e:
            print(f"⚠️  Error writing {file_path}: {e}")
            return False

    return False


def update_all_imports(root_dir: Path, mapping: Dict[str, str], exclude_dirs: List[str] = None) -> Tuple[int, int]:
    """Update imports in all Python files."""
    if exclude_dirs is None:
        exclude_dirs = ['__pycache__', '.git', 'venv', 'env', '.pytest_cache']

    updated = 0
    total = 0

    for py_file in root_dir.rglob("*.py"):
        # Skip excluded directories
        if any(excl in py_file.parts for excl in exclude_dirs):
            continue

        total += 1
        if update_imports_in_file(py_file, mapping):
            updated += 1

    return updated, total


def main():
    # Parse arguments
    phase = 'all'
    if len(sys.argv) > 1:
        if sys.argv[1] == '--phase' and len(sys.argv) > 2:
            phase = sys.argv[2]

    root = Path(__file__).parent.parent / "src" / "nwc_webapp"

    if not root.exists():
        print(f"❌ Error: {root} does not exist")
        sys.exit(1)

    print(f"🔧 Running import updates (phase: {phase})")
    print(f"📁 Root directory: {root}\n")

    # Select mapping based on phase
    if phase == 'utils':
        mapping = UTILS_REMOVALS
        print("Phase: Removing utils.py imports\n")
    elif phase == 'logging':
        mapping = LOGGING_UPDATES
        print("Phase: Updating logging config imports\n")
    elif phase == 'all':
        mapping = {**UTILS_REMOVALS, **LOGGING_UPDATES, **ALL_PACKAGE_RENAMES}
        print("Phase: Applying all updates\n")
    else:
        print(f"❌ Unknown phase: {phase}")
        print("Valid phases: utils, logging, all")
        sys.exit(1)

    updated, total = update_all_imports(root, mapping)

    print(f"\n✅ Complete: Updated {updated}/{total} files")

    if updated > 0:
        print("\n⚠️  Remember to:")
        print("   1. Review the changes with git diff")
        print("   2. Test the application")
        print("   3. Commit the changes if everything works")


if __name__ == "__main__":
    main()
