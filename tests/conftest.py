"""
Pytest configuration — adds the project root to sys.path so tests can import
from rag/, agents/, models/, etc. without installing the package.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
