"""pytest configuration — ensure the project root is on sys.path."""
import sys
from pathlib import Path

# Add the workspace root so `import src.xxx` works from the tests/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
