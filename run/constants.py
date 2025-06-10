import os
from pathlib import Path

PROJECT_DIR = str(Path(__file__).resolve().parent.parent)
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")
