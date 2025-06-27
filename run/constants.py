import os
from pathlib import Path

# Project Structure
PROJECT_DIR = str(Path(__file__).resolve().parent.parent)
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")

# Paths to EDM utils for loading EDM-like models
EDM_UTILS_DIR = os.path.join(PROJECT_DIR, "src", "edm_utils")
TORCH_UTILS_DIR = os.path.join(EDM_UTILS_DIR, "torch_utils")
DNNLIB_DIR = os.path.join(EDM_UTILS_DIR, "dnnlib")
