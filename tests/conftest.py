import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")

# Scripts in src/ import each other as top-level modules (e.g. `from chunking import ...`);
# the API is imported as the `api` package from the project root.
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
