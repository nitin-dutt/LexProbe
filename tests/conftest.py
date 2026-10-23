import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")

# Scripts in src/ import each other as top-level modules (e.g. `from chunking import ...`);
# the API is imported as the `api` package from the project root.
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

# The API tests use the 6-clause blend runtime with a fake model. The default API mode is "single" (v6b, 7 clauses,
# dataset profile v4), which must be started in its own process (see tests/test_single_model.py).
os.environ.setdefault("SCORING", "blend")
