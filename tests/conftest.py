import os
import sys

# Scripts in src/ import each other as top-level modules (e.g. `from chunking import ...`)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
