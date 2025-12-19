
import sys
import os

# Set path to include package
sys.path.insert(0, "/home/scur1900/darling_lukas/verl")

print("Initial modules:", "torch" in sys.modules, "faiss" in sys.modules)

from verl.retrieval.engine.index_builder import IndexBuilder

print("After import IndexBuilder:", "torch" in sys.modules, "faiss" in sys.modules)

if "torch" in sys.modules or "faiss" in sys.modules:
    print("FAILURE: Heavy modules loaded prematurely.")
    sys.exit(1)
else:
    print("SUCCESS: Lazy loading works.")
