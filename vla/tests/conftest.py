import sys
import pathlib

# repo root on sys.path so `import transformer_flow` / `import vla.*` work under pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
