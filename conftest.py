import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax

# The C++ reference is double precision; float32 noise would drown the golden tolerance.
jax.config.update("jax_enable_x64", True)
