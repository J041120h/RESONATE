#!/usr/bin/env python
"""Execute docs/tutorial.ipynb in place (working directory docs/), storing the outputs."""
import sys, time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

p = Path(__file__).resolve().parents[1] / "docs" / "tutorial.ipynb"
nb = nbformat.read(p, as_version=4)
t0 = time.time()
NotebookClient(nb, timeout=7200, kernel_name="python3", resources={"metadata": {"path": str(p.parent)}}).execute()
nbformat.write(nb, p)
print(f"executed {p} in {(time.time() - t0) / 60:.1f} min")
