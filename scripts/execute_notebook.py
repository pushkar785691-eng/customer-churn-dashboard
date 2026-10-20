"""Execute the portfolio notebook in place as a reproducibility check."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient


path = Path("notebooks/customer_churn_analysis.ipynb")
notebook = nbformat.read(path, as_version=4)
client = NotebookClient(
    notebook,
    timeout=600,
    kernel_name="python3",
    resources={"metadata": {"path": str(Path.cwd())}},
)
client.execute()
nbformat.write(notebook, path)
print(f"Executed and saved {path}")

