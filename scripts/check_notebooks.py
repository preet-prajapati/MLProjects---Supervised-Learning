"""Validate notebook format and syntax; optionally execute from a fresh kernel."""

import argparse
import ast

import nbformat
from nbclient import NotebookClient

from ml_portfolio.data import PROJECTS, ROOT
from ml_portfolio.evaluate import write_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", choices=["all", "bundled", *PROJECTS])
    parser.add_argument(
        "--in-place", action="store_true", help="Save verified outputs in source notebooks"
    )
    args = parser.parse_args()
    for project in PROJECTS:
        path = ROOT / "projects" / project / "notebooks" / "analysis.ipynb"
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
        execute = (
            args.execute == "all"
            or args.execute == project
            or (args.execute == "bundled" and project != "california-housing")
        )
        if execute:
            print(f"Executing {project}...", flush=True)
            NotebookClient(
                nb,
                timeout=1800,
                kernel_name="python3",
                resources={"metadata": {"path": str(path.parent)}},
            ).execute()
            destination = (
                path if args.in_place else ROOT / "reports" / "executed" / f"{project}.ipynb"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            nbformat.write(nb, destination)
        print(f"OK: {project}", flush=True)
    if args.execute:
        write_summary(ROOT / "reports")


if __name__ == "__main__":
    main()
