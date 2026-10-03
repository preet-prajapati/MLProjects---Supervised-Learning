"""Repository CLI: explicit downloads, reproducible runs, no silent skips."""

import argparse

from .data import PROJECTS, ROOT, config, download_housing
from .evaluate import fit_project, save_result, write_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List all seven projects")
    commands.add_parser("download-housing", help="Download the optional housing data")
    run = commands.add_parser("run", help="Train and evaluate a project")
    run.add_argument("project", choices=[*PROJECTS, "all", "bundled"])
    run.add_argument("--output", type=str, default=str(ROOT / "reports"))
    args = parser.parse_args()
    if args.command == "list":
        for project in PROJECTS:
            print(f"{project:22} {config(project)['title']}")
    elif args.command == "download-housing":
        print(download_housing())
    else:
        from pathlib import Path

        projects = (
            PROJECTS
            if args.project == "all"
            else PROJECTS[:-1]
            if args.project == "bundled"
            else [args.project]
        )
        for project in projects:
            print(f"Training {project}...", flush=True)
            result = fit_project(project)
            save_result(result, Path(args.output) / project)
            print(result["report"]["test_metrics"], flush=True)
        write_summary(args.output)


if __name__ == "__main__":
    main()
