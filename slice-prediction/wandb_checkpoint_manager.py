"""
wandb_checkpoint_manager.py

Simple CLI utility to list and download model checkpoint artifacts from Weights &
Biases. This is a convenience tool — it uses the wandb.Api to query artifacts and
download files.

Commands:
  list --project <proj> [--entity <ent>]
  download --project <proj> --artifact-name <name> [--entity <ent>] [--dest .]
  download_best --project <proj> [--entity <ent>] [--dest .]
  inspect <local_checkpoint>

Examples:
  python wandb_checkpoint_manager.py list --project my-proj
  python wandb_checkpoint_manager.py download_best --project my-proj --dest ./ckpts

Note: requires WANDB_API_KEY env var or configured CLI.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import wandb


def list_artifacts(project: str, entity: Optional[str] = None, type_filter: Optional[str] = None):
    api = wandb.Api()
    proj = f"{entity}/{project}" if entity else project
    print(f"Listing artifacts for project: {proj}")
    # wandb.Api().artifacts(project=proj) is expensive; use run search instead
    # We'll iterate over runs and collect artifacts
    runs = api.runs(proj)
    artifacts = []
    for r in runs:
        for a in r.logged_artifacts():
            if type_filter and a.type != type_filter:
                continue
            artifacts.append((r.id, r.name, a))

    for run_id, run_name, art in artifacts:
        print(f"run: {run_name} ({run_id})  artifact: {art.name}:{art.version}  type: {art.type}")


def download_artifact(project: str, artifact_name: str, entity: Optional[str] = None, dest: str = "."):
    api = wandb.Api()
    proj = f"{entity}/{project}" if entity else project
    print(f"Searching for artifact {artifact_name} in project {proj}")
    runs = api.runs(proj)
    for r in runs:
        for a in r.logged_artifacts():
            if artifact_name in a.name:
                print(f"Found artifact: {a.name}:{a.version} from run {r.name}")
                path = a.download(root=dest)
                print(f"Downloaded to: {path}")
                return str(path)
    raise RuntimeError(f"Artifact {artifact_name} not found in project {proj}")


def download_best(project: str, entity: Optional[str] = None, dest: str = "."):
    api = wandb.Api()
    proj = f"{entity}/{project}" if entity else project
    print(f"Looking for artifacts tagged 'best' in project {proj}")
    runs = api.runs(proj)
    for r in runs:
        for a in r.logged_artifacts():
            # artifact aliases are accessible via a.aliases
            try:
                aliases = a.aliases
            except Exception:
                aliases = []
            if any("best" in alias for alias in aliases):
                print(f"Found best artifact: {a.name}:{a.version} from run {r.name}")
                path = a.download(root=dest)
                print(f"Downloaded to: {path}")
                return str(path)
    raise RuntimeError(f"No artifact with alias 'best' found in project {proj}")


def inspect_local_checkpoint(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # attempt to read torch checkpoint metadata without requiring torch if possible
    try:
        import torch

        ckpt = torch.load(path, map_location="cpu")
        print(json.dumps({k: (type(v).__name__ if not isinstance(v, (int, float, str, list, dict)) else v) for k, v in ckpt.items()}, indent=2))
    except Exception:
        print("Could not load with torch; attempting to show file size and basic info.")
        st = os.stat(path)
        print(f"path: {path}\nsize: {st.st_size} bytes\nmtime: {st.st_mtime}")


def main():
    parser = argparse.ArgumentParser(prog="wandb_checkpoint_manager")
    sub = parser.add_subparsers(dest="cmd")

    p_list = sub.add_parser("list")
    p_list.add_argument("--project", default=None,
                        help="W&B project name. If omitted, will use WANDB_PROJECT env var")
    p_list.add_argument("--entity", default=None,
                        help="W&B entity name. If omitted, will use WANDB_ENTITY env var")
    p_list.add_argument("--type", default=None)

    p_download = sub.add_parser("download")
    p_download.add_argument("--project", default=None,
                            help="W&B project name. If omitted, will use WANDB_PROJECT env var")
    p_download.add_argument("--artifact-name", required=True)
    p_download.add_argument("--entity", default=None,
                            help="W&B entity name. If omitted, will use WANDB_ENTITY env var")
    p_download.add_argument("--dest", default=".")

    p_best = sub.add_parser("download_best")
    p_best.add_argument("--project", default=None,
                        help="W&B project name. If omitted, will use WANDB_PROJECT env var")
    p_best.add_argument("--entity", default=None,
                        help="W&B entity name. If omitted, will use WANDB_ENTITY env var")
    p_best.add_argument("--dest", default=".")

    p_inspect = sub.add_parser("inspect")
    p_inspect.add_argument("path", help="Local checkpoint file to inspect")

    args = parser.parse_args()
    # Resolve project/entity from args or environment variables
    args.project = args.project or os.getenv("WANDB_PROJECT")
    args.entity = args.entity or os.getenv("WANDB_ENTITY")
    if args.cmd == "list":
        if not args.project:
            parser.error("--project or WANDB_PROJECT env var must be provided")
        list_artifacts(args.project, args.entity, args.type)
    elif args.cmd == "download":
        if not args.project:
            parser.error("--project or WANDB_PROJECT env var must be provided")
        download_artifact(args.project, args.artifact_name, args.entity, args.dest)
    elif args.cmd == "download_best":
        if not args.project:
            parser.error("--project or WANDB_PROJECT env var must be provided")
        download_best(args.project, args.entity, args.dest)
    elif args.cmd == "inspect":
        inspect_local_checkpoint(args.path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
