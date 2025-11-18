"""Script to download and prepare Gemma model weights."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download google/gemma-3-1b-it model to local specified directory"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="google/gemma-3-1b-it",
        help="Hugging Face repository ID, default is google/gemma-3-1b-it",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Specify model version/branch, default is latest version",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/gemma-3-1b-it",
        help="Model save path, default is ./models/gemma-3-1b-it",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token for restricted repositories, optional",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Only use locally cached files, do not access network",
    )
    parser.add_argument(
        "--allow_patterns",
        type=str,
        nargs="*",
        default=None,
        help="Allowed file pattern list, e.g. *.safetensors",
    )
    parser.add_argument(
        "--ignore_patterns",
        type=str,
        nargs="*",
        default=None,
        help="Ignored file pattern list",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Downloading repository: {args.repo_id}")
    print(f"[INFO] Saving path: {output_dir}")

    download_kwargs = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "local_dir": str(output_dir),
        "local_dir_use_symlinks": False,
        "allow_patterns": args.allow_patterns,
        "ignore_patterns": args.ignore_patterns,
        "local_files_only": args.local_files_only,
    }

    if args.token:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = args.token

    snapshot_download(**download_kwargs)

    print("[INFO] Model downloaded, can use --model_path in workflow/main.py to specify this directory.")


if __name__ == "__main__":
    main()
