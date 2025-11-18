"""Parallel scheduling script for running workflow/main.py in parallel on a single machine.

Purpose: Split JSON files in input directory into chunks and run `workflow/main.py` in parallel.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


def _chunk_list(items: List[Path], chunks: int) -> List[List[Path]]:
    if chunks <= 0:
        raise ValueError("chunks must be a positive integer")
    if not items:
        return [[]]

    chunk_size = math.ceil(len(items) / chunks)
    return [items[i * chunk_size : (i + 1) * chunk_size] for i in range(chunks) if items[i * chunk_size : (i + 1) * chunk_size]]


def _ensure_empty_dir(path: Path) -> None:
    if path.exists():
        if any(path.iterdir()):
            raise ValueError(f"Directory {path} already exists and is not empty, please clean or change the path to avoid overwriting")
    else:
        path.mkdir(parents=True, exist_ok=True)


def _create_symlinks(source_files: Iterable[Path], destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for file_path in source_files:
        link_path = destination_dir / file_path.name
        if link_path.exists():
            continue
        os.symlink(file_path.resolve(), link_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split input files into multiple subdirectories and run workflow/main.py in parallel")
    parser.add_argument("--input-dir", required=True, help="Directory containing JSON input files")
    parser.add_argument("--output-root", required=True, help="Output root directory, different sub-tasks will write to sub-folders")
    parser.add_argument("--chunks", type=int, default=4, help="Number of parallel sub-tasks")
    parser.add_argument("--model-path", required=True, help="Local Gemma model path")
    parser.add_argument("--torch-dtype", default="bfloat16", help="Argument --torch_dtype passed to main.py")
    parser.add_argument("--device-map", default="auto", help="Argument --device_map passed to main.py")
    parser.add_argument("--attn-implementation", default=None, help="Argument --attn_implementation passed to main.py")
    parser.add_argument("--batch-size", type=int, default=1, help="Argument --batch_size passed to main.py")
    parser.add_argument("--max-retry", type=int, default=2, help="Argument --max_retry passed to main.py")
    parser.add_argument("--memory-use-count", type=int, default=2, help="Argument --memory_use_count passed to main.py")
    parser.add_argument("--memory-buffer-size", type=int, default=10, help="Argument --memory_buffer_size passed to main.py")
    parser.add_argument("--cuda-device", default="0", help="Set value of CUDA_VISIBLE_DEVICES")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, help="Pass other arguments to main.py")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root_dir = Path(__file__).resolve().parent
    main_py = root_dir / "workflow" / "main.py"
    if not main_py.exists():
        raise FileNotFoundError(f"workflow/main.py not found (at {main_py})")

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in directory {input_dir}")

    chunks = _chunk_list(json_files, args.chunks)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    temp_input_root = output_root / "input_chunks"
    temp_input_root.mkdir(parents=True, exist_ok=True)

    processes = []
    for index, chunk_files in enumerate(chunks):
        if not chunk_files:
            continue

        chunk_input_dir = temp_input_root / f"chunk_{index:02d}"
        chunk_output_dir = output_root / f"chunk_{index:02d}"

        _ensure_empty_dir(chunk_input_dir)
        _ensure_empty_dir(chunk_output_dir)
        _create_symlinks(chunk_files, chunk_input_dir)

        cmd = [
            sys.executable,
            str(main_py),
            "--llm_provider",
            "gemma_local",
            "--model_path",
            args.model_path,
            "--torch_dtype",
            args.torch_dtype,
            "--device_map",
            args.device_map,
            "--batch_size",
            str(args.batch_size),
            "--max_retry",
            str(args.max_retry),
            "--memory_use_count",
            str(args.memory_use_count),
            "--memory_buffer_size",
            str(args.memory_buffer_size),
            "--input_path",
            str(chunk_input_dir),
            "--output_path",
            str(chunk_output_dir),
        ]

        if args.attn_implementation:
            cmd.extend(["--attn_implementation", args.attn_implementation])

        if args.extra_args:
            cmd.extend(args.extra_args)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_device

        process = subprocess.Popen(cmd, env=env)
        processes.append((index, process, cmd, chunk_output_dir))
        print(f"[chunk {index:02d}] Start command: {' '.join(cmd)}")

    exit_code = 0
    for index, process, cmd, out_dir in processes:
        ret = process.wait()
        if ret != 0:
            exit_code = ret
            print(f"[chunk {index:02d}] Process exit code {ret}, please check output directory {out_dir} or log")
        else:
            print(f"[chunk {index:02d}] Completed, results located in {out_dir}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

