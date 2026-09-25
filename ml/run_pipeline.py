#!/usr/bin/env python3
"""
Run the full ML pipeline: collect data → train.

Usage:
    python -m ml.run_pipeline
    python -m ml.run_pipeline --skip-collect   # train only (CSV exists)
    python -m ml.run_pipeline --collect-only
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="F1 ML full pipeline")
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    if not args.skip_collect:
        run([sys.executable, "-m", "ml.collect_data"])

    if args.collect_only:
        return

    train_cmd = [sys.executable, "-m", "ml.train"]
    if args.epochs:
        train_cmd.extend(["--epochs", str(args.epochs)])
    run(train_cmd)

    print("\nPipeline complete. Start the app and open /predictions")


if __name__ == "__main__":
    main()
