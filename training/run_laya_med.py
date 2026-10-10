#!/usr/bin/env python3
"""Run Laya-Med baseline evaluation.

This script evaluates the Laya-Med baseline and saves results to results/laya_med
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results/laya_med"))
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()
    print("Laya-Med evaluation placeholder - implement adapter.")
    print(f"Run with: python {__file__} --max-samples 50 --append")


if __name__ == "__main__":
    main()
