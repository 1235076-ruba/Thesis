"""
Train SecureBERT pairwise conflict detector.

Uses the **Existing conflict** column: each non-empty cell holds the Requirement ID
that the row's requirement conflicts with.

Usage:
  python train_conflict_detector.py
  python train_conflict_detector.py Requirement_DS.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

from detect_conflicts import train_conflict_detector


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train SecureBERT conflict detector on labeled requirement pairs."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=root / "Requirement_DS.xlsx",
        help="Excel with Requirement ID, Requirement text, Existing conflict",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=root / "models" / "requirement_conflict_securebert",
        help="Directory to save the conflict detector",
    )
    parser.add_argument(
        "--encoder-dir",
        type=Path,
        default=root / "models" / "requirement_securebert",
        help="SecureBERT tokenizer/encoder directory (from train_classifier.py)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    result = train_conflict_detector(
        args.input,
        args.model_dir,
        encoder_dir=args.encoder_dir,
    )

    print(f"Saved conflict model -> {result.model_dir}")
    print(f"Mode: {result.mode}")
    print(
        f"Pairs: {result.positive_pairs} conflict + {result.negative_pairs} non-conflict"
    )
    print(f"Train accuracy: {result.train_accuracy:.3f}")
    print(f"Train F1: {result.train_f1:.3f} (threshold={result.threshold:.2f})")


if __name__ == "__main__":
    main()
