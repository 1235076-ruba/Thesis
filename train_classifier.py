"""
Train SecureBERT requirement classifier (clear / unclear / incomplete).

Usage:
  python train_classifier.py
  python train_classifier.py Requirement_DS.xlsx --epochs 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

from classify_requirements import train_classifier


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Fine-tune SecureBERT on labeled requirements."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=root / "Requirement_DS.xlsx",
        help="Excel file with Requirement text and Classification columns",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=root / "models" / "requirement_securebert",
        help="Directory to save the fine-tuned model",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Force full fine-tune (not recommended for <100 labeled rows)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    result = train_classifier(
        args.input,
        args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        force_finetune=args.finetune,
    )

    print(f"Saved model -> {result.model_dir}")
    print(f"Mode: {result.mode}")
    print(f"Train size: {result.train_size} | Val size: {result.val_size}")
    print(f"Val accuracy: {result.val_accuracy:.3f}")
    print(f"Val F1 (macro): {result.val_f1_macro:.3f}")
    print(f"Labels: {', '.join(result.labels)}")


if __name__ == "__main__":
    main()
