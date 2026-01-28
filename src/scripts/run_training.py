#!/usr/bin/env python
"""
Run LoRA fine-tuning training.

Usage:
    python -m src.scripts.run_training --data data/train_final.jsonl --epochs 3
"""
import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training import LoRATrainer
from src.config.settings import Models, OUTPUTS_DIR, DataFiles


def main():
    parser = argparse.ArgumentParser(description="Run LoRA fine-tuning")
    parser.add_argument(
        "--model",
        type=str,
        default=Models.QWEN3_1_7B,
        help="Base model name"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DataFiles.TRAIN_FINAL),
        help="Path to training data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUTS_DIR),
        help="Output directory for checkpoints"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lora-rank", type=int, default=64, help="LoRA rank")
    args = parser.parse_args()

    print("="*50)
    print("LoRA Fine-tuning")
    print("="*50)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"LR: {args.lr}")
    print(f"Batch: {args.batch_size}")
    print(f"LoRA Rank: {args.lora_rank}")
    print("="*50)

    # Initialize trainer
    trainer = LoRATrainer(
        model_name=args.model,
        output_dir=Path(args.output),
        lora_rank=args.lora_rank,
    )

    # Load model
    trainer.load_model()

    # Prepare dataset
    trainer.prepare_dataset(Path(args.data))

    # Train
    trainer.train(
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
    )

    print("\nTraining complete!")
    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
