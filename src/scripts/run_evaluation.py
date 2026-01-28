#!/usr/bin/env python
"""
Run model evaluation with blind test.

Usage:
    python -m src.scripts.run_evaluation --model outputs_1.7b --test-data data/raw/train_final1.jsonl
"""
import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data import DataLoader
from src.training import ModelEvaluator
from src.visualization import ReportGenerator
from src.config.settings import OUTPUTS_DIR, DataFiles, DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="Run model evaluation")
    parser.add_argument(
        "--model",
        type=str,
        default=str(OUTPUTS_DIR),
        help="Path to model"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=str(DataFiles.TEST_DATA),
        help="Path to test data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATA_DIR / "evaluation_result.md"),
        help="Path to save report"
    )
    args = parser.parse_args()

    # Load test data
    print(f"Loading test data: {args.test_data}")
    test_data = DataLoader.load_jsonl(Path(args.test_data))
    print(f"  Loaded {len(test_data)} samples")

    # Run evaluation
    print(f"\nRunning evaluation with model: {args.model}")
    evaluator = ModelEvaluator(model_path=args.model)
    results = evaluator.evaluate(test_data)

    # Print summary
    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Total: {results['total']}")
    print(f"Correct: {results['correct']}")
    print(f"Accuracy: {results['accuracy']:.1f}%")
    print(f"\nCategory Accuracy:")
    for cat, data in sorted(results['category_accuracy'].items()):
        print(f"  {cat}: {data['accuracy']:.1f}% ({data['correct']}/{data['total']})")

    # Generate report
    print(f"\nGenerating report: {args.output}")
    ReportGenerator.generate_blind_test_report(
        results,
        Path(args.output),
        model_name=args.model
    )

    # Generate error analysis
    error_path = Path(args.output).parent / "error_analysis.md"
    ReportGenerator.generate_error_analysis(results, error_path)
    print(f"Error analysis saved: {error_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
