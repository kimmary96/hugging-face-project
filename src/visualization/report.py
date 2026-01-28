"""
Report generation utilities for evaluation results.
"""
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import CATEGORIES


class ReportGenerator:
    """Generate various reports from evaluation and matching results."""

    @staticmethod
    def generate_blind_test_report(
        results: Dict,
        output_path: Path,
        model_name: str = "",
        test_method: str = "페르소나 제외"
    ) -> str:
        """
        Generate markdown report for blind test results.

        Args:
            results: Evaluation results dictionary
            output_path: Path to save report
            model_name: Model name for header
            test_method: Description of test method

        Returns:
            Generated markdown content
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        lines = [
            f"# 블라인드 테스트 결과\n",
            f"> 테스트 일시: {now}\n",
            f"> 모델: `{model_name}`\n",
            f"> 테스트 방식: **{test_method}**\n",
            "",
            "## 요약\n",
            "| 항목 | 값 |",
            "|------|-----|",
            f"| 총 테스트 | {results['total']}개 |",
            f"| 정답 | {results['correct']}개 |",
            f"| 오답 | {results['total'] - results['correct']}개 |",
            f"| **정확도** | **{results['accuracy']:.1f}%** |",
            "",
        ]

        # Category accuracy
        if 'category_accuracy' in results:
            lines.extend([
                "## Category별 정확도\n",
                "| Category | 정답/전체 | 정확도 |",
                "|----------|----------|--------|",
            ])

            for cat in sorted(results['category_accuracy'].keys()):
                data = results['category_accuracy'][cat]
                lines.append(
                    f"| {cat} | {data['correct']}/{data['total']} | {data['accuracy']:.1f}% |"
                )
            lines.append("")

        content = '\n'.join(lines)

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return content

    @staticmethod
    def generate_comparison_report(
        results_list: List[Dict],
        model_names: List[str],
        output_path: Path
    ) -> str:
        """
        Generate comparison report between multiple models.

        Args:
            results_list: List of evaluation results
            model_names: List of model names
            output_path: Path to save report

        Returns:
            Generated markdown content
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        lines = [
            f"# 모델 비교 리포트\n",
            f"> 생성 일시: {now}\n",
            "",
            "## 전체 정확도 비교\n",
            "| 모델 | 정확도 | 정답/전체 |",
            "|------|--------|----------|",
        ]

        for name, results in zip(model_names, results_list):
            lines.append(
                f"| {name} | {results['accuracy']:.1f}% | {results['correct']}/{results['total']} |"
            )

        lines.append("")

        # Category comparison
        lines.extend([
            "## 카테고리별 비교\n",
            "| 카테고리 | " + " | ".join(model_names) + " |",
            "|----------|" + "|".join(["--------"] * len(model_names)) + "|",
        ])

        for cat in CATEGORIES:
            row = [cat]
            for results in results_list:
                if cat in results.get('category_accuracy', {}):
                    acc = results['category_accuracy'][cat]['accuracy']
                    row.append(f"{acc:.1f}%")
                else:
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")

        content = '\n'.join(lines)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return content

    @staticmethod
    def generate_error_analysis(
        results: Dict,
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate error analysis report.

        Args:
            results: Evaluation results with detailed results list
            output_path: Optional path to save report

        Returns:
            Generated markdown content
        """
        errors = [r for r in results.get('results', []) if not r.get('is_correct')]

        lines = [
            "# 오답 분석\n",
            f"총 오답 수: {len(errors)}개\n",
            "",
            "## 오답 목록\n",
            "| # | 입력 | 예측 | 정답 |",
            "|---|------|------|------|",
        ]

        for err in errors:
            input_short = err['input'][:40] + "..." if len(err['input']) > 40 else err['input']
            lines.append(
                f"| {err['idx']} | {input_short} | {err['predicted']} | {err['ground_truth']} |"
            )

        # Error pattern analysis
        lines.extend(["", "## 오답 패턴 분석\n"])

        confusion = {}
        for err in errors:
            key = f"{err['ground_truth']} → {err['predicted']}"
            confusion[key] = confusion.get(key, 0) + 1

        lines.extend([
            "| 정답 → 예측 | 횟수 |",
            "|------------|------|",
        ])

        for pattern, count in sorted(confusion.items(), key=lambda x: -x[1]):
            lines.append(f"| {pattern} | {count} |")

        content = '\n'.join(lines)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)

        return content
