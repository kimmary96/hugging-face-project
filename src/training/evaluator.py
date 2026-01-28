"""
Model evaluation and blind testing utilities.
"""
import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import CATEGORIES, TrainingConfig, OUTPUTS_DIR, DataFiles
from config.prompts import INSTRUCTION_CATEGORY_CLASSIFICATION


class ModelEvaluator:
    """
    Evaluator for category classification models.

    Supports blind testing (without persona) and full evaluation.
    """

    def __init__(
        self,
        model_path: str = None,
        max_seq_length: int = TrainingConfig.MAX_SEQ_LENGTH
    ):
        """
        Initialize evaluator.

        Args:
            model_path: Path to trained model
            max_seq_length: Maximum sequence length
        """
        self.model_path = model_path or str(OUTPUTS_DIR)
        self.max_seq_length = max_seq_length
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load model for evaluation."""
        from unsloth import FastLanguageModel

        print(f"Loading model: {self.model_path}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)
        print("Model loaded!")

    def evaluate(
        self,
        test_data: List[Dict],
        instruction: str = INSTRUCTION_CATEGORY_CLASSIFICATION
    ) -> Dict:
        """
        Run evaluation on test data.

        Args:
            test_data: List of test samples with 'input' and ground truth
            instruction: System instruction for classification

        Returns:
            Dictionary with accuracy metrics and detailed results
        """
        if self.model is None:
            self.load_model()

        results = []
        correct = 0
        total = 0
        per_category = defaultdict(lambda: {"correct": 0, "total": 0})

        for i, item in enumerate(test_data):
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{len(test_data)}")

            # Get ground truth
            ground_truth = self._extract_ground_truth(item)
            if not ground_truth:
                continue

            # Get prediction
            user_input = self._create_blind_input(item)
            predicted = self._predict(user_input, instruction)

            # Score
            total += 1
            per_category[ground_truth]["total"] += 1
            is_correct = predicted == ground_truth

            if is_correct:
                correct += 1
                per_category[ground_truth]["correct"] += 1

            results.append({
                "idx": i + 1,
                "input": user_input[:100] + "..." if len(user_input) > 100 else user_input,
                "predicted": predicted,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
            })

        accuracy = correct / total * 100 if total > 0 else 0

        # Calculate per-category accuracy
        category_accuracy = {}
        for cat in CATEGORIES:
            cat_data = per_category[cat]
            if cat_data["total"] > 0:
                cat_acc = cat_data["correct"] / cat_data["total"] * 100
                category_accuracy[cat] = {
                    "correct": cat_data["correct"],
                    "total": cat_data["total"],
                    "accuracy": cat_acc
                }

        return {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "category_accuracy": category_accuracy,
            "results": results
        }

    def _predict(self, user_input: str, instruction: str) -> str:
        """Make a single prediction."""
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_input}
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt"
        ).to("cuda")

        outputs = self.model.generate(
            input_ids=inputs,
            max_new_tokens=512,
            temperature=TrainingConfig.TEMPERATURE,
            top_p=TrainingConfig.TOP_P,
            top_k=TrainingConfig.TOP_K,
        )

        decoded = self.tokenizer.decode(
            outputs[0][inputs.shape[1]:],
            skip_special_tokens=True
        )

        return self._parse_category(decoded)

    def _parse_category(self, output: str) -> str:
        """Extract category from model output."""
        try:
            obj = json.loads(output)
            return obj.get('category', '')
        except json.JSONDecodeError:
            pass

        # Try to find JSON block
        start = output.find('{')
        end = output.rfind('}')
        if start != -1 and end != -1:
            try:
                obj = json.loads(output[start:end+1])
                return obj.get('category', '')
            except json.JSONDecodeError:
                pass

        # Try regex
        match = re.search(r'"category"\s*:\s*"([^"]+)"', output)
        if match:
            return match.group(1)

        return ''

    def _extract_ground_truth(self, item: Dict) -> Optional[str]:
        """Extract ground truth category from test item."""
        try:
            output = json.loads(item.get('output', '{}'))
            return output.get('category', '')
        except json.JSONDecodeError:
            return None

    def _create_blind_input(self, item: Dict) -> str:
        """Create blind input (without persona) from test item."""
        original = item.get('input', '')

        # Extract components
        items_match = re.search(r'물건목록:\s*\[([^\]]+)\]', original)
        price_match = re.search(r'평균가격:\s*(\S+)', original)
        freq_match = re.search(r'촬영빈도:\s*(\S+)', original)

        items = items_match.group(1) if items_match else ""
        price = price_match.group(1) if price_match else ""
        freq = freq_match.group(1) if freq_match else ""

        return f"물건목록: [{items}], 평균가격: {price}, 촬영빈도: {freq}"

    def generate_report(
        self,
        eval_results: Dict,
        output_path: Path,
        model_name: str = ""
    ):
        """
        Generate markdown report from evaluation results.

        Args:
            eval_results: Results from evaluate()
            output_path: Path to save markdown report
            model_name: Model name for report header
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        lines = [
            f"# 블라인드 테스트 결과\n",
            f"> 테스트 일시: {now}\n",
            f"> 모델: `{model_name or self.model_path}`\n",
            f"> 테스트 방식: **페르소나 제외**, 물건목록 + 가격 + 빈도만 입력\n",
            "",
            "## 요약\n",
            "| 항목 | 값 |",
            "|------|-----|",
            f"| 총 테스트 | {eval_results['total']}개 |",
            f"| 정답 | {eval_results['correct']}개 |",
            f"| 오답 | {eval_results['total'] - eval_results['correct']}개 |",
            f"| **정확도** | **{eval_results['accuracy']:.1f}%** |",
            "",
            "## Category별 정확도\n",
            "| Category | 정답/전체 | 정확도 |",
            "|----------|----------|--------|",
        ]

        for cat in sorted(eval_results['category_accuracy'].keys()):
            data = eval_results['category_accuracy'][cat]
            lines.append(
                f"| {cat} | {data['correct']}/{data['total']} | {data['accuracy']:.1f}% |"
            )

        lines.append("")

        # Save report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"Report saved to: {output_path}")
