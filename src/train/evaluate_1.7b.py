"""
Qwen3-1.7B 평가 스크립트 (evaluate_1.7b.py)

별도 테스트셋으로 모델 정확도를 측정합니다.
- 테스트 데이터: data/raw/train_final1.jsonl
- 평가 지표: 전체 정확도, Category별 정확도, 혼동 행렬
"""

import json
from pathlib import Path
from collections import defaultdict
from unsloth import FastLanguageModel

# ==========================================
# 설정
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "outputs_1.7b"
TEST_DATA_PATH = PROJECT_ROOT / "data/raw/train_final1.jsonl"
MAX_SEQ_LENGTH = 2048

# Non-thinking Mode 추론 설정
GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0,
    "max_new_tokens": 512,
    "use_cache": True,
}

# Instruction
INSTRUCTION = "유저의 촬영 물건과 패턴을 분석하여 모임 카테고리와 분위기를 추천하세요."


def load_test_data(path: Path) -> list:
    """테스트 데이터 로드"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            # 정답 추출
            output = json.loads(entry['output'])
            data.append({
                'input': entry['input'],
                'ground_truth': output['category'],
                'full_output': output
            })
    return data


def parse_category(output_str: str) -> str:
    """모델 출력에서 category 추출"""
    try:
        # 직접 파싱 시도
        obj = json.loads(output_str)
        return obj.get('category', '')
    except json.JSONDecodeError:
        pass

    # { } 블록 찾기
    start = output_str.find('{')
    end = output_str.rfind('}')
    if start != -1 and end != -1:
        try:
            obj = json.loads(output_str[start:end+1])
            return obj.get('category', '')
        except json.JSONDecodeError:
            pass

    # "category": "..." 패턴 찾기
    import re
    match = re.search(r'"category"\s*:\s*"([^"]+)"', output_str)
    if match:
        return match.group(1)

    return ''


def print_confusion_matrix(confusion: dict, categories: list):
    """혼동 행렬 출력"""
    print("\n[혼동 행렬]")
    print("(행: 정답, 열: 예측)")
    print()

    # 헤더
    header = "정답\\예측".ljust(12)
    for cat in categories:
        header += cat[:6].center(8)
    print(header)
    print("-" * (12 + 8 * len(categories)))

    # 행
    for true_cat in categories:
        row = true_cat[:10].ljust(12)
        for pred_cat in categories:
            count = confusion.get((true_cat, pred_cat), 0)
            row += str(count).center(8)
        print(row)


def main():
    print("=" * 60)
    print("Qwen3-1.7B 모델 평가")
    print("=" * 60)

    # 경로 확인
    if not MODEL_PATH.exists():
        print(f"[ERROR] 모델이 존재하지 않습니다: {MODEL_PATH}")
        return

    if not TEST_DATA_PATH.exists():
        print(f"[ERROR] 테스트 데이터가 존재하지 않습니다: {TEST_DATA_PATH}")
        return

    # ==========================================
    # 1. 테스트 데이터 로드
    # ==========================================
    print(f"\n[1/3] 테스트 데이터 로드: {TEST_DATA_PATH}")
    test_data = load_test_data(TEST_DATA_PATH)
    print(f"  총 {len(test_data)}개 샘플")

    # Category 분포 확인
    category_counts = defaultdict(int)
    for item in test_data:
        category_counts[item['ground_truth']] += 1

    print("\n  [정답 Category 분포]")
    for cat, count in sorted(category_counts.items()):
        print(f"    {cat}: {count}개")

    # ==========================================
    # 2. 모델 로드
    # ==========================================
    print(f"\n[2/3] 모델 로드: {MODEL_PATH}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(MODEL_PATH),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print("  모델 로드 완료!")

    # ==========================================
    # 3. 평가 실행
    # ==========================================
    print(f"\n[3/3] 평가 실행 중...")

    correct = 0
    total = 0
    per_category_correct = defaultdict(int)
    per_category_total = defaultdict(int)
    confusion = defaultdict(int)  # (true, pred) -> count
    errors = []  # 틀린 케이스 저장

    for i, item in enumerate(test_data):
        # 진행률 표시
        if (i + 1) % 10 == 0:
            print(f"  진행: {i + 1}/{len(test_data)}")

        # Qwen3 Chat Template 구성
        messages = [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": item['input']}
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, **GENERATION_CONFIG)

        generated = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # Category 추출
        predicted = parse_category(generated)
        ground_truth = item['ground_truth']

        # 집계
        total += 1
        per_category_total[ground_truth] += 1
        confusion[(ground_truth, predicted)] += 1

        if predicted == ground_truth:
            correct += 1
            per_category_correct[ground_truth] += 1
        else:
            errors.append({
                'input': item['input'][:100] + '...',
                'ground_truth': ground_truth,
                'predicted': predicted,
                'raw_output': generated[:200]
            })

    # ==========================================
    # 4. 결과 출력
    # ==========================================
    print("\n" + "=" * 60)
    print("평가 결과")
    print("=" * 60)

    # 전체 정확도
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n[전체 정확도]")
    print(f"  {correct}/{total} = {accuracy:.1f}%")

    # Category별 정확도
    print(f"\n[Category별 정확도]")
    categories = sorted(per_category_total.keys())
    for cat in categories:
        cat_correct = per_category_correct[cat]
        cat_total = per_category_total[cat]
        cat_acc = cat_correct / cat_total * 100 if cat_total > 0 else 0
        print(f"  {cat}: {cat_correct}/{cat_total} = {cat_acc:.1f}%")

    # 혼동 행렬
    all_categories = sorted(set(
        [k[0] for k in confusion.keys()] +
        [k[1] for k in confusion.keys() if k[1]]
    ))
    print_confusion_matrix(confusion, all_categories)

    # 오류 케이스 샘플
    if errors:
        print(f"\n[오류 케이스 샘플] (최대 5개)")
        for err in errors[:5]:
            print(f"\n  Input: {err['input']}")
            print(f"  정답: {err['ground_truth']}")
            print(f"  예측: {err['predicted']}")
            print(f"  Raw: {err['raw_output'][:100]}...")
            print("-" * 40)

    print("\n" + "=" * 60)
    print("평가 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
