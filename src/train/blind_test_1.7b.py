"""
Qwen3-1.7B 블라인드 테스트 스크립트 (blind_test_1.7b.py)

페르소나 없이 물건 목록만으로 카테고리를 정확히 맞추는지 테스트합니다.
- 입력: 물건목록 + 가격 + 빈도 (페르소나 제외)
- 출력: Markdown 형식 (노션 붙여넣기용)
"""

import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from unsloth import FastLanguageModel

# ==========================================
# 설정
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "outputs_1.7b"
TEST_DATA_PATH = PROJECT_ROOT / "data/raw/train_final1.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data/blind_test_result_all.md"
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

# Instruction (블라인드 버전)
INSTRUCTION = "물건 목록과 구매 패턴을 분석하여 적합한 모임 카테고리를 추천하세요."


def load_test_data(path: Path) -> list:
    """테스트 데이터 로드"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            output = json.loads(entry['output'])

            # 원본 input에서 정보 추출
            original_input = entry['input']

            # 페르소나 추출
            persona_match = re.search(r'페르소나:\s*([^,]+)', original_input)
            persona = persona_match.group(1).strip() if persona_match else "알 수 없음"

            # 물건목록 추출
            items_match = re.search(r'물건목록:\s*\[([^\]]+)\]', original_input)
            items = items_match.group(1).strip() if items_match else ""

            # 평균가격 추출
            price_match = re.search(r'평균가격:\s*(\S+)', original_input)
            price = price_match.group(1).strip() if price_match else ""

            # 촬영빈도 추출
            freq_match = re.search(r'촬영빈도:\s*(\S+)', original_input)
            frequency = freq_match.group(1).strip() if freq_match else ""

            data.append({
                'persona': persona,  # 숨길 정보
                'items': items,
                'price': price,
                'frequency': frequency,
                'ground_truth': output['category'],
                'original_input': original_input
            })
    return data


def create_blind_input(item: dict) -> str:
    """페르소나 제외한 블라인드 입력 생성"""
    return f"물건목록: [{item['items']}], 평균가격: {item['price']}, 촬영빈도: {item['frequency']}"


def parse_category(output_str: str) -> str:
    """모델 출력에서 category 추출"""
    try:
        obj = json.loads(output_str)
        return obj.get('category', '')
    except json.JSONDecodeError:
        pass

    start = output_str.find('{')
    end = output_str.rfind('}')
    if start != -1 and end != -1:
        try:
            obj = json.loads(output_str[start:end+1])
            return obj.get('category', '')
        except json.JSONDecodeError:
            pass

    match = re.search(r'"category"\s*:\s*"([^"]+)"', output_str)
    if match:
        return match.group(1)

    return ''


def truncate_items(items: str, max_len: int = 30) -> str:
    """물건 목록을 적절한 길이로 자르기"""
    if len(items) <= max_len:
        return items
    return items[:max_len] + "..."


def main():
    print("블라인드 테스트 시작...")

    # 경로 확인
    if not MODEL_PATH.exists():
        print(f"[ERROR] 모델이 존재하지 않습니다: {MODEL_PATH}")
        return

    if not TEST_DATA_PATH.exists():
        print(f"[ERROR] 테스트 데이터가 존재하지 않습니다: {TEST_DATA_PATH}")
        return

    # 테스트 데이터 로드
    print(f"테스트 데이터 로드: {TEST_DATA_PATH}")
    test_data = load_test_data(TEST_DATA_PATH)
    print(f"  총 {len(test_data)}개 샘플")

    # 모델 로드
    print(f"모델 로드: {MODEL_PATH}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(MODEL_PATH),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print("  모델 로드 완료!")

    # 평가 실행
    print("블라인드 테스트 실행 중...")
    results = []
    correct = 0
    total = 0
    per_category_correct = defaultdict(int)
    per_category_total = defaultdict(int)

    for i, item in enumerate(test_data):
        if (i + 1) % 10 == 0:
            print(f"  진행: {i + 1}/{len(test_data)}")

        # 블라인드 입력 생성
        blind_input = create_blind_input(item)

        # Qwen3 Chat Template
        messages = [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": blind_input}
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

        predicted = parse_category(generated)
        ground_truth = item['ground_truth']

        total += 1
        per_category_total[ground_truth] += 1
        is_correct = predicted == ground_truth

        if is_correct:
            correct += 1
            per_category_correct[ground_truth] += 1

        results.append({
            'idx': i + 1,
            'items': item['items'],
            'price': item['price'],
            'frequency': item['frequency'],
            'predicted': predicted,
            'ground_truth': ground_truth,
            'is_correct': is_correct,
            'persona': item['persona'],  # 마지막에 공개
        })

    # Markdown 출력 생성
    accuracy = correct / total * 100 if total > 0 else 0
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    md_output = []
    md_output.append(f"# 블라인드 테스트 결과\n")
    md_output.append(f"> 테스트 일시: {now}\n")
    md_output.append(f"> 모델: `{MODEL_PATH.name}`\n")
    md_output.append(f"> 테스트 방식: **페르소나 제외**, 물건목록 + 가격 + 빈도만 입력\n")
    md_output.append("")

    # 요약
    md_output.append("## 요약\n")
    md_output.append(f"| 항목 | 값 |")
    md_output.append(f"|------|-----|")
    md_output.append(f"| 총 테스트 | {total}개 |")
    md_output.append(f"| 정답 | {correct}개 |")
    md_output.append(f"| 오답 | {total - correct}개 |")
    md_output.append(f"| **정확도** | **{accuracy:.1f}%** |")
    md_output.append("")

    # Category별 정확도
    md_output.append("## Category별 정확도\n")
    md_output.append("| Category | 정답/전체 | 정확도 |")
    md_output.append("|----------|----------|--------|")
    for cat in sorted(per_category_total.keys()):
        cat_correct = per_category_correct[cat]
        cat_total = per_category_total[cat]
        cat_acc = cat_correct / cat_total * 100 if cat_total > 0 else 0
        md_output.append(f"| {cat} | {cat_correct}/{cat_total} | {cat_acc:.1f}% |")
    md_output.append("")

    # 상세 결과
    md_output.append("## 상세 결과\n")
    md_output.append("| # | 물건목록 | 가격 | 빈도 | 예측 | 정답 | 결과 | 페르소나 (숨김) |")
    md_output.append("|---|---------|------|------|------|------|------|----------------|")

    for r in results:
        items_short = truncate_items(r['items'], 25)
        result_emoji = "✅" if r['is_correct'] else "❌"
        md_output.append(
            f"| {r['idx']} | {items_short} | {r['price']} | {r['frequency']} | "
            f"{r['predicted']} | {r['ground_truth']} | {result_emoji} | {r['persona']} |"
        )
    md_output.append("")

    # 오답 분석
    errors = [r for r in results if not r['is_correct']]
    if errors:
        md_output.append("## 오답 분석\n")
        md_output.append("| # | 물건목록 | 예측 | 정답 | 페르소나 |")
        md_output.append("|---|---------|------|------|---------|")
        for r in errors:
            items_short = truncate_items(r['items'], 40)
            md_output.append(
                f"| {r['idx']} | {items_short} | {r['predicted']} | {r['ground_truth']} | {r['persona']} |"
            )
        md_output.append("")

    # 파일 저장
    md_content = "\n".join(md_output)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(md_content)

    # 콘솔 출력
    print("\n" + "=" * 60)
    print(md_content)
    print("=" * 60)
    print(f"\n결과 저장됨: {OUTPUT_PATH}")
    print("위 내용을 복사하여 노션에 붙여넣으세요!")


if __name__ == "__main__":
    main()
