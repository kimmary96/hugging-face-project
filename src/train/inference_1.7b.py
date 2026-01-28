"""
Qwen3-1.7B 검증 스크립트 (inference_1.7b.py)

파인튜닝된 모델의 추론 성능을 검증합니다.
- Non-thinking mode (enable_thinking=False)
- Qwen3 Chat Template 사용
"""

import json
from pathlib import Path
from unsloth import FastLanguageModel

# ==========================================
# 설정
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "outputs_1.7b"
MAX_SEQ_LENGTH = 2048

# Non-thinking Mode 추론 설정 (Best Practices)
GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0,
    "max_new_tokens": 512,
    "use_cache": True,
}

# ==========================================
# 테스트 케이스
# ==========================================
TEST_CASES = [
    # Case 1: 명확한 운동 취미
    {
        "persona": "30대 남성 직장인",
        "items": ["테니스 라켓", "테니스화", "스포츠 타월", "생수", "물티슈"],
        "price": "중가",
        "frequency": "자주"
    },
    # Case 2: 혼동 가능 (육아 vs 자기계발)
    {
        "persona": "아이 체험 활동을 정책 제안서로 정리하는 50대 남성",
        "items": ["체험 키트", "사례 기록지", "제안서 초안", "근거 정리표"],
        "price": "저가",
        "frequency": "가끔"
    },
    # Case 3: 복합 의도 (반려동물 + 취미)
    {
        "persona": "반려견과 캠핑을 즐기는 30대 여성",
        "items": ["캠핑 의자", "강아지 리드줄", "텐트", "간식 파우치", "물티슈"],
        "price": "고가",
        "frequency": "가끔"
    },
    # Case 4: 육아
    {
        "persona": "첫 아이를 출산한 30대 초보 엄마",
        "items": ["아기 띠", "젖병 소독기", "유모차 모빌", "배냇저고리", "키친타월"],
        "price": "중가",
        "frequency": "자주"
    },
    # Case 5: 자기계발
    {
        "persona": "코딩 테스트를 준비하는 20대 취준생",
        "items": ["알고리즘 문제집", "듀얼 모니터", "기계식 키보드", "에너지 드링크"],
        "price": "고가",
        "frequency": "자주"
    },
]

# Instruction (학습 데이터와 동일)
INSTRUCTION = "유저의 촬영 물건과 패턴을 분석하여 모임 카테고리와 분위기를 추천하세요."


def format_input(case: dict) -> str:
    """테스트 케이스를 input 형식으로 변환"""
    return (
        f"페르소나: {case['persona']}, "
        f"물건목록: {case['items']}, "
        f"평균가격: {case['price']}, "
        f"촬영빈도: {case['frequency']}"
    )


def parse_output(output_str: str) -> dict:
    """모델 출력에서 JSON 파싱 시도"""
    # JSON 블록 추출 시도
    try:
        # 직접 파싱
        return json.loads(output_str)
    except json.JSONDecodeError:
        pass

    # { } 블록 찾기
    start = output_str.find('{')
    end = output_str.rfind('}')
    if start != -1 and end != -1:
        try:
            return json.loads(output_str[start:end+1])
        except json.JSONDecodeError:
            pass

    return None


def main():
    print("=" * 60)
    print("Qwen3-1.7B 파인튜닝 모델 검증")
    print("=" * 60)

    # 모델 경로 확인
    if not MODEL_PATH.exists():
        print(f"[ERROR] 모델이 존재하지 않습니다: {MODEL_PATH}")
        print("먼저 python src/train/train_1.7b.py를 실행하세요.")
        return

    # 추론 설정 로드 (있으면)
    config_path = MODEL_PATH / "inference_config.json"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            saved_config = json.load(f)
            print(f"추론 설정 로드됨: {config_path}")
            print(f"  Mode: {saved_config.get('mode', 'unknown')}")

    # ==========================================
    # 1. 모델 로드
    # ==========================================
    print("\n[1/2] 학습된 모델 로드 중...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(MODEL_PATH),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    # 추론 모드로 전환 (속도 2배 향상)
    FastLanguageModel.for_inference(model)
    print("  모델 로드 완료!")

    # ==========================================
    # 2. 추론 실행
    # ==========================================
    print("\n[2/2] 추론 테스트 시작...")
    print("=" * 60)

    results = []
    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n[Test Case {i}]")
        print(f"  Persona: {case['persona']}")
        print(f"  Items: {case['items']}")

        # Qwen3 Chat Template 구성
        input_text = format_input(case)
        messages = [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": input_text}
        ]

        # Non-thinking mode로 프롬프트 생성
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        # 토크나이징
        inputs = tokenizer(
            [prompt],
            return_tensors="pt",
        ).to("cuda")

        # 생성
        outputs = model.generate(
            **inputs,
            **GENERATION_CONFIG,
        )

        # 디코딩 (프롬프트 부분 제외)
        generated = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # 결과 파싱
        parsed = parse_output(generated)

        print(f"\n  [Raw Output]")
        print(f"  {generated[:500]}...")  # 처음 500자만 출력

        if parsed:
            print(f"\n  [Parsed Result]")
            print(f"    Category: {parsed.get('category', 'N/A')}")
            if 'tags' in parsed:
                print(f"    Tags: {parsed.get('tags', [])}")
            if 'reasoning' in parsed:
                print(f"    Reasoning: {parsed.get('reasoning', 'N/A')[:100]}...")
            if 'hard_negative' in parsed:
                hn = parsed['hard_negative']
                print(f"    Hard Negative:")
                print(f"      - Confusing: {hn.get('confusing', 'N/A')}")
                print(f"      - Reason: {hn.get('reason', 'N/A')[:100]}...")
        else:
            print(f"\n  [WARNING] JSON 파싱 실패")

        results.append({
            "case": case,
            "output": generated,
            "parsed": parsed
        })

        print("-" * 60)

    # ==========================================
    # 3. 결과 요약
    # ==========================================
    print("\n" + "=" * 60)
    print("검증 완료!")
    print("=" * 60)

    success_count = sum(1 for r in results if r['parsed'] is not None)
    print(f"총 테스트: {len(results)}개")
    print(f"JSON 파싱 성공: {success_count}개")
    print(f"JSON 파싱 실패: {len(results) - success_count}개")

    # 카테고리 분포
    categories = [r['parsed']['category'] for r in results if r['parsed']]
    if categories:
        print(f"\n[카테고리 분포]")
        for cat in set(categories):
            print(f"  - {cat}: {categories.count(cat)}개")


if __name__ == "__main__":
    main()
