"""
추가 데이터 생성 스크립트 (gen_supplement.py)

Qwen3-14B를 사용하여 약점 보완용 Hard Case 데이터를 생성합니다.
- Model: unsloth/Qwen3-14B-unsloth-bnb-4bit
- Mode: Thinking Mode (기본값)
- 목표: 300개 Hard Case 생성
"""

import torch
from unsloth import FastLanguageModel
import json
import os
import gc
from tqdm import tqdm
import re

# ==========================================
# 1. 설정 (Configuration) - 16GB VRAM 최적화
# ==========================================
MODEL_ID = "unsloth/Qwen3-14B-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 2048  # Thinking Mode는 토큰이 더 필요
OUTPUT_FILE = "data/supplement_data_300.jsonl"

# 생성 설정 (Thinking Mode 권장)
GENERATION_CONFIG = {
    "max_new_tokens": 2048,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "use_cache": True,
}

# ==========================================
# 2. 타겟 프롬프트 설계 (약점 보완용)
# ==========================================
SYSTEM_PROMPT = """당신은 '당근마켓 구매 패턴 생성기'입니다.
현재 AI 모델이 '취미', '자기계발', '육아', '반려동물'을 서로 헷갈려하고 있습니다.
이 경계를 명확히 구분할 수 있는 **'어려운(Hard)' 케이스**를 생성해야 합니다.

다음 3가지 유형의 데이터를 균형 있게 생성하세요 (JSON 형식):

1. **[유형 A: 자기계발 vs 취미]**
   - 겉보기엔 취미 같지만, 실제로는 '자격증', '취업', '부업', '대회' 목적이 뚜렷한 물건 목록.
   - 예: (바리스타 자격증 책 + 앞치마 -> 자기계발), (GTQ 수험서 + 포토샵 -> 자기계발)

2. **[유형 B: 육아 vs 취미/살림]**
   - 요리나 만들기를 하지만, 목적이 철저히 '아이를 위한' 것인 목록.
   - 예: (유기농 밀가루 + 캐릭터 쿠키틀 + 아이 간식 포장지 -> 육아)

3. **[유형 C: 특수 반려동물]**
   - 개/고양이가 아닌 곤충, 파충류, 소동물, 물고기 관련 용품.
   - 예: (밀웜 + 톱밥 + 사육통 -> 반려동물)

**출력 형식 (JSON Only, 한 줄씩):**
{{"instruction": "유저의 촬영 물건과 패턴을 분석하여 모임 카테고리와 분위기를 추천하세요.", "input": "페르소나: [페르소나], 물건목록: [[물건1], [물건2], ...], 평균가격: [가격대], 촬영빈도: [빈도]", "output": "{{\\"category\\":\\"[카테고리]\\",\\"hard_negative\\":{{\\"confusing\\":\\"[혼동카테고리]\\",\\"reason\\":\\"[왜 혼동될 수 있는지 설명]\\"}}}}"}}

위 형식의 JSON을 정확히 {count}개 생성해주세요. 각 JSON은 별도의 줄에 출력하세요."""


def load_model():
    """Qwen3-14B 모델 로드"""
    print(f"🔄 모델 로딩 중... ({MODEL_ID})")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # RTX 40시리즈는 자동으로 bfloat16 적용
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)
    print("✅ 모델 로드 완료!")

    return model, tokenizer


def parse_thinking_output(raw_output: str) -> str:
    """Thinking Mode 출력에서 <think>...</think> 제거 후 최종 응답 추출"""
    if "</think>" in raw_output:
        parts = raw_output.split("</think>")
        return parts[-1].strip()
    return raw_output.strip()


def extract_json_lines(text: str) -> list:
    """텍스트에서 유효한 JSON 라인들 추출"""
    results = []

    # 코드 블록 제거
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)

    # 줄 단위로 처리
    for line in text.split('\n'):
        line = line.strip()
        if not line or not line.startswith('{'):
            continue

        try:
            obj = json.loads(line)
            # 필수 필드 확인
            if 'instruction' in obj and 'input' in obj and 'output' in obj:
                results.append(obj)
        except json.JSONDecodeError:
            # JSON 블록 추출 시도
            start = line.find('{')
            end = line.rfind('}')
            if start != -1 and end != -1:
                try:
                    obj = json.loads(line[start:end+1])
                    if 'instruction' in obj and 'input' in obj and 'output' in obj:
                        results.append(obj)
                except json.JSONDecodeError:
                    continue

    return results


def generate_batch(model, tokenizer, count: int = 5) -> list:
    """배치 단위로 데이터 생성"""
    user_prompt = SYSTEM_PROMPT.format(count=count)

    messages = [
        {"role": "system", "content": "You are a helpful data generation assistant."},
        {"role": "user", "content": user_prompt}
    ]

    # Thinking Mode (기본값): enable_thinking 파라미터 없이 사용
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    # 생성
    outputs = model.generate(
        input_ids=inputs,
        **GENERATION_CONFIG
    )

    # 디코딩
    raw_output = tokenizer.decode(
        outputs[0][inputs.shape[1]:],
        skip_special_tokens=True
    )

    # Thinking 블록 제거
    final_output = parse_thinking_output(raw_output)

    # JSON 라인 추출
    items = extract_json_lines(final_output)

    return items


def main():
    # 출력 디렉토리 확인
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # 모델 로드
    model, tokenizer = load_model()

    # 이어하기 확인 (Resume)
    current_count = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            current_count = sum(1 for line in f if line.strip())
        print(f"📂 기존 파일 발견: {current_count}개 생성됨. 이어서 생성합니다.")

    # 설정
    total_target = 300
    batch_size = 5  # 한 번에 요청할 개수

    print(f"\n🚀 데이터 생성 시작! (목표: {total_target}개)")

    # 생성 루프
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        pbar = tqdm(total=total_target, initial=current_count, desc="생성 중")

        while current_count < total_target:
            try:
                items = generate_batch(model, tokenizer, batch_size)

                for item in items:
                    if current_count >= total_target:
                        break

                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    f.flush()
                    current_count += 1
                    pbar.update(1)

            except Exception as e:
                print(f"\n⚠️ 에러 발생 (건너뜀): {e}")
                continue

        pbar.close()

    # 메모리 정리
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n✅ 완료! 저장된 파일: {OUTPUT_FILE}")
    print(f"총 {current_count}개 데이터 생성됨")


if __name__ == "__main__":
    main()
