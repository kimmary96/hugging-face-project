# src/task1_inference/inference_unsloth_test.py

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import json
import torch
import glob
import re
from unsloth import FastLanguageModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- [최적화 설정] ---
LLM_MODEL_ID = "unsloth/Qwen3-14B-unsloth-bnb-4bit" # 로컬에 이 모델이 있다고 가정

# 2. VRAM 절약 핵심 설정
# 14B 모델 기준: 1024(안전), 2048(권장/16GB 충분), 4096(위험)
MAX_SEQ_LENGTH = 2048
DTYPE = None # RTX 40시리즈는 자동으로 bfloat16이 적용됨 (가장 빠름)
LOAD_IN_4BIT = True # 필수: 메모리를 1/4로 줄여줌

EMBED_MODEL_ID = "BAAI/bge-m3"
INPUT_FILE = "./data/raw/dummy_users1.json"
OUTPUT_DIR = "./data/processed"
OUTPUT_BASE = "user_profiles_qwen3_result"

def get_next_output_file():
    """기존 파일을 확인하고 다음 번호의 파일명을 반환"""
    existing = glob.glob(f"{OUTPUT_DIR}/{OUTPUT_BASE}*.json")
    if not existing:
        return f"{OUTPUT_DIR}/{OUTPUT_BASE}_1.json"

    max_num = 0
    for f in existing:
        match = re.search(rf"{OUTPUT_BASE}_?(\d+)?\.json", f)
        if match:
            num = int(match.group(1)) if match.group(1) else 0
            max_num = max(max_num, num)

    return f"{OUTPUT_DIR}/{OUTPUT_BASE}_{max_num + 1}.json"

def main():
    OUTPUT_FILE = get_next_output_file()
    print(f">>> 출력 파일: {OUTPUT_FILE}")
    print(f">>> [1/3] Unsloth Qwen 3 로드 중... ({LLM_MODEL_ID})")

    # 모델 & 토크나이저 로드
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = LLM_MODEL_ID,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
        fix_tokenizer = True,
    )

    FastLanguageModel.for_inference(model)

    # 임베딩 모델 로드
    print(f">>> [2/3] 임베딩 모델 로드 중... ({EMBED_MODEL_ID})")
    embed_model = SentenceTransformer(EMBED_MODEL_ID)

    # 입력 데이터 로드
    print(f">>> 입력 데이터 로드 중... ({INPUT_FILE})")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        users = json.load(f)

# [수정 포인트 1] 토큰 제한 대폭 상향 (생각할 시간을 충분히 줌)
    # 128 -> 2048 (Qwen 3는 말이 많습니다)
    GEN_MAX_TOKENS = 2048 

    print(f">>> [3/3] Qwen 3 추론 시작 (Thinking Mode)...")
    processed_data = []
    
    # [수정 포인트 2] 시스템 프롬프트: 한국어 출력 강제 & 포맷 명시
    system_prompt = """You are an expert Data Analyst AI.
    1. First, analyze the user's purchase history deeply inside <think> tags. (Reasoning must be in English or Korean).
    2. After thinking, output EXACTLY 3 Korean keywords that represent the user's persona.
    3. Do NOT output anything else after the keywords.
    
    Format example:
    <think>
    User bought diapers and toys... implies parenting context...
    </think>
    육아, 유아용품, 출산선물"""

    for user in tqdm(users, desc="Processing Users"):
        items = user['items']
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"구매 물품: {items}\n\n관심사는?"}
        ]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")

        outputs = model.generate(
            input_ids=inputs, 
            max_new_tokens=GEN_MAX_TOKENS, # [핵심] 여기서 끊기지 않게 늘림
            use_cache=True,
            temperature=0.6, # [팁] Thinking 모델은 창의성이 좀 필요해서 0.6 추천
            top_p=0.95
        )
        
        # 입력 토큰 이후의 생성된 토큰만 디코딩
        raw_output = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0].strip()

        # [수정 포인트 3] 파싱 로직 강화 (비상 착륙 기능)
        final_answer = ""
        thought_process = ""

        if "</think>" in raw_output:
            # 정상적으로 생각이 끝난 경우
            parts = raw_output.split("</think>")
            thought_process = parts[0].replace("<think>", "").strip()
            final_answer = parts[-1].strip()
        else:
            # 만약 2048 토큰으로도 모자라서 끊겼거나 태그가 꼬인 경우
            # 그냥 전체 텍스트를 저장하고 나중에 확인
            thought_process = raw_output
            final_answer = "추론_실패(Token_Limit)" 
            # (Qwen 3가 너무 길게 생각하면 이런 일이 생길 수 있음)

        # 공백이거나 이상한 문자 제거
        final_answer = final_answer.replace("\n", " ").strip()

        # 벡터화 (정답이 있을 때만)
        if final_answer and "추론_실패" not in final_answer:
            vector = embed_model.encode(final_answer).tolist()
        else:
            vector = [0.0] * 1024 # 더미 벡터 (에러 방지)

        processed_data.append({
            "user_id": user["user_id"],
            "items": items,
            "inferred_interests": final_answer,
            "thought_process": thought_process[:500] + "...", # 로그엔 앞부분만 저장
            "embedding": vector
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    print(f">>> ✅ [Qwen 3 완료] 결과 저장됨: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()