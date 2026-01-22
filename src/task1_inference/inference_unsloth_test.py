# src/task1_inference/inference_unsloth_test.py

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import json
import torch
from unsloth import FastLanguageModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- [최적화 설정] ---
# 1. 모델 ID: Unsloth가 제공하는 Qwen 3 4bit 버전
LLM_MODEL_ID = "unsloth/Qwen3-14B-unsloth-bnb-4bit"

# 2. VRAM 절약 핵심 설정
MAX_SEQ_LENGTH = 1024 # 14B 모델은 16GB VRAM에서 안전하게 1024부터 권장
DTYPE = None # Auto (Bfloat16 for RTX 40 series)
LOAD_IN_4BIT = True # 필수

EMBED_MODEL_ID = "BAAI/bge-m3"
INPUT_FILE = "./data/raw/dummy_users.json"
OUTPUT_FILE = "./data/processed/user_profiles_qwen3_result.json"

def main():
    print(f">>> [1/3] Unsloth Qwen 3 로드 중... ({LLM_MODEL_ID})")

    # 모델 & 토크나이저 로드
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = LLM_MODEL_ID,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
        # [Qwen 3 전용 최적화] 토큰 포지션 임베딩 설정
        fix_tokenizer = True,
    )

    # 추론 모드 (속도 2배)
    FastLanguageModel.for_inference(model)

    print(f">>> [2/3] 임베딩 모델 로드 중... ({EMBED_MODEL_ID})")
    embed_model = SentenceTransformer(EMBED_MODEL_ID, device="cuda")

    # 데이터 로드
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            users = json.load(f)
    except FileNotFoundError:
        print(f"❌ 오류: {INPUT_FILE} 파일이 없습니다.")
        return

    print(f">>> [3/3] Qwen 3 추론 시작 (Thinking Mode 테스트)...")
    processed_data = []
    
    # Qwen 3의 강점인 '생각하는' 프롬프트
    system_prompt = """당신은 유저의 숨겨진 니즈를 파악하는 AI입니다. 
    구매 물품을 분석하여, 표면적인 카테고리가 아닌 '구체적인 관심사' 3가지를 도출하세요."""

    for user in tqdm(users):
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

        # 생성
        outputs = model.generate(
            input_ids=inputs, 
            max_new_tokens=128,
            use_cache=True,
            temperature=0.3
        )
        
        generated_text = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0].strip()
        vector = embed_model.encode(generated_text).tolist()

        processed_data.append({
            "user_id": user["user_id"],
            "items": items,
            "inferred_interests": generated_text,
            "embedding": vector
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    print(f">>> ✅ [Qwen 3 완료] 결과 저장됨: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
