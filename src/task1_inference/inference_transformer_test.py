# src/task1_inference/inference_unsloth_test.py

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- [최적화 설정] ---
# 1. 모델 ID: Qwen 3 모델 (transformers 호환)
LLM_MODEL_ID = "Qwen/Qwen3-14B-Instruct"

# 2. VRAM 절약 핵심 설정
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

EMBED_MODEL_ID = "BAAI/bge-m3"
INPUT_FILE = "./data/raw/dummy_users.json"
OUTPUT_FILE = "./data/processed/user_profiles_qwen3_result.json"

def main():
    print(f">>> [1/3] Qwen 3 로드 중... ({LLM_MODEL_ID})")

    # 4bit 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)

    # 모델 로드 (4bit 양자화)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

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

    print(f">>> [완료] 결과 저장됨: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()