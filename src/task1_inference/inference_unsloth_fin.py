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
LLM_MODEL_ID = "unsloth/Qwen3-14B-unsloth-bnb-4bit" # 로컬에 이 모델이 있다고 가정

# 2. VRAM 절약 핵심 설정
# 14B 모델 기준: 1024(안전), 2048(권장/16GB 충분), 4096(위험)
MAX_SEQ_LENGTH = 2048 
DTYPE = None # RTX 40시리즈는 자동으로 bfloat16이 적용됨 (가장 빠름)
LOAD_IN_4BIT = True # 필수: 메모리를 1/4로 줄여줌

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
        fix_tokenizer = True,
    )

    FastLanguageModel.for_inference(model)

    print(f">>> [2/3] 임베딩 모델 로드 중... ({EMBED_MODEL_ID})")
    embed_model = SentenceTransformer(EMBED_MODEL_ID, device="cuda")

    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            users = json.load(f)
    except FileNotFoundError:
        print(f"❌ 오류: {INPUT_FILE} 파일이 없습니다.")
        return

    print(f">>> [3/3] Qwen 3 추론 시작 (Thinking Mode)...")
    processed_data = []
    
    # [수정] 모델 본능에 맞는 <think> 태그 사용 & 닫는 태그(</think>) 명시
    system_prompt = """You are an AI assistant.
    First, think deeply about the user's hidden interests inside <think> tags.
    Then, output exactly 3 Korean keywords that best represent their persona.
    Format: <think> reasoning process... </think> 키워드1, 키워드2, 키워드3"""

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

        outputs = model.generate(
            input_ids=inputs, 
            max_new_tokens=256, # 생각이 길어질 수 있으니 조금 늘림
            use_cache=True,
            temperature=0.3
        )
        
        raw_output = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0].strip()

        # [수정] 들여쓰기 안으로 넣음 & </think> 뒤에 있는 텍스트(정답)만 추출
        final_answer = raw_output
        thought_process = "..."

        if "</think>" in raw_output:
            parts = raw_output.split("</think>")
            thought_process = parts[0].replace("<think>", "").strip() # 생각 부분
            final_answer = parts[-1].strip() # 정답 부분
        
        # 로그에는 생각의 일부만 보여주고, 정답 출력
        # print(f"💡 AI 생각: {thought_process[:50]}...") 
        # print(f"✅ 최종 키워드: {final_answer}")

        vector = embed_model.encode(final_answer).tolist()

        processed_data.append({
            "user_id": user["user_id"],
            "items": items,
            "inferred_interests": final_answer,
            "thought_process": thought_process, # 나중에 분석용으로 저장해두면 좋음
            "embedding": vector
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    print(f">>> ✅ [Qwen 3 완료] 결과 저장됨: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()