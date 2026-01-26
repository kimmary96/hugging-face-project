import torch
from unsloth import FastLanguageModel
import json
import os
import gc
from tqdm import tqdm
import random

# ==========================================
# 1. 설정 (Configuration) - 16GB VRAM 최적화
# ==========================================
MODEL_ID = "unsloth/Qwen3-14B-unsloth-bnb-4bit" # ※ Qwen3가 아직 HF에 없다면 2.5 사용 (User 입력명에 맞게 수정 가능)
MAX_SEQ_LENGTH = 1024 # 메모리 안전을 위해 1024로 제한 (피드백 반영)
OUTPUT_FILE = "synthetic_train_data.jsonl"
TARGET_COUNT = 100 # 우선 100개 생성 (Pilot Run)

# 랜덤성을 위한 시드 데이터 (다양성 확보용)
USER_SEGMENTS = ["20대 대학생", "30대 직장인", "40대 주부", "50대 자영업자", "새로 이사온 1인 가구"]
INTERESTS = ["운동/배드민턴", "독서/자기계발", "맛집탐방", "반려동물 산책", "동네 친구 만들기", "영어회화"]

# ==========================================
# 2. 모델 로드 (Memory Safe Loading)
# ==========================================
print(f"🔄 모델 로드 중... ({MODEL_ID})")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_ID,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None, # Auto detection (bfloat16 for RTX 40 series)
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) # 추론 모드 전환 (메모리 절약)
print("✅ 모델 로드 완료!")

# ==========================================
# 3. 프롬프트 템플릿 (System Instruction)
# ==========================================
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
당신은 당근마켓의 AI 추천 전문가입니다.
주어진 '유저 정보'를 분석하여, 해당 유저가 가장 관심을 가질만한 '모임(동네생활)'을 하나 추천하고, 그 논리적인 추천 사유를 작성하세요.
결과는 반드시 JSON 형식으로만 출력하세요.

### Input:
{}

### Response:
"""

# ==========================================
# 4. 데이터 생성 루프
# ==========================================
print(f"🚀 데이터 생성 시작 (목표: {TARGET_COUNT}개)")

# 기존 파일이 있으면 이어서 작성하지 않고, 덮어쓸지 확인 필요 (여기선 append 모드 사용)
mode = 'a' if os.path.exists(OUTPUT_FILE) else 'w'

with open(OUTPUT_FILE, mode, encoding='utf-8') as f:
    for i in tqdm(range(TARGET_COUNT)):
        
        # 4-1. 랜덤 입력 생성 (다양성 주입)
        user_profile = f"특징: {random.choice(USER_SEGMENTS)}, 관심사: {random.choice(INTERESTS)}"
        
        # 4-2. 프롬프트 구성
        inputs = tokenizer(
            [alpaca_prompt.format(user_profile)],
            return_tensors = "pt",
        ).to("cuda")

        # 4-3. 모델 추론 (Generation)
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens = 512, # 출력 길이 확보
                use_cache = True,
                temperature = 0.8, # 창의성 부여
            )
            
            # 4-4. 결과 디코딩 및 저장
            decoded_output = tokenizer.batch_decode(outputs)
            response_text = decoded_output[0].split("### Response:")[-1].replace("<|endoftext|>", "").strip()
            
            # JSONL 포맷으로 저장 (Instruction Tuning용 구조)
            data_entry = {
                "instruction": "유저 정보를 바탕으로 적절한 모임을 추천하고 사유를 설명하세요.",
                "input": user_profile,
                "output": response_text
            }
            f.write(json.dumps(data_entry, ensure_ascii=False) + "\n")
            
        except Exception as e:
            print(f"⚠️ Error at index {i}: {e}")

        # 4-5. 메모리 관리 (매우 중요 - VRAM 누수 방지)
        del inputs, outputs
        if i % 10 == 0: # 10번마다 캐시 비우기
            torch.cuda.empty_cache()
            gc.collect()

print(f"🎉 데이터 생성 완료! 파일 저장됨: {OUTPUT_FILE}")