import json
import os
from unsloth import FastLanguageModel
from tqdm import tqdm

# ==========================================
# ⚙️ 설정
# ==========================================
TARGET_COUNT = 1000  # 목표 데이터 개수
OUTPUT_FILE = "synthetic_data_1000.jsonl"
MODEL_NAME = "unsloth/Qwen3-14B-unsloth-bnb-4bit" # Teacher 모델

# ==========================================
# 1. 모델 로드 (Teacher)
# ==========================================
print(f"🔄 Teacher 모델 로딩 중... ({MODEL_NAME})")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# ==========================================
# 2. 이어하기 확인 (Resume)
# ==========================================
current_count = 0
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        current_count = sum(1 for line in f)
    print(f"📂 기존 파일 발견: {current_count}개 생성됨. 이어서 생성합니다.")

# ==========================================
# 3. 데이터 생성 루프
# ==========================================
prompt_template = """당신은 당근마켓의 AI 추천 데이터 생성기입니다.
다양한 유저 페르소나(나이, 직업, 관심사, 상황)를 가정하고, 그에 맞는 동네생활 모임 추천 데이터를 JSON 형식으로 생성하세요.
창의적이고 구체적인 상황을 설정해주세요.

[출력 형식]
{"instruction": "유저 정보를 바탕으로 적절한 모임을 추천하고 사유를 설명하세요.", "input": "특징: {유저특징}, 관심사: {관심사}", "output": "{\\"recommendation\\": \\"{추천모임명}\\", \\"reasoning\\": \\"{논리적인 추천 사유}\\"}"}
"""

print(f"🚀 데이터 생성 시작! (목표: {TARGET_COUNT}개)")

# 파일이 없으면 새로 만들고, 있으면 뒤에 이어붙임 ('a')
with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
    pbar = tqdm(total=TARGET_COUNT, initial=current_count)
    
    while current_count < TARGET_COUNT:
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_template}
            ]
            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")

            outputs = model.generate(
                inputs, 
                max_new_tokens=512,
                temperature=0.8, # 창의성 부여
                do_sample=True,
                use_cache=True
            )
            
            decoded = tokenizer.batch_decode(outputs)
            response = decoded[0].split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()
            
            # JSON 포맷 검증은 나중에 정제 단계에서 하므로 일단 저장
            if "{" in response and "}" in response:
                # 가짜 instruction 구조로 래핑하여 저장 (학습 포맷 맞춤)
                # 실제 모델 출력에서 JSON 부분만 파싱해서 저장하는 것이 좋으나, 
                # 여기서는 Teacher가 출력한 전체 텍스트를 raw 데이터로 저장.
                # 정제 스크립트가 이를 처리하도록 함.
                entry = {"raw_output": response} 
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush() # 즉시 디스크 기록
                
                current_count += 1
                pbar.update(1)
                
        except Exception as e:
            print(f"⚠️ 에러 발생 (건너뜀): {e}")
            continue

print("✨ 데이터 생성 완료!")